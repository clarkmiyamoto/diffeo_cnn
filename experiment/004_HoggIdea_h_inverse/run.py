import argparse
import concurrent

import numpy as np
import torch
import torch.optim as optim

from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

from tqdm import tqdm
import matplotlib.pyplot as plt

def LOAD_gInv_N_g_I(target_pic, number):
    # Load N(g * I)
    data_dir = '/vast/xj2173/diffeo/data/all_cnn_layers/'
    data_name = [data_dir + f'15-100-4-4-3-224-224_image-{target_pic}_activation_layer-{number}.pt']
    data = [torch.load(file_name, map_location='cpu') for file_name in tqdm(data_name)]
    data = torch.stack(data, dim=0)
    res_of_layer = data.shape[-1]

    # Load g^{-1}_{naive}
    data_dir = '/vast/xj2173/diffeo/data/all_cnn_layers/'
    file_name = data_dir + '15-100-4-4-3-224-224_inv_grid_sample.pt'
    inv_diffeos_maps = torch.load(file_name, map_location='cpu')
    inv_diffeos_maps = torch.stack(inv_diffeos_maps)
    inv_diffeos_maps = inv_diffeos_maps.reshape(15 * 100, 224, 224, 2)
    inv_diffeos_maps = inv_diffeos_maps.permute(0, 3, 1, 2)
    inv_diffeos_maps = torch.nn.functional.interpolate(inv_diffeos_maps, size=(res_of_layer, res_of_layer), mode='bilinear', align_corners=False)
    inv_diffeos_maps = inv_diffeos_maps.permute(0, 2, 3, 1)

    # Apply g^{-1}_{naive}
    mode = 'bilinear'
    data_inv = [torch.nn.functional.grid_sample(pic_data, inv_diffeos_maps, mode = mode) for pic_data in tqdm(data)]
    data_inv = torch.stack(data_inv, dim=0)

    del inv_diffeos_maps, data

    return data_inv

def LOAD_N_I(target_pic, number):
    data_dir = '/vast/xj2173/diffeo/data/reference/'
    data_name = [data_dir + f'val_image-{target_pic}_activation_layer-{number}.pt']
    ref_data = [torch.load(file_name, map_location='cpu').squeeze(0) for file_name in tqdm(data_name)]
    ref_data = torch.stack(ref_data)

    return ref_data



def learn_h_inv(feature, label):
    # Define the loss function
    def loss(A, features, labels):
        predictions = torch.einsum('ab,axy->bxy', A, features)
        return torch.mean((predictions - labels) ** 2)


    # Initialize A with normal distribution
    features_shape = (len(feature), len(feature))  # Replace with the actual shape of your features
    A = torch.randn(features_shape, requires_grad=True)

    # Hyperparameters
    learning_rate = 0.0001
    num_iterations = 100000
    threshold = 1e-6  # Define a threshold for change in loss
    patience = 100  # Define the number of iterations to wait before stopping if no improvement
    counter = 0  # Initialize a counter to track the number of iterations without significant change
    previous_loss = float('inf')  # Initialize the previous loss to a high value


    # Define the optimizer
    optimizer = optim.Adam([A], lr=learning_rate)

    for i in range(num_iterations):
        optimizer.zero_grad()  # Zero the gradients before each iteration
        current_loss = loss(A, feature, label)
        current_loss.backward()  # Backpropagate to compute gradients
        optimizer.step()  # Update parameters
        
        if i % 500 == 0:
            print(f"Iteration {i}: Loss = {current_loss.item()}")
        
        # Check for early stopping
        if abs(previous_loss - current_loss.item()) < threshold:
            counter += 1
            if counter >= patience:
                print(f"Early stopping at iteration {i}: Loss = {current_loss.item()}")
                break
        else:
            counter = 0  # Reset counter if there is significant change

        previous_loss = current_loss.item()  # Update previous loss
    
    return A # This is h_inv

def run_simulation(layer_idx, diffeo_idx):
    feature = data_inv[layer_idx, diffeo_idx, :, :, :]
    label = ref_data[layer_idx, :, :, :]

    A = learn_h_inv(feature, label)

    return diffeo_idx, A








if __name__ == "__main__":
    ### Parse Arguments in run.py calling
    parser = argparse.ArgumentParser(description="Hogg's Idea script.")
    parser.add_argument("--layer", type=int, help="Layers of neural network")
    parser.add_argument("--picture", type=int, help="Picture id")
    parser.add_argument("--num_cores", type=int, help="Number of CPU cores")

    
    # Parse the arguments
    args = parser.parse_args()
    
    # Parameters for simulation
    target_pic = f"{args.picture:04}"
    number = f"{args.layer:02}"
    num_workers = args.num_cores
    data_inv = LOAD_gInv_N_g_I(target_pic, number) # Load g^-1 N(g * I)
    ref_data = LOAD_N_I(target_pic, number)        # Load N(I)

    if len(data_inv.shape) != 5:
        raise ValueError('data_inv has wrong shape')

    # Using ThreadPoolExecutor for threads or ProcessPoolExecutor for processes
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        # Submit tasks for each diffeo_idx
        futures = [executor.submit(process_diffeo, diffeo_idx) for diffeo_idx in range(0, int(data_inv.shape[1]))]

        # Retrieve results as they complete
        results = []
        for future in concurrent.futures.as_completed(futures):
            results.append(future.result())

    # Sort results to preserve order (optional, if needed)
    results.sort(key=lambda x: x[0])

    # Save results using PyTorch
    torch.save(results, f'hInv_Pic{target_pic}_Layer{number}.pth')






