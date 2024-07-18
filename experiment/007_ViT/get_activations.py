import timm
import torch
from torchvision import transforms

from PIL import Image

import numpy as np
from tqdm import tqdm

def load_model():
    model = timm.create_model('vit_base_patch16_224', pretrained=True)
    model.eval()

    print('Loaded model!')

    return model

def get_activation(model, layer_num: int, features: 'torch.tensor'):
    # Construct Hook
    intermediate_outputs = []
    def hook(module, input, output):
        intermediate_outputs.append(output)
    model.blocks[layer_num].register_forward_hook(hook)

    # Run Model-- data goes to hook
    with torch.no_grad():
        output = model(images)
        del output

    # Return output of hidden layer
    return intermediate_outputs[0]

def get_all_activations(model, features: 'torch.tensor') -> 'torch.tensor':
    layer_ids = range(len(model.blocks))
    activations = [get_activation(model, layer_num=layer_num, features=features) for layer_num in layer_ids]
    return torch.stack(activations)


def load_dataset():
    path = '/vast/xj2173/diffeo/imagenet'
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor()
    ])
    dataset = torchvision.datasets.ImageNet(path, 
                                            split='val', 
                                            transform=transform)
    return dataset

def diffeo_images(dataset: 'torchvision.dataset') -> 'torch.tensor':
    '''
    Load in set of diffeomorphisms ${g_i}_i$
    '''
    pixels = 224
    number_of_diffeo = 20
    image_id = 0

    feature, _ = dataset[image_id]
    features.to(device)
    feature = feature.unsqueeze(0).expand(number_of_diffeo, -1, -1, -1)

    sparse_diffeos = sparse_diffeo_container(pixels, pixels)
    diffeo_strength_list = [0.01, 0.05, 0.1, 0.125, 0.15, 0.175, 0.2, 0.225, 0.25, 0.275, 0.3, 0.35, 0.4, 0.45, 0.5]
    for strength in diffeo_strength_list:
        sparse_diffeos.sparse_AB_append(4, 4, 3, strength, number_of_diffeo)
    sparse_diffeos.get_all_grid()
    sparse_diffeos.to(device)

    return sparse_diffeos(features)

    



def main():

    model = load_model()
    dataset = load_dataset()
    diffeos = load_diffeos(dataset, )

    transformed_images = diffeo_images(images=features, diffeos=diffeos)
    print(transformed_images.shape)
    results = [get_all_activations(model, features=image) for image in tqdm(transformed_images)]

    torch.save(results, 'ViT_Activations_Over_Diffeos.pt')
        





if __name__ == '__main__':
    main()