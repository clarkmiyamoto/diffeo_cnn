import torch
from torch import nn
import math

class BiasOnly_gridInv(nn.Module):
    def __init__(self, grid):
        super(BiasOnly_gridInv, self).__init__()
        self.grid = grid
        self.bias = nn.Parameter(torch.zeros_like(grid))
    def forward(self):
        return self.grid + self.bias
    
def find_inv_grid(flow_grid, mode ='bilinear', learning_rate = 0.001, epochs = 10000, early_stopping = True):
  x_length, y_length, _ = flow_grid.squeeze().shape
  x = torch.linspace(-1, 1, steps = x_length)
  y = torch.linspace(-1, 1, steps = y_length)
  X, Y = torch.meshgrid(x, y, indexing='ij')
  reference = torch.stack((X, Y, X * Y, torch.cos(2*math.pi*X) * torch.cos(2*math.pi*Y)), dim=0).unsqueeze(0)
    
  find_inv_model = BiasOnly_gridInv(torch.stack((Y, X), dim=-1).unsqueeze(0))
  loss_fn = nn.MSELoss()
  optimizer = torch.optim.Adam(find_inv_model.parameters(), lr = learning_rate)

  num_epochs = epochs
  loss_hist = []
  min_loss = 1e30
  early_stopping_count = 0
  for epoch in range(num_epochs):
    optimizer.zero_grad()
    output = find_inv_model()
    #distort = torch.nn.functional.grid_sample(reference, flow_grid, mode = mode)
    inv_distort = torch.nn.functional.grid_sample(reference, output, mode = mode)
    #restored_left  = torch.nn.functional.grid_sample(distort, output, mode = mode)
    restored_right = torch.nn.functional.grid_sample(inv_distort, flow_grid, mode = mode)
    #left_loss = loss_fn(reference, restored_left)
    right_loss = loss_fn(reference, restored_right)
    loss = right_loss #+ left_loss #+ (torch.exp(torch.abs(left_loss-right_loss)**2) - 1)
    #loss =  left_loss + right_loss
    loss.backward()
    optimizer.step()
    if (epoch + 1) % 500 == 0:
          loss_hist.append(loss.item())
          if loss_hist[-1]/min_loss >= 0.95: early_stopping_count += 1
          if loss_hist[-1] < min_loss: min_loss = loss_hist[-1]
    if early_stopping and early_stopping_count >=5: break

  with torch.no_grad():
    flow_grid_inverse_neural = find_inv_model().detach().clone()

  return flow_grid_inverse_neural, loss_hist