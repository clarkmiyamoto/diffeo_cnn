# Image Processing
import numpy as np
import jax.numpy as jnp
import jax

from PIL import Image
import cv2 as cv

# Data Structure
import pandas as pd

# Etc
import warnings
import os

def sin_distortion(x_length: int,
                   y_length: int,
                   A_nm: jnp.array) -> jnp.array:
  """
  Sin distortion for `cv2.remap` function.

  Args:
  - x_length (int): Length of x-axis of image.
  - y_length (int): Length of y-axis of image.
  - A_nm (np.array): Square matrix of coefficents. Sets size of cut off

  Returns:
  (np.array): Size `x_length` * `y_length`.
  """
  if A_nm.shape[0] != A_nm.shape[1]:
    raise ValueError('A_nm must be square matrix.')

  # Create Coordinates
  x = jnp.arange(x_length)
  y = jnp.arange(y_length)
  X, Y = jnp.meshgrid(x, y)

  # Create Diffeo
  x_pert = jnp.linspace(0,1, x_length)
  y_pert = jnp.linspace(0,1, y_length)

  n = jnp.arange(1,A_nm.shape[0] + 1)
  x_basis = jnp.sin(jnp.pi * jnp.outer(n, x_pert))
  y_basis = jnp.sin(jnp.pi * jnp.transpose(jnp.outer(n, y_pert)))

  perturbation =  y_basis @ A_nm @ x_basis

  x_map = X + perturbation
  y_map = Y + perturbation

  return np.array(x_map), np.array(y_map)

def apply_transformation(image_tensor,
                         A_nm: jnp.array,
                         interpolation_type =cv.INTER_LINEAR):
  """
  Wrapper of `sin_distort`. Gets torch.tensor and
  returns the distorted torch.tensor according to $A_nm$.

  Args:
    image_tensor (torch.tensor): Inputted image.
    A_nm (jnp.array): Characterizes diffeo according to `sin_distort`
    interpolation_type (cv.):

  Returns
    image_tensor_deformed (torch.tensor): Diffeo applied to `image_tensor`.
  """
  # Convert to numpy array
  image_numpy = image_tensor.cpu().detach().numpy()

  # Create deformation map
  x_length, y_length = tuple(image_tensor.shape[1:3])
  x_map, y_map = sin_distortion(x_length, y_length, A_nm)

  # Apply Deformation per channel
  deformed_per_channel = [cv.remap(channel, x_map, y_map, interpolation_type) for channel in image_numpy]
  image_numpy_deformed = np.stack(deformed_per_channel, axis=0)

  # Convert back to tensor
  image_tensor_deformed = torch.from_numpy(image_numpy_deformed)

  return image_tensor_deformed