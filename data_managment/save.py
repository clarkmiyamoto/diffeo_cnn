import os
import inspect

import jax.numpy as jnp
import numpy as np


# Get directory of save.py
script_dir = os.path.dirname(os.path.abspath(__file__))

# Get the caller script's directory
frame = inspect.stack()[1]
caller_script_path = frame.filename
caller_directory = os.path.dirname(os.path.abspath(caller_script_path))
caller_dir_name = os.path.basename(caller_directory)

# Get caller script's name, without extension
caller_script_name = os.path.basename(caller_script_path).split('.')[0]

def save_diffeo(name: str, 
                allow_pickle: bool = True):
    diffeo_path = "../data/diffeo/"
    path = os.path.join(script_dir, 
                        diffeo_path, 
                        caller_dir_name, 
                        caller_script_name)
    jnp.save(os.path.join(path, name), allow_pickle=allow_pickle)

def save_activation(name:str,
                    allow_pickle: bool = True):
    activation_path = "../data/activation/"
    path = os.path.join(script_dir, 
                        activation_path, 
                        caller_dir_name, 
                        caller_script_name)
    np.save(os.path.join(path, name), allow_pickle=allow_pickle)