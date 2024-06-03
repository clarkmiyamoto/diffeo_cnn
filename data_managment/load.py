import os
import inspect

import jax.numpy as jnp
import numpy as np

# Get directory of load.py
script_dir = os.path.dirname(os.path.abspath(__file__))

def load_diffeo(name: str, 
                allow_pickle: bool = True):
    diffeo_path = "../data/diffeo/"
    path = os.path.join(script_dir, 
                        diffeo_path)
    data = jnp.save(os.path.join(path, name), allow_pickle=allow_pickle)
    return data

def load_activation(name:str,
                    allow_pickle: bool = True):
    activation_path = "../data/activation/"
    path = os.path.join(script_dir, 
                        activation_path)
    np.save(os.path.join(path, name), allow_pickle=allow_pickle)

