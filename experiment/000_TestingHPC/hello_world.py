import os
import sys

# Get current file directory & add `diffeo_CNN` to global path
current_dir = os.path.dirname(os.path.abspath(__file__))
diffeo_cnn_path = os.path.abspath(os.path.join(current_dir, '../../..'))
sys.path.append(diffeo_cnn_path)

from data_managment.directory import pwd

print(pwd())


from data_managment.save import save_activation
import numpy as np
array = np.array([1,2,3,4,5])
save_activation(name='hello_world.npz', arr=array)
