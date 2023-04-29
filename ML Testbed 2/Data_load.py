import numpy as np

# Load the Ising model data for temperature 1.0
ising_data = np.load('ising_data_T1.00.npz')
X = ising_data['inputs']
y = ising_data['targets']




