import numpy as np

data = np.load('ising_data.npz')
print(data.files)
# Access and print the 'X' array
X = data['data']
print(X)

# Access and print the 'y' array
y = data['labels']
print(y)