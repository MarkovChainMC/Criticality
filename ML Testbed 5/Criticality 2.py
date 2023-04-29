import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ortho_group

def relu(x):
    return np.maximum(x, 0)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def generate_orthogonal_network(num_layers, layer_size):
    weights = []
    for i in range(num_layers):
        weight_matrix = ortho_group.rvs(layer_size)
        weights.append(weight_matrix)
    return weights

def compute_eigenvalue_spectrum(weights, activation_function):
    num_layers = len(weights)
    eigvals = []
    for i in range(num_layers):
        if i == 0:
            x = weights[i]
        else:
            x = np.dot(weights[i], x)
        if activation_function == "relu":
            x = relu(x)
        elif activation_function == "sigmoid":
            x = sigmoid(x)
        eigval = np.max(np.abs(np.linalg.eigvals(x)))
        eigvals.append(eigval)
    return eigvals

# Generate random orthogonal network
num_layers = 50
layer_size = 100
weights = generate_orthogonal_network(num_layers, layer_size)

# Compute eigenvalue spectrum for ReLU and sigmoid activation functions
eigvals_relu = compute_eigenvalue_spectrum(weights, "relu")
eigvals_sigmoid = compute_eigenvalue_spectrum(weights, "sigmoid")

# Plot eigenvalue spectra
plt.plot(np.arange(1, num_layers+1), eigvals_relu, label="ReLU")
plt.plot(np.arange(1, num_layers+1), eigvals_sigmoid, label="sigmoid")
plt.xlabel("Layer")
plt.ylabel("Maximum absolute eigenvalue")
plt.legend()
plt.show()
