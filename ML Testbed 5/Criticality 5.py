import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD

# Define the largest eigenvalue of weight matrix as a function of its size
def largest_eigenvalue(size, alpha):
    return alpha * np.sqrt(size)

# Generate random Gaussian neural network with given layer sizes
def generate_network(layer_sizes, alpha):
    network = Sequential()
    for i in range(1, len(layer_sizes)):
        size_in = layer_sizes[i-1]
        size_out = layer_sizes[i]
        weight_matrix = np.random.normal(loc=0.0, scale=1.0/np.sqrt(size_in), size=(size_in, size_out))
        eigenvalue = largest_eigenvalue(size_in, alpha)
        weight_matrix *= eigenvalue / np.max(np.abs(np.linalg.eig(weight_matrix)[0]))
        bias_vector = np.zeros(size_out)
        network.add(Dense(size_out, input_shape=(size_in,), activation='relu', kernel_initializer=lambda shape, dtype: weight_matrix, bias_initializer=lambda shape, dtype: bias_vector))
    return network

# Define the critical value of the parameter alpha
alpha_c = 1.0

# Define the layer sizes of the DNN
layer_sizes = [100, 100, 100]

# Generate random Gaussian neural networks with different alpha values
alphas = [0.1, 0.5, 1.0, 1.5, 2.0]
networks = [generate_network(layer_sizes, alpha) for alpha in alphas]

# Compile and train the DNNs on a simple classification problem
x_train = np.random.normal(loc=0.0, scale=1.0, size=(1000, layer_sizes[0]))
y_train = (x_train[:, 0] > 0).astype(float)
for network in networks:
    network.compile(loss='binary_crossentropy', optimizer=SGD(lr=0.01), metrics=['accuracy'])
    history = network.fit(x_train, y_train, validation_split=0.2, epochs=100, batch_size=16, verbose=0)

# Plot the largest eigenvalue spectra of the weight matrices for the DNNs
for i in range(len(networks)):
    eigenvalues = np.linalg.eigvals(networks[i].layers[0].get_weights()[0])
    plt.plot(np.arange(1, len(eigenvalues)+1), np.sort(np.abs(eigenvalues))[::-1], label=r'$\alpha={}$'.format(alphas[i]))
plt.legend()
plt.xlabel('Eigenvalue index')
plt.ylabel('Magnitude')
plt.show()
