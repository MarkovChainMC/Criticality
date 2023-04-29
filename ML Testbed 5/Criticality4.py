import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD
from matplotlib import pyplot as plt

# Define a function to generate a random Gaussian neural network
def generate_gaussian_network(n_layers, n_neurons):
    model = Sequential()
    for i in range(n_layers):
        model.add(Dense(n_neurons, input_dim=1, activation='relu',
                        kernel_initializer='random_normal',
                        bias_initializer='zeros'))
    model.add(Dense(1, activation='sigmoid',
                    kernel_initializer='random_normal',
                    bias_initializer='zeros'))
    return model

# Define a function to generate the training data
def generate_data():
    x = np.random.uniform(-10, 10, 1000)
    y = np.sin(x) / x
    return x, y

# Evaluate the performance of the model at different learning rates
def evaluate_learning_rates(n_layers, n_neurons):
    learning_rates = [0.001, 0.01, 0.1, 1.0]
    n_epochs = 50
    x_train, y_train = generate_data()
    x_test, y_test = generate_data()
    for lr in learning_rates:
        model = generate_gaussian_network(n_layers, n_neurons)
        model.compile(loss='mse', optimizer=SGD(lr=lr))
        history = model.fit(x_train, y_train, epochs=n_epochs, verbose=0,
                            validation_data=(x_test, y_test))
        plt.plot(history.history['val_loss'], label='lr={}'.format(lr))
    plt.legend()
    plt.title('Validation Loss vs. Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Validation Loss')
    plt.show()

# Call the function to evaluate the performance of the model
evaluate_learning_rates(5,10)
