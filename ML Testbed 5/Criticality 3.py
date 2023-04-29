import numpy as np
from tensorflow import keras

# define the neural network architecture
model = keras.Sequential([
    keras.layers.Dense(128, activation='relu', input_shape=(784,)),
    keras.layers.Dense(10, activation='softmax')
])

# compile the model
model.compile(optimizer=keras.optimizers.SGD(learning_rate=0.01),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# generate random input data
x_train = np.random.rand(1000, 784)
y_train = np.random.randint(0, 10, 1000)

# train the model
epochs = 50
histories = []
for lr in [0.001, 0.01, 0.1]:
    model.optimizer.learning_rate = lr
    history = model.fit(x_train, y_train, epochs=epochs, validation_split=0.2)
    histories.append(history)

# plot the validation accuracy vs learning rate
import matplotlib.pyplot as plt

for history, lr in zip(histories, [0.001, 0.01, 0.1]):
    plt.plot(history.history['val_accuracy'], label=f'lr={lr}')

plt.xlabel('Epoch')
plt.ylabel('Validation Accuracy')
plt.legend()
plt.show()
