import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras

# Define the Ising model simulation function
def simulate_ising(beta, L=16):
    state = np.random.choice([-1, 1], size=(L, L, L, L))
    for _ in range(10):
        for x in range(L):
            for y in range(L):
                for z in range(L):
                    for w in range(L):
                        neighbors = state[(x+1)%L,y,z,w] + state[x,(y+1)%L,z,w] + state[x,y,(z+1)%L,w] + state[x,y,z,(w+1)%L] + state[(x-1)%L,y,z,w] + state[x,(y-1)%L,z,w] + state[x,y,(z-1)%L,w] + state[x,y,z,(w-1)%L]
                        dE = 2*state[x,y,z,w]*neighbors
                        if dE < 0 or np.random.rand() < np.exp(-beta*dE):
                            state[x,y,z,w] *= -1
    return state

# Define the temperature range
T = np.linspace(1,2.05,num=5)

# Generate Ising model data for each temperature
data = []
for beta in 1/T:
    state = simulate_ising(beta)
    data.append(state.reshape(-1, 16, 16, 16))

# Define the CNN model
model = keras.Sequential([
    keras.layers.Conv3D(32, kernel_size=(3, 3, 3), activation='relu', input_shape=(16, 16, 16, 1)),
    keras.layers.MaxPooling3D(pool_size=(2, 2, 2)),
    keras.layers.Conv3D(64, kernel_size=(3, 3, 3), activation='relu'),
    keras.layers.MaxPooling3D(pool_size=(2, 2, 2)),
    keras.layers.Flatten(),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model for each temperature
history = []
for i, beta in enumerate(1/T):
    print("Training model for beta =", beta)
    X_train = data[i][:, :, :, :, np.newaxis]
    y_train = np.zeros(len(X_train))
    y_train[len(X_train)//2:] = 1
    history.append(model.fit(X_train, y_train, epochs=20, validation_split=0.2))

# Plot the training and validation accuracy for each temperature
fig, axs = plt.subplots(nrows=1, ncols=5, figsize=(20, 15))
for i, ax in enumerate(axs.flatten()):
    ax.plot(history[i].history['accuracy'], label='Training Accuracy')
    ax.plot(history[i].history['val_accuracy'], label='Validation Accuracy')
    ax.set_title("beta = {:.2f}".format(1/T[i]))
    ax.legend()
plt.tight_layout()
plt.show()
