# Creating the training and testing dataset
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

# load the data and labels from the file
data = np.load('ising_data.npz')
X = data['data']
y = data['labels']

# split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Reshape the data for input into the CNN
X_train = X_train.reshape(-1, 10, 10, 1)
X_test = X_test.reshape(-1, 10, 10, 1)

# Normalize the data
X_train = X_train / 2 + 0.5
X_test = X_test / 2 + 0.5

# Define the CNN architecture
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=(10, 10, 1)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(3, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=20, validation_data=(X_test, y_test))

# Save the trained model
model.save('Ising_training.h5')