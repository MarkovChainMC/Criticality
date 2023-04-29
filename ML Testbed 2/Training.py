import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf

# Load the Ising model data for temperature 1.0
ising_data = np.load('ising_data_T1.00.npz')
X = ising_data['inputs']
y = ising_data['targets']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train = np.reshape(X_train, (-1, 1))
X_test = np.reshape(X_test, (-1, 1))
y_train = np.reshape(y_train, (-1, 1))
y_test = np.reshape(y_test, (-1, 1))


# Define the model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam', loss='mse', metrics=['mae', 'mse'])

# Train the model
history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_test, y_test))

# Evaluate the model
loss, mae, mse = model.evaluate(X_test, y_test)

# Make predictions
y_pred = model.predict(X_test)

# Calculate R2 score
from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_pred)
print('R2 score:', r2)
