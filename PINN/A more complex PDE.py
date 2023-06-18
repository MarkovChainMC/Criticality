import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


# Define your loss function
def loss_fn(model, x_train, y_train):
    with tf.GradientTape() as tape:
        tape.watch(model.trainable_variables)
        y_pred = model(x_train)
        loss = tf.reduce_mean(tf.square(y_pred - y_train))
    gradients = tape.gradient(loss, model.trainable_variables)
    return loss, gradients

# Create your model and define the architecture
model = tf.keras.Sequential([
    tf.keras.layers.Dense(32, activation='relu', input_shape=(1,)),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1)
])

# Generate your training data
np.random.seed(0)
x_train = np.random.uniform(-1, 1, (100, 1))
y_train =10*(x_train)**10+9*(x_train)**9+np.exp(-x_train**2) + np.random.normal(0, 0.1, (100, 1))

# Convert the training data to tensors
x_train = tf.convert_to_tensor(x_train, dtype=tf.float32)
y_train = tf.convert_to_tensor(y_train, dtype=tf.float32)

# Define your optimizer
optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)

# Training loop
epochs = 1000
for epoch in range(epochs):
    loss, gradients = loss_fn(model, x_train, y_train)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    if epoch % 100 == 0:
        print(f"Epoch {epoch}: Loss = {loss:.4f}")

# Generate some test data for plotting
x_test_plot = np.linspace(-1, 1, 100).reshape((-1, 1))
x_test_plot = tf.convert_to_tensor(x_test_plot, dtype=tf.float32)

# Make predictions on the test data
y_pred_plot = model(x_test_plot)

# Convert tensors back to numpy arrays for plotting
x_train_plot = x_train.numpy()
y_train_plot = y_train.numpy()
x_test_plot = x_test_plot.numpy()
y_pred_plot = y_pred_plot.numpy()

# Plot the training data and the predictions
plt.figure(figsize=(8, 6))
plt.scatter(x_train_plot, y_train_plot, color='blue', label='Training Data')
plt.plot(x_test_plot, y_pred_plot, color='red', label='Model Predictions')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Training Data and Model Predictions')
plt.legend()
plt.grid(True)
plt.show()