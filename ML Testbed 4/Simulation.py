import numpy as np
from sklearn.linear_model import LinearRegression
from tensorflow import keras
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

def f(h, J):
    # Construct the transverse Ising model Hamiltonian
    H = np.array([[J, h], [h, -J]])
    # Compute the eigenvalues and eigenvectors of the Hamiltonian
    eigvals, eigvecs = np.linalg.eigh(H)
    # Determine the critical exponents based on the eigenvalues
    if eigvals[0] == eigvals[1]:
        nu = 1
        beta = 1/2
        gamma = 1
    else:
        nu = 1/2
        beta = (eigvals[1] - eigvals[0]) / 2
        gamma = (eigvals[1] - eigvals[0])**(-1)
    return np.array([nu, beta, gamma])


# Define the neural network model
def build_model(input_shape):
    model = keras.Sequential([
        keras.layers.Dense(32, activation='relu', input_shape=input_shape),
        keras.layers.Dense(32, activation='relu'),
        keras.layers.Dense(3) # Output layer with 3 nodes for the critical exponents
    ])
    optimizer = keras.optimizers.Adam(learning_rate=0.001)
    model.compile(loss='mse', optimizer=optimizer)
    return model

# Generate some training data
n_samples = 500000
x_train = np.random.uniform(-1, 1, size=(n_samples, 2))
y_train = np.zeros((n_samples, 3))
for i in range(n_samples):
    h, J = x_train[i]
    # Compute the critical exponents using some function f
    y_train[i] = f(h, J)

# Build and train the neural network model
model = build_model(input_shape=(2,))
model.fit(x_train, y_train, epochs=50, batch_size=32)

# Use the trained model to predict critical exponents for new parameter values
x_test = np.array([[0.5, 0.5], [0.7, -0.3], [-0.8, 0.2]])
y_pred = model.predict(x_test)

print('Predicted critical exponents:')
print(y_pred)

# Generate the training data
h_train = np.random.uniform(low=-1, high=1, size=(1000,))
J_train = np.random.uniform(low=-1, high=1, size=(1000,))
X_train = np.vstack((h_train, J_train)).T
y_train = np.array([f(h, J) for h, J in X_train])

# Train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Generate the test data
h_test = np.random.uniform(low=-1, high=1, size=(100,))
J_test = np.random.uniform(low=-1, high=1, size=(100,))
X_test = np.vstack((h_test, J_test)).T
y_test = np.array([f(h, J) for h, J in X_test])

# Predict the critical exponents for the test data using the trained model
y_pred = model.predict(X_test)

# Compute the R-squared value for the model
r2 = r2_score(y_test, y_pred)

print("R-squared value:", r2)

