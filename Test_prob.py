#Test the model to get a vector of probabilities

import numpy as np
from keras.models import load_model

# Load the saved model
model = load_model('Ising_training.h5')

# Generate a new spin configuration (in this case, randomly)
new_spin_config = np.random.choice([-1, 1], size=(10, 10))

# Reshape and normalize the new spin configuration
new_spin_config = new_spin_config.reshape(1, 10, 10, 1)
new_spin_config = new_spin_config.astype('float32')
new_spin_config /= 4.0

# Make a prediction on the new spin configuration
prediction = model.predict(new_spin_config)

# Print the predicted class
print(prediction)
