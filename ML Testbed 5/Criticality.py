import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification

X, y = make_classification(n_samples=1000, n_features=10, n_classes=2, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = Sequential()
model.add(Dense(64, input_dim=10, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
sgd = SGD(lr=0.01, momentum=0.9)
model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])
learning_rates = [0.0001, 0.001, 0.01, 0.1, 1]

fig, axs = plt.subplots(1, len(learning_rates), figsize=(20, 5))

for i, lr in enumerate(learning_rates):
    sgd = SGD(lr=lr, momentum=0.9)
    model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])
    history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=50, batch_size=32, verbose=0)
    axs[i].plot(history.history['accuracy'], label='Training Accuracy')
    axs[i].plot(history.history['val_accuracy'], label='Validation Accuracy')
    axs[i].set_title('Learning Rate = ' + str(lr))
    axs[i].set_xlabel('Epoch')
    axs[i].set_ylabel('Accuracy')
    axs[i].legend()

plt.show()