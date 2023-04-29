import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import  train_test_split
from sklearn.metrics import r2_score

x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
y = np.array([2, 4, 6, 8, 10, 12, 14, 16, 18, 20])
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=42)
X_train = np.reshape(X_train, (-1, 1))
X_test = np.reshape(X_test, (-1, 1))
y_train = np.reshape(y_train, (-1, 1))
y_test = np.reshape(y_test, (-1, 1))


# create a Linear Regression model
model = LinearRegression()

# fit the model with training data
model.fit(X_train, y_train)

# predict target values for test data
y_pred = model.predict(X_test)

# evaluate the model using R-squared score

r2 = r2_score(y_test, y_pred)
print("R-squared score:", r2)
print(y_test)