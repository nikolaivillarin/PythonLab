import numpy as np

from sklearn.datasets import fetch_openml
from sklearn.linear_model import SGDClassifier

mnist = fetch_openml('mnist_784', version=1)

# X - represents the data
# y - represents the target values
X, y = mnist["data"], mnist["target"]

# Convert y to int since it's currently a string
y = y.astype(np.uint8)

# Split the data to training set and test set
# MNIST data is already shuffled with the proper folds.
# The first 60,000 records are for training
# The last 10,000 records are for testing
X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]

# Binary classifier convert the target values for both
# training set and test set to an array of booleans
y_train_0 = (y_train == 0)
y_test_0 = (y_test == 0)

# Instantiate the Model
sgd_clf = SGDClassifier(random_state=42)

# Train the model
sgd_clf.fit(X_train, y_train_0)

# Test the model by making a prediction
zero_digit = X[1]
sgd_clf.predict([zero_digit])