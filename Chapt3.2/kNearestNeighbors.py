import numpy as np

from sklearn.datasets import fetch_openml
from sklearn.neighbors import KNeighborsClassifier

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

# Creates a boolean target for digits that are above 7
# Data Structure: [False, False, True,...]
y_train_large = (y_train >= 7)

# Creates a boolean target for digits that are odd
# Data Structure: [False, False, True,...]
y_train_odd = (y_train % 2 == 1)

# Combine the large digit array and the odd digit array
# Data Structure:
# array([
# [False, True],
# [False, False],
# [False, False],
# ...
# ])
# As you can see this has multiple labels
y_multilabel = np.c_[y_train_large, y_train_odd]

# Initialize the model
knn_clf = KNeighborsClassifier()

# Train the model on the multiple label
knn_clf.fit(X_train, y_multilabel)

# Make a prediction
zero_digit = X[1]

# Below returns: array([[False, False]])
# Zero is not large or odd
knn_clf.predict([zero_digit])

# Getting F1 Score
from sklearn.metrics import f1_score
from sklearn.model_selection import cross_val_predict

# Get's the predictions in a boolean array. True if it predicts as a 5 false otherwise
y_train_knn_pred = cross_val_predict(knn_clf, X_train, y_multilabel, cv=3)

f1_score(y_multilabel, y_train_knn_pred, average="macro")