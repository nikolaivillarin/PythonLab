import numpy as np
import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn.linear_model import LogisticRegression

iris = datasets.load_iris()

# First part of [:, 3:]
# : - get's all the first level items
# 3: goes into the second level and gets only the petal width
X = iris["data"][:, 3:]

# Convert the boolean values to int (0 or 1)
y = (iris["target"] == 2).astype(np.int)

# Instantiate Model
log_reg = LogisticRegression()

# Train the Model
log_reg.fit(X, y)

# Create an array of evenly spaced numbers from 0 - 3 with 
# 1000 iterations. Data Structure: [0.0, 0.003, 0.006, 0.009..., 3.0]
# These numbers represent the Petal Width in cm that we will be testing
X_new = np.linspace(0, 3, 1000)

# Reshape changes the data structure to: [[0.0], [0.003], [0.006], [0.009]..., [3.0]]
# First parameter specifies how many item should the array have. In this example it's set
# to -1 which is basically not specified.
# Second parameter is how many items per array. In this case it's set to 1
X_new = X_new.reshape(-1, 1)

# Getting the class probabilities
y_proba = log_reg.predict_proba(X_new)

# Plot to see the graph
# y_proba[:, 1] - Orders probabilities in ascending order
plt.plot(X_new, y_proba[:, 1], "g-", label="Iris-Virginica")

# y_proba[:, 001] - Orders probabilities in descending order
plt.plot(X_new, y_proba[:, 0], "b--", label="Not Iris-Virginica")