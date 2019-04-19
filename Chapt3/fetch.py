import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from sklearn.datasets import fetch_openml

mnist = fetch_openml('mnist_784', version=1)

mnist.keys()

X, y = mnist["data"], mnist["target"]

X.shape

y.shape

#--== Display the Image ==--
# Get the First Character data. This is a 1 level array of pixels
some_digit = X[0]

# Reshape the array to a two level array
some_digit_image = some_digit.reshape(28, 28)

# show the image
plt.imshow(some_digit_image, cmap = mpl.cm.binary, interpolation="nearest")
plt.axis("off")
plt.show()

# since we prefer numbers, cast y to integers
y = y.astype(np.uint8)

# create a test set and set it aside before inspecting the data closely
X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]

# Training a Binary Classifier
y_train_5 = (y_train == 5) # True for all 5s, False for all other digits
y_test_5 = (y_test == 5)