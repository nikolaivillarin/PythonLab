import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from sklearn.datasets import fetch_openml
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone

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
# plt.imshow(some_digit_image, cmap = mpl.cm.binary, interpolation="nearest")
# plt.axis("off")
# plt.show()

# since we prefer numbers, cast y to integers
y = y.astype(np.uint8)

# create a test set and set it aside before inspecting the data closely
X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]

# Training a Binary Classifier
y_train_5 = (y_train == 5) # True for all 5s, False for all other digits
y_test_5 = (y_test == 5)


#region sgdClassifer.py
sgd_clf = SGDClassifier(random_state=42, max_iter=500)
sgd_clf.fit(X_train, y_train_5)

# Predict if the image is a 5
sgd_clf.predict([some_digit])
#endregion

#region customCrossVal.py
skfolds = StratifiedKFold(n_splits=3, random_state=42)

# Split the data and labels into three folds
for train_index, test_index in skfolds.split(X_train, y_train_5):
    clone_clf = clone(sgd_clf)

    # Use the index in the split to get your train data and labels
    X_train_folds = X_train[train_index]
    y_train_folds = y_train[train_index]

    # Use the test index in the split to get your test data and labels
    X_test_fold = X_train[test_index]
    y_test_fold = y_train_5[test_index]

    # Train the algorithm based on the fold
    clone_clf.fit(X_train_folds, y_train_folds)

    # Make the predictions
    y_pred = clone_clf.predict(X_test_fold)

    # Since we know the labels then we can just see if the predictions
    # match the labels
    n_correct = sum(y_pred == y_test_fold)

    print(n_correct / len(y_pred))
#endregion