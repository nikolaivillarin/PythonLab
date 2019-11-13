from sklearn.datasets import make_moons
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler

# Change #1 - Importing SVC instead of LinearSVC
from sklearn.svm import SVC

# Creates a data points are shaped as two interleaving half circles
moons = make_moons()

# First Item in Array is an array of co-ordinates
# Data Structure: [1.94905575, 0.18489178], [0.00205461, 0.43592978]...
X = moons[0]

# Second Item in Array are target values
# Data Structure: [1, 1, 0, 0...]
y = moons[1]

# Instantiate SVM Model
# Change #2 - Initializes SVC
polynomial_svm_clf = Pipeline([
    ("scaler", StandardScaler()),
    ("svm_clf", SVC(kernel="poly", degree=3, coef0=1, C=5))
])

# Train the Model
polynomial_svm_clf.fit(X, y)

# Excample of making a prediction
polynomial_svm_clf.predict([[1.94905575, 0.18489178]])