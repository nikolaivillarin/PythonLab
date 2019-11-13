from sklearn.datasets import make_moons
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
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
rbf_kernel_svm_clf = Pipeline([
    ("scaler", StandardScaler()),
    ("svm_clf", SVC(kernel="rbf", gamma=5, C=0.001))
])

# Train the Model
rbf_kernel_svm_clf.fit(X, y)

# Excample of making a prediction
rbf_kernel_svm_clf.predict([[1.94905575, 0.18489178]])