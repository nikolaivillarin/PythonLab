from sklearn.datasets import make_moons
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler

# Import SVR
from sklearn.svm import SVR

# Creates a data points are shaped as two interleaving half circles
moons = make_moons()

# First Item in Array is an array of co-ordinates
# Data Structure: [1.94905575, 0.18489178], [0.00205461, 0.43592978]...
X = moons[0]

# Second Item in Array are target values
# Data Structure: [1, 1, 0, 0...]
y = moons[1]

# Instantiate SVM Model
svm_reg = SVR(kernel="poly", degree="2", C=100, epsilon=0.1)

# Train the Model
svm_reg.fit(X, y)

# Excample of making a prediction
svm_reg.predict([[1.94905575, 0.18489178]])