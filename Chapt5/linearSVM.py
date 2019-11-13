import numpy as np
from sklearn import datasets
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

# Dataset contains a set of 150 records under five attributes:
# Petal Length, Petal Width, Sepal Length, Sepal Width and Species
# Row in dataset:
# [Sepal Length, Sepal Width, Petal Length, Petal Width, Species]
# [     5.1    ,     3.5    ,     1.4     ,     0.2    , l. setosa]
iris = datasets.load_iris()

# petal length, petal width. Data structure: [1.4, 0.2], [1.3, 0.2]...
X = iris["data"][:, (2, 3)]

# Iris-Virginica. 
# Target data structure [0, 1, 2]
# Target data references target_names ['setosa', 'versicolor', 'virginica']
# iris["target"] == 2 returns boolean True/False
# .asType converts the boolean to numeric 0.0 (False), 1.0 (True)
# Final data structure [0.0, 1.0...]
y = (iris["target"] == 2).astype(np.float64)

# Initialize SVM Model
svm_clf = Pipeline([
    ("scaler", StandardScaler()),
    ("linear_svc", LinearSVC(C=1, loss="hinge"))
])

# Train the Model
svm_clf.fit(X, y)