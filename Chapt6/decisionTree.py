from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier

# Dataset contains a set of 150 records under five attributes:
# Petal Length, Petal Width, Sepal Length, Sepal Width and Species
# Row in dataset:
# [Sepal Length, Sepal Width, Petal Length, Petal Width, Species]
# [     5.1    ,     3.5    ,     1.4     ,     0.2    , l. setosa]
iris = load_iris()

# petal length, petal width. Data structure: [1.4, 0.2], [1.3, 0.2]...
X = iris.data[:, 2:]

# Iris-Virginica. 
# Target data structure [0, 1, 2]
# Target data references target_names ['setosa', 'versicolor', 'virginica']
y = iris.target

# Initialize Model
tree_clf = DecisionTreeClassifier(max_depth=2)

# Train Model
tree_clf.fit(X, y)


# -- Visualize the Tree --

from sklearn.tree import export_graphviz

# iris.target_names:
# ['setosa', 'versicolor', 'virginica']
export_graphviz(
    tree_clf,
    out_file="./Chapt6/iris_tree.dot",
    feature_names=iris.feature_names[2:],
    class_names=iris.target_names,
    rounded=True,
    filled=True
)

# Prediction
tree_clf.predict_proba([[5, 1.5]])
tree_clf.predict([[5, 1.5]])