from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone

from Chapt3.sgdClassifier import *

skfolds = StratifiedKFold(n_splits=3, random_state=42)

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