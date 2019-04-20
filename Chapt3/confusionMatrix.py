from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix

from Chapt3.sgdClassifier import *

# Get's the predictions in a boolean array. True if it predicts as a 5 false otherwise
y_train_pred = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3)
\