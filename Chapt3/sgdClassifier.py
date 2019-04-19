from sklearn.linear_model import SGDClassifier

from Chapt3.fetch import *

sgd_clf = SGDClassifier(random_state=42)
sgd_clf.fit(X_train, y_train_5)