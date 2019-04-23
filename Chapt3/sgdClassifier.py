from sklearn.linear_model import SGDClassifier

from Chapt3.fetch import *

sgd_clf = SGDClassifier(random_state=42, max_iter=100)
sgd_clf.fit(X_train, y_train_5)

# Predict if the image is a 5
sgd_clf.predict([some_digit])