from Chapt3.rocCurve import *

# Example of Scikit-Learn automatically using OvA strategy
sgd_clf.fit(X_train, y_train) # y_train, not y_train_5
sgd_clf.predict([some_digit])

# Call Decision function to see the scores for each class 0-9
some_digit_scores = sgd_clf.decision_function([some_digit])
some_digit_scores