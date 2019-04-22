from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import f1_score

from Chapt3.sgdClassifier import *

# Get's the predictions in a boolean array. True if it predicts as a 5 false otherwise
y_train_pred = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3)

confusion_matrix(y_train_5, y_train_pred)

# Check Accuarcy
precision_score(y_train_5, y_train_pred)

recall_score(y_train_5, y_train_pred)

# Calculate F1 score
f1_score(y_train_5, y_train_pred)

# Decision Function
y_scores = sgd_clf.decision_function([some_digit])
y_scores

threshold = 0

y_some_digit_pred = (y_scores > threshold)

threshold = 300

y_some_digit_pred = (y_scores > threshold)
y_some_digit_pred

# Deciding what threshold to use
# Step 1 - Get the scores for your data
y_scores = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3,
    method="decision_function")

# Step 2 - Compute the precision and recall for all possible thresholds
#        - using the precision_recall_curve() function:
from sklearn.metrics import precision_recall_curve

precisions, recalls, thresholds = precision_recall_curve(y_train_5, y_scores)

# Step 3 - Visual the scores by mapping
def plot_precision_recall_vs_threshold(precisions, recalls, thresholds):
    plt.plot(recalls, precisions[:-1], "b--", label="Precision")
    plt.plot(thresholds, recalls[:-1], "g-", label="Recall")
    plt.xlabel("Threshold")
    plt.legend(loc="center left")
    plt.ylim([0,1])

plot_precision_recall_vs_threshold(precisions, recalls, thresholds)
plt.show()

# Step 4 - Choose your threshold
y_train_pred_90 = (y_scores > 1000)

#check the scores
precision_score(y_train_5, y_train_pred_90)

recall_score(y_train_5, y_train_pred_90)