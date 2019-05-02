from Chapt3.confusionMatrix import *

from sklearn.metrics import roc_curve
from sklearn.model_selection import cross_val_predict

fpr, tpr, thresholds = roc_curve(y_train_5, y_scores)

def plot_roc_curve(fpr, tpr, label=None):
    plt.plot(fpr, tpr, linewidth=2, label=label)
    plt.plot([0, 1], [0, 1], 'w--')
    plt.axis([0, 1, 0, 1])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')

plot_roc_curve(fpr, tpr)
plt.show()

# Measure area under the curve
from sklearn.metrics import roc_auc_score

roc_auc_score(y_train_5, y_scores)

# Performance Measure for RandomForestClassifier
from sklearn.ensemble import RandomForestClassifier

forest_clf = RandomForestClassifier(random_state=42)

y_probas_forest = cross_val_predict(forest_clf, X_train, y_train_5, cv=3,
    method="predict_proba")

# Grab the second column which is the probability that it is a 5
y_scores_forest = y_probas_forest[:,1] # score = proba of positive class

# fpr = False Positive Rate
# tpr = True Positive Rate
fpr_forest, tpr_forest, thresholds_forest = roc_curve(y_train_5, y_scores_forest)

# Plot the curve. It is useful to plot the first ROC curve as well to see
# how they compare
plt.plot(fpr, tpr, "b:", label="SGD")
plot_roc_curve(fpr_forest, tpr_forest, "Random Forest")
plt.legend(loc="lower right")
plt.show()

roc_auc_score(y_train_5, y_scores_forest)