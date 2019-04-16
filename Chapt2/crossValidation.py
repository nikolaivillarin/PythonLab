from sklearn.model_selection import cross_val_score

from Chapt2.decisionTreeRegressor import *

# --== Get the Scores ==--
scores = cross_val_score(tree_reg, housing_prepared, housing_labels,
    scoring="neg_mean_squared_error", cv=10)

tree_standardDeviation_scores = np.sqrt(-scores)
# Scikit-Learn's cross-validation features expect a utility function (greater is better)
# rather than a cost function (lower is better), so the scoring function will
# return negative values

# --== Look at the Results ==--
def display_scores(scores):
    print("Scores: ", scores)
    print("Mean: ", scores.mean())
    print("Standard Deviation: ", scores.std())

display_scores(tree_standardDeviation_scores)

# --== Results ==--
# The Decision Tree has a score of approximately 71,407, generally +-2,439


# --== Look at the Results for LinearRegression ==--
lin_scores = cross_val_score(lin_reg, housing_prepared, housing_labels,
    scoring="neg_mean_squared_error", cv=10)

lin_rmse_scores = np.sqrt(-lin_scores)
display_scores(lin_rmse_scores)


# --== Look at the Results for RandomForestRegressor  ==--