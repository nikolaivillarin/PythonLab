import main
import pandas as pd
import numpy as np

from sklearn.model_selection import StratifiedShuffleSplit

housing = main.load_housing_data()

housing["income_cat"] = pd.cut(housing["median_income"],
    bins = [0., 1.5, 3.0, 4.5, 6., np.inf],
    labels = [1, 2, 3, 4, 5])

housing["income_cat"].hist()

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

for train_index, test_index in split.split(housing, housing["income_cat"]):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]

strat_test_set["income_cat"].value_counts() / len(strat_test_set)


#region Undoing Example of Cutting Data
for set_ in (strat_train_set, strat_test_set):
    set_.drop("income_cat", axis=1, inplace=True)
#endregion