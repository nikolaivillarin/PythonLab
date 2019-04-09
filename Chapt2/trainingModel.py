from sklearn.linear_model import LinearRegression

from Chapt2.columnTransformer import *

lin_reg = LinearRegression()
lin_reg.fit(housing_prepared, housing_labels)

some_data = housing.iloc[:5]
some_data