from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

from Chapt2.columnTransformer import *

lin_reg = LinearRegression()
lin_reg.fit(housing_prepared, housing_labels)

some_data = housing.iloc[:5]
some_data

some_labels = housing_labels.iloc[:5]
some_labels

some_data_prepared = full_pipeline.transform(some_data)
some_data_prepared

print("Predictions:", lin_reg.predict(some_data_prepared))
print("Labels", list(some_labels))