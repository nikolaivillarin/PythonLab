import numpy
import matplotlib.pyplot as plot
import pandas

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

from Chapt2.columnTransformer import *

# Instantiate The LinearRegression object
linearRegressor = LinearRegression()

# Train the model
linearRegressor.fit(housing_prepared, housing_labels)

# Test the Model
some_data = housing.iloc[:5]
some_data

some_data_prepared = full_pipeline.transform(some_data)
some_data_prepared

yPrediction = linearRegressor.predict(some_data_prepared)
yPrediction