from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

from Chapt2.columnTransformer import *

# --== Initialize the Model ==--
lin_reg = LinearRegression()

# --== Train the Model ==--
# housing_prepared =
#   All the data with columns for housing. This data has been cleaned. IE:
#   - Numeric values are transformed using the num_pipeline which uses a:
#   -- SimpleImputer to add missing values using the median
#   -- CombinedAttributesAdder which adds new attribute bedrooms_per_room
#   -- StandardScalar which normalizes the values so they range from 0 to 1
#   - Category values are converted using the OneHotEncoder
# housing_labels = 
#   median_house_value or the value we are predicting.
#   This is all the values in the median_house_value column
lin_reg.fit(housing_prepared, housing_labels)

# --== Prepare Test Data ==--
some_labels = housing_labels.iloc[:5]
some_labels

some_data = housing.iloc[:5]
some_data

some_data_prepared = full_pipeline.transform(some_data)
some_data_prepared

# --== Make some predictions ==--
print("Predictions:", lin_reg.predict(some_data_prepared))
print("Labels", list(some_labels))


# --== Check how Accurate the Results Are ==--
# Notice that we are checking the entire housing_prepared data instead of
# some_data_prepared. Were basically checking if this algorithm works well with
# our dataset
housing_predictions = lin_reg.predict(housing_prepared)
housing_predictions

predictions_variance = mean_squared_error(housing_labels, housing_predictions)
standardDeviation = np.sqrt(predictions_variance)
standardDeviation

# Note: This is an example of a model underfitting the training data. This occurs
# when your model is too simple to learn the underlying structure of the data