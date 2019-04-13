from sklearn.tree import DecisionTreeRegressor

from Chapt2.trainingModel import *

# --== Initialize Model ==--
tree_reg = DecisionTreeRegressor()

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
tree_reg.fit(housing_prepared, housing_labels)

# --== Check how accurate this model is with the dataset ==--
housing_predictions = tree_reg.predict(housing_prepared)

variance = mean_squared_error(housing_labels, housing_predictions)
standardDeviation = np.sqrt(variance)
standardDeviation

# Note: Wait, What!? No Error at all?
# It is more likely that the model has badly overfit the data