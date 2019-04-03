import Chapt2.cuttingData
import matplotlib.pyplot as plt

from pandas.plotting import scatter_matrix

#region Discover and Visualize the Data to Gain Insights

# Copy the training set
housing = strat_train_set.copy()

# Since there is geographical information (latitude and longitude),
# it is a good idea to create a scatterplot of all districts
# Alpha Property - Setting this makes it much easier to visualize the places that
#                - have hight density
housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.1)

# cmap (Color Map) - called jet, which ranges from blue (low values) to red (high values)
# Option s - radius of each circle - in our case we are using district's population
# Option c - color - in our case to represent the price
housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.4,
    s=housing["population"]/100, label="population", figsize=(10,7),
    c="median_house_value", cmap=plt.get_cmap("jet"), colorbar=True
)
plt.legend()

# Looking for Correlations
corr_matrix = housing.corr()
corr_matrix["median_house_value"].sort_values(ascending=False)

# Scatter_Matrix function
attributes = ["median_house_value", "median_income", "total_rooms",
    "housing_median_age"]

scatter_matrix(housing[attributes], figsize=(12, 8))

housing.plot(kind="scatter", x="median_income", y="median_house_value", alpha=0.1)

# Experimenting with Attribute Combinations
# Creating new Attributes
housing["rooms_per_household"] = housing["total_rooms"]/housing["households"]
housing["bedrooms_per_room"] = housing["total_bedrooms"]/housing["total_rooms"]
housing["population_per_household"] = housing["population"]/housing["households"]

#endregion
