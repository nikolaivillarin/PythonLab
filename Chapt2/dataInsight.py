import main

housing = main.load_housing_data()

#region Examples of show attributes and gaining insight of data
housing.head()
housing.info()
housing["ocean_proximity"].value_counts()
housing.describe()
#endregion