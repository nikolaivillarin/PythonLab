from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

from Chapt2.prepareData import *
from Chapt2.pipelines import *

num_attribs = list(housing_num)
cat_attribs = ["ocean_proximity"]

full_pipeline = ColumnTransformer([
    ("num", num_pipeline, num_attribs),
    ("cat", OneHotEncoder(), cat_attribs)
])

housing_prepared = full_pipeline.fit_transform(housing)

housing_prepared