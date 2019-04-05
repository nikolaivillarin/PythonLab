import Chapt2.customTransforms
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScalar
from sklearn.impute import SimpleImputer

num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy="median")),
    ('attribs_adder', CombinedAttributeAdder()),
    ('std_scalar', StandardScalar())
])

housing_num_tr = num_pipeline.fit_transform(housing_num)

housing_num_tr