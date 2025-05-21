# Required for data preparation pipelines and transformers
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def prepare_data(housing):
    """
    Data Preparation:
    This section includes a custom transformer to add combined attributes, defines pipelines for numerical and categorical features, and uses `ColumnTransformer` to apply the appropriate transformations to different columns.
    """

    # Dynamically get indices for important columns (to be passed to transformer)
    rooms_ix = housing.columns.get_loc("total_rooms")
    bedrooms_ix = housing.columns.get_loc("total_bedrooms")
    population_ix = housing.columns.get_loc("population")
    households_ix = housing.columns.get_loc("households")

    # Custom Transformer to add combined attributes
    class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
        def __init__(self, add_bedrooms_per_room=True):  # no *args or **kargs
            self.add_bedrooms_per_room = add_bedrooms_per_room

        def fit(self, X, y=None):
            return self  # nothing else to do

        def transform(self, X):
            rooms_per_household = X[:, rooms_ix] / X[:, households_ix]
            population_per_household = X[:, population_ix] / X[:, households_ix]
            if self.add_bedrooms_per_room:  #
                bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
                return np.c_[
                    X,
                    rooms_per_household,
                    population_per_household,
                    bedrooms_per_room,
                ]
            else:
                return np.c_[X, rooms_per_household, population_per_household]

        def get_feature_names_out(self, input_features=None):
            if input_features is None:
                return ["rooms_per_hhold", "pop_per_hhold", "bedrooms_per_room"]
            elif self.add_bedrooms_per_room:
                return list(input_features) + [
                    "rooms_per_hhold",
                    "pop_per_hhold",
                    "bedrooms_per_room",
                ]
            else:
                return list(input_features) + ["rooms_per_hhold", "pop_per_hhold"]

    # Separate numerical attributes
    housing_num = housing.drop("ocean_proximity", axis=1)

    # List numerical and categorical attribute names
    num_attribs = list(housing_num)
    cat_attribs = ["ocean_proximity"]

    # Create numerical processing pipeline
    num_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("attribs_adder", CombinedAttributesAdder()),
            ("std_scaler", StandardScaler()),
        ]
    )

    # Create full pipeline using ColumnTransformer
    full_pipeline = ColumnTransformer(
        [
            (
                "num",
                num_pipeline,
                num_attribs,
            ),  # You pass num_attribs (a list of numeric column names) to the 'num' part of the ColumnTransformer.
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_attribs),
        ]  # Note: The `handle_unknown='ignore'` setting in `OneHotEncoder` is added in the source's exercise solutions to prevent errors during cross-validation if a category (like 'ISLAND') is only present in some folds.
    )

    # Apply the full pipeline to the training data
    housing_prepared = full_pipeline.fit_transform(housing)

    return housing_prepared, full_pipeline
