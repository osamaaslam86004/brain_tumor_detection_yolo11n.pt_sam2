# Required for data preparation pipelines and transformers
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


class SpecificFeatureSelector(BaseEstimator, TransformerMixin):
    def __init__(self, feature_names):
        self.feature_names = feature_names

    def fit(self, X, y=None):
        # X is a NumPy array, so we can't select by name here
        return self

    def transform(self, X):
        # Select by fixed indices: median_income (0), population_per_household (added by transformer), and INLAND (from OneHotEncoder)
        # These indices are hard-coded based on the order of transformation:
        #   - median_income is the 0th column (after numerical pipeline)
        #   - population_per_household is the 10th (added after numeric attributes: it's the second engineered feature)
        #   - INLAND is the last column in the one-hot encoded array
        return np.c_[X[:, [0, 10]], X[:, -1]]


def prepare_data(strat_train_set):
    """
    Data Preparation:
    This section includes a custom transformer to add combined attributes, defines pipelines for numerical and categorical features, and uses `ColumnTransformer` to apply the appropriate transformations to different columns.
    """

    # Dynamically get indices for important columns (to be passed to transformer)
    rooms_ix = strat_train_set.columns.get_loc("total_rooms")
    bedrooms_ix = strat_train_set.columns.get_loc("total_bedrooms")
    population_ix = strat_train_set.columns.get_loc("population")
    households_ix = strat_train_set.columns.get_loc("households")

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

    # Separate numerical attributes
    strat_train_set_num = strat_train_set.drop("ocean_proximity", axis=1)

    # List numerical and categorical attribute names
    num_attribs = list(strat_train_set_num)
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
    strat_train_set_prepared = full_pipeline.fit_transform(strat_train_set)

    # Select specific features (median_income, population_per_household, INLAND)
    feature_selector = SpecificFeatureSelector(
        ["median_income", "population_per_household", "INLAND"]
    )
    strat_train_set_with_selected_features_prepared = feature_selector.fit_transform(
        strat_train_set_prepared
    )

    return strat_train_set_with_selected_features_prepared
