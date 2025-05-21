import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# Custom transformer to add engineered features
class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        add_bedrooms_per_room=True,
        rooms_ix=None,
        bedrooms_ix=None,
        population_ix=None,
        households_ix=None,
    ):
        self.add_bedrooms_per_room = add_bedrooms_per_room
        self.rooms_ix = rooms_ix
        self.bedrooms_ix = bedrooms_ix
        self.population_ix = population_ix
        self.households_ix = households_ix

    def fit(self, X, y=None):
        # Just check indices are set
        if None in [
            self.rooms_ix,
            self.bedrooms_ix,
            self.population_ix,
            self.households_ix,
        ]:
            raise ValueError(
                "Column indices must be provided to CombinedAttributesAdder"
            )
        return self

    def transform(self, X):
        rooms_per_household = X[:, self.rooms_ix] / X[:, self.households_ix]
        population_per_household = X[:, self.population_ix] / X[:, self.households_ix]
        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:, self.bedrooms_ix] / X[:, self.rooms_ix]
            return np.c_[
                X,
                rooms_per_household,
                population_per_household,
                bedrooms_per_room,
            ]
        else:
            return np.c_[X, rooms_per_household, population_per_household]


# Custom transformer to select the 3 features by index
class SpecificFeatureSelector(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Select by fixed indices: median_income (0), population_per_household (10), INLAND (last)
        return np.c_[X[:, [0, 10]], X[:, -1]]


def make_full_pipeline(df):

    rooms_ix = df.columns.get_loc("total_rooms")
    bedrooms_ix = df.columns.get_loc("total_bedrooms")
    population_ix = df.columns.get_loc("population")
    households_ix = df.columns.get_loc("households")

    num_attribs = list(df.drop("ocean_proximity", axis=1))
    cat_attribs = ["ocean_proximity"]

    attribs_adder = CombinedAttributesAdder(
        rooms_ix=rooms_ix,
        bedrooms_ix=bedrooms_ix,
        population_ix=population_ix,
        households_ix=households_ix,
    )
    num_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("attribs_adder", attribs_adder),
            ("std_scaler", StandardScaler()),
        ]
    )

    # Full pipeline with feature selection and estimator
    full_pipeline = Pipeline(
        [
            (
                "preparation",
                ColumnTransformer(
                    [
                        ("num", num_pipeline, num_attribs),
                        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_attribs),
                    ]
                ),
            ),
            ("feature_selector", SpecificFeatureSelector()),
            ("regressor", RandomForestRegressor(n_estimators=100, random_state=42)),
        ]
    )
    return full_pipeline


def selected_features_pipeline_that_predict(strat_train_set, strat_test_set):

    X_train = strat_train_set.drop("median_house_value", axis=1)
    y_train = strat_train_set["median_house_value"]

    X_test = strat_train_set.drop("median_house_value", axis=1)
    y_test = strat_train_set["median_house_value"]

    pipeline = make_full_pipeline(X_train)
    train_model = pipeline.fit(X_train, y_train)

    # Now you can predict directly on raw dataframes:
    y_pred = pipeline.predict(X_test)
    print(f"prediction: {y_pred}")

    return pipeline, train_model, X_train, y_train, X_test, y_test, y_pred
