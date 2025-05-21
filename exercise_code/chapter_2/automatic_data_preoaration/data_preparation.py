import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV  # Added GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (  # Added MinMaxScaler
    MinMaxScaler,
    OneHotEncoder,
    StandardScaler,
)


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

    # Define the parameter grid for GridSearchCV
    param_grid = [
        {
            "preparation__num__imputer__strategy": ["mean", "median", "most_frequent"],
            "preparation__num__std_scaler": [StandardScaler(), MinMaxScaler()],
        }
    ]

    # Initialize GridSearchCV
    grid_search = GridSearchCV(
        pipeline,
        param_grid,
        cv=3,  # You can adjust the number of cross-validation folds
        scoring="neg_mean_squared_error",  # Using negative MSE as the scoring metric
        n_jobs=-1,  # Use all available cores
        verbose=2,
        error_score="raise",
    )

    # Fit GridSearchCV on the training data
    grid_search = grid_search.fit(X_train, y_train)

    # Get the best estimator from the grid search
    best_pipeline = grid_search.best_estimator_

    # Now you can predict directly on raw dataframes:
    y_pred = best_pipeline.predict(X_test)

    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best negative MSE: {grid_search.best_score_}")  # Best score is negative MSE
    print(f"Predictions using the best pipeline: {y_pred}")

    return best_pipeline, best_pipeline, X_train, y_train, X_test, y_test, y_pred
