# Required for hyperparameter tuning
import pandas as pd
from sklearn.model_selection import GridSearchCV

param_grid = {
    # Number of trees in the forest.  More trees generally lead to better performance but increase training time.
    "regressor__n_estimators": [50, 100, 150, 200],
    # The number of features to consider when looking for the best split.  Controls the diversity of trees in the forest
    "regressor__max_features": [1, 2, 3, 4],
}


def fine_tune_model(pipeline, X_train, y_train):
    """
    Fine-Tune Model Using Randomized Search:
    This code demonstrates using `GridSearchCV` to find better hyperparameters for the `RandomForestRegressor`.
    """

    grid_search = GridSearchCV(
        pipeline,
        param_grid=param_grid,
        cv=5,
        scoring="neg_mean_squared_error",
        verbose=2,
        error_score="raise",  # to raise errors during fitting
    )

    # Run grid search
    grid_search = grid_search.fit(X_train, y_train)

    # Get the best model from the search
    fine_tuned_model = grid_search.best_estimator_

    # Dynamically get dummy column names from training set
    ocean_columns = list(pd.get_dummies(X_train["ocean_proximity"]).columns)

    # Combine with manually engineered features
    feature_names = ["median_income", "population_per_household"] + ocean_columns

    # If you want to print feature importances, extract from regressor step
    feature_importances = fine_tuned_model.named_steps["regressor"].feature_importances_

    print("\nTop Feature Importances:")
    for name, score in zip(feature_names, feature_importances):
        print(f"{name}: {score:.4f}")

    print("Best hyperparameters found:", grid_search.best_params_)

    return fine_tuned_model, grid_search
