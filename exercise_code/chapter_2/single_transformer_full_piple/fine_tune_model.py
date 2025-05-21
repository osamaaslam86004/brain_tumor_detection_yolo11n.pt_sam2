# Required for hyperparameter tuning
import pandas as pd
from scipy.stats import randint
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV

param_distribs = {
    # You are passing a full pipeline (with preprocessing and a regressor step named 'regressor')
    #  to RandomizedSearchCV.
    # The hyperparameters you want to tune belong to the RandomForestRegressor inside the pipeline, not directly to the pipeline.
    # Therefore, you must prefix the hyperparameter names  "regressor__n_estimators" "n_estimators"
    "regressor__n_estimators": randint(low=1, high=200),
    "regressor__max_features": randint(low=1, high=3),
}


def fine_tune_model(pipeline, X_train, y_train):
    """
    Fine-Tune Model Using Randomized Search:
    This code demonstrates using `RandomizedSearchCV` to find better hyperparameters for the `RandomForestRegressor`.
    """

    rnd_search = RandomizedSearchCV(
        pipeline,
        param_distributions=param_distribs,
        n_iter=10,  # this means "try 10 different random combinations"
        cv=5,  # For each candidate, you're doing 5-fold cross-validation.
        # This means: Split the dataset into 5 equal parts.
        # Use 4 parts to train, 1 part to test — repeat this 5 times,
        # changing the test part each time.
        # This helps avoid overfitting.
        # Totaling 50 fits = 10 candidates × 5 folds = 50 model trainings
        scoring="neg_mean_squared_error",
        random_state=42,
        verbose=2,
    )

    # Run randomized search
    rnd_search = rnd_search.fit(X_train, y_train)

    # Get the best model from the search
    fine_tuned_model = rnd_search.best_estimator_

    # Dynamically get dummy column names from training set
    ocean_columns = list(pd.get_dummies(X_train["ocean_proximity"]).columns)

    # Combine with manually engineered features
    feature_names = ["median_income", "population_per_household"] + ocean_columns

    # If you want to print feature importances, extract from regressor step
    feature_importances = fine_tuned_model.named_steps["regressor"].feature_importances_

    print("\nTop Feature Importances:")
    for name, score in zip(feature_names, feature_importances):
        print(f"{name}: {score:.4f}")

    return fine_tuned_model, rnd_search
