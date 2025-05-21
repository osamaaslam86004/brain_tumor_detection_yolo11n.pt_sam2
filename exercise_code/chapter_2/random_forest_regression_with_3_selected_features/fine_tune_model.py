# Required for hyperparameter tuning
from scipy.stats import randint
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV

# Define the distribution of hyperparameters to sample from
param_distribs = {
    # Try different numbers of trees (n_estimators) from 1 to 200
    "n_estimators": randint(low=1, high=200),
    # Try different numbers of features (max_features) between 1 and 8
    "max_features": randint(low=1, high=8),
}


def fine_tune_model(x_train, y_train):
    """
    Fine-Tune Model Using Randomized Search:
    This code demonstrates using `RandomizedSearchCV` to find better hyperparameters for the `RandomForestRegressor`.
    """

    # Set up and run RandomizedSearchCV
    forest_reg = RandomForestRegressor(random_state=42)

    rnd_search = RandomizedSearchCV(
        forest_reg,
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
    rnd_search = rnd_search.fit(x_train, y_train)

    # Get the best model from the search
    final_model = rnd_search.best_estimator_

    # Since we are using only 3 features, hardcode their names
    feature_names = ["median_income", "population_per_household", "INLAND"]
    feature_importances = final_model.feature_importances_

    print("\nTop 3 Feature Importances:")
    for name, score in zip(feature_names, feature_importances):
        print(f"{name}: {score:.4f}")

    return final_model, rnd_search
