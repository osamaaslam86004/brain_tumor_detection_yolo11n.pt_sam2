# Required for model evaluation
import numpy as np
from scipy import stats
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score


# Function to display scores
def display_scores(scores):
    print("Scores:", scores)
    print("Mean:", scores.mean())
    print("Standard deviation:", scores.std())


def evaluate_model(forest_reg, x_train, y_train):
    """
    Evaluate Model Using Cross-Validation
    This section evaluates the trained model using K-fold cross-validation and prints the scores
    """
    # Evaluate RandomForestRegressor using cross-validation
    forest_scores = cross_val_score(
        forest_reg,
        x_train,
        y_train,
        scoring="neg_mean_squared_error",
        cv=10,
    )
    forest_rmse_scores = np.sqrt(-forest_scores)

    return forest_rmse_scores


def evaluate_fine_tuned_model(final_model, strat_test_set):
    # Separate features and labels for the test set
    X_test = strat_test_set.drop("median_house_value", axis=1)
    y_test = strat_test_set["median_house_value"].copy()

    # Select the same top features used during training
    median_income = X_test["median_income"]
    population_per_household = X_test["population"] / X_test["households"]
    INLAND = (X_test["ocean_proximity"] == "INLAND").astype(float)

    # Create new feature matrix (n_samples x 3)
    X_test_selected = np.c_[median_income, population_per_household, INLAND]

    # Predict using the final model
    final_predictions = final_model.predict(X_test_selected)

    # Evaluate
    final_mse = mean_squared_error(y_test, final_predictions)
    final_rmse = np.sqrt(final_mse)

    print("\nFinal RMSE on the test set:", final_rmse)

    # Compute a 95% confidence interval for the test RMSE
    squared_errors = (final_predictions - y_test) ** 2
    confidence = 0.95
    # Using t-distribution (appropriate for smaller sample sizes, though large here)
    t_interval = np.sqrt(
        stats.t.interval(
            confidence,
            len(squared_errors) - 1,
            loc=squared_errors.mean(),
            scale=stats.sem(squared_errors),
        )
    )

    return t_interval
