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


def evaluate_model(model, housing_prepared, housing_labels):
    """
    Evaluate Model Using Cross-Validation
    """
    # Evaluate model using cross-validation
    scores = cross_val_score(
        model,
        housing_prepared,
        housing_labels,
        scoring="neg_mean_squared_error",
        cv=10,
    )
    rmse_scores = np.sqrt(-scores)

    return rmse_scores


def evaluate_fine_tuned_model(final_model, full_pipeline, strat_test_set):
    # Separate features and labels for the test set
    X_test = strat_test_set.drop("median_house_value", axis=1)
    y_test = strat_test_set["median_house_value"].copy()

    # Prepare the test data using the pipeline fitted on the training data
    X_test_prepared = full_pipeline.transform(X_test)

    # Make predictions on the prepared test set
    final_predictions = final_model.predict(X_test_prepared)

    # Compute the Mean Squared Error and Root Mean Squared Error on the test set
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
