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


def evaluate_model(pipeline, x_train, y_train):
    """
    Evaluate Model Using Cross-Validation
    This section evaluates the trained model using K-fold cross-validation and prints the scores
    """
    # Evaluate RandomForestRegressor using cross-validation
    forest_scores = cross_val_score(
        pipeline,
        x_train,
        y_train,
        scoring="neg_mean_squared_error",
        cv=10,
    )
    forest_rmse_scores = np.sqrt(-forest_scores)

    return forest_rmse_scores


def evaluate_fine_tuned_model(pipeline, X_test, y_test):
    """
    Evaluate the fine-tuned pipeline on the test set and return the 95% confidence interval for RMSE.
    """
    y_pred = pipeline.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    print(f"\nFinal RMSE on the test set: {rmse:.2f}")

    # Compute 95% confidence interval using t-distribution
    squared_errors = (y_pred - y_test) ** 2
    confidence = 0.95
    t_interval = np.sqrt(
        stats.t.interval(
            confidence,
            len(squared_errors) - 1,
            loc=squared_errors.mean(),
            scale=stats.sem(squared_errors),
        )
    )
    print(f"95% confidence interval for RMSE: {t_interval}")
    return t_interval
