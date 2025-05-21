# Required for hyperparameter tuning
from scipy.stats import expon, reciprocal
from sklearn.model_selection import RandomizedSearchCV
from sklearn.svm import SVR


def analyaze_best_model_error(full_pipeline, final_model):
    """
    Analyzes the best model's errors, printing feature importances
    for RandomForest, or coefficients for linear SVR.
    """

    if hasattr(final_model, "coef_"):
        print("\nCoefficients of the best SVR model:")
        #  Note:  SVR with kernel='linear' has coef_, but not other kernels.
        #   Adapt for non-linear kernels if needed.
        feature_names = full_pipeline.get_feature_names_out()  # Get feature names
        for feature, coef in zip(
            feature_names, final_model.coef_[0]
        ):  # Assuming single output
            print(f"{feature}: {coef:.4f}")
    else:
        print(f"Coefficients not available for SVR with kernel: {final_model.kernel}")


def fine_tune_model(housing_prepared, housing_labels, full_pipeline):
    """
    Fine-Tune Model Using Randomized Search for RandomForest or SVR.
    """

    param_distribs = [
        {"kernel": ["linear"], "C": reciprocal(20, 200000)},
        {"kernel": ["rbf"], "C": reciprocal(20, 200000), "gamma": expon(scale=1.0)},
    ]
    model = SVR()
    scoring = "neg_mean_squared_error"  # Or try "neg_root_mean_squared_error"

    rnd_search = RandomizedSearchCV(
        model,
        param_distributions=param_distribs,
        n_iter=10,
        cv=5,
        scoring=scoring,
        random_state=42,
        verbose=2,
    )

    rnd_search.fit(housing_prepared, housing_labels)

    final_model = rnd_search.best_estimator_
    analyaze_best_model_error(full_pipeline, final_model)  # Analyze
    return final_model, rnd_search
