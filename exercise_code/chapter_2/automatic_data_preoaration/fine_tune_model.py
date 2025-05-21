# Required for hyperparameter tuning
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import OneHotEncoder

param_grid = {
    # Accessing the regressor within the 'full_pipeline'
    "regressor__n_estimators": [50, 100, 150, 200],  # Number of trees
    "regressor__max_features": [1, 2, 3, 4],  # Features per split
}


def fine_tune_model(pipeline, X_train, y_train):
    """
    Fine-Tune Model Using Randomized Search:
    """
    # Removed param_grid as it's now passed to RandomizedSearchCV

    # Updated to use pipeline directly (it already includes the regressor)
    random_search = RandomizedSearchCV(
        pipeline,
        param_distributions=param_grid,  # Corrected: use param_distributions
        n_iter=10,  # Number of random combinations to try
        cv=5,  # Cross-validation folds
        verbose=2,
        error_score="raise",
        scoring="neg_mean_squared_error",  # Add scoring consistent with data_preparation.py
    )

    # Run the randomized search
    random_search.fit(X_train, y_train)

    # Get the best model and print results
    fine_tuned_model = random_search.best_estimator_

    # Dynamically get dummy column names - adjust for pipeline
    # Note: This part assumes 'ocean_proximity' is one-hot encoded
    # and that the feature selector is still present. If you change
    # either of these, adjust this part accordingly.
    preparation_step = fine_tuned_model.named_steps["preparation"]
    # Find the one-hot encoder within the preparation step
    one_hot_encoder = None
    for name, transformer, columns in preparation_step.transformers_:
        if isinstance(transformer, OneHotEncoder):
            one_hot_encoder = transformer
            break

    if one_hot_encoder:
        ocean_columns = list(
            one_hot_encoder.get_feature_names_out(input_features=["ocean_proximity"])
        )
    else:
        ocean_columns = []
        print(
            "Warning: OneHotEncoder not found in pipeline. Feature importances might be incorrect."
        )

    # Reconstruct feature names accounting for pipeline and feature selection
    feature_names = ["median_income", "population_per_household"] + ocean_columns

    if "feature_selector" in fine_tuned_model.named_steps:
        # Feature selection is present, adjust feature names accordingly
        selector = fine_tuned_model.named_steps["feature_selector"]
        # In SpecificFeatureSelector, we hardcode the features selected. You
        # can adjust this logic if you change the selection mechanism.
        selected_indices = [0, 1, 2]  # Indices after feature selection
        feature_names = [feature_names[i] for i in selected_indices]
        # Print feature importances (handle cases where the selector might not preserve order)
    feature_importances = fine_tuned_model.named_steps["regressor"].feature_importances_

    print("\nTop Feature Importances:")
    for name, score in zip(feature_names, feature_importances):
        print(f"{name}: {score:.4f}")
    print("Best hyperparameters found:", random_search.best_params_)

    return fine_tuned_model, random_search
