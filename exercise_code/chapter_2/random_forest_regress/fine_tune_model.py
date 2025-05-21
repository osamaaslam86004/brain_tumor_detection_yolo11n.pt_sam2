# Required for hyperparameter tuning
from scipy.stats import expon, randint, reciprocal
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

# Define the distribution of hyperparameters to sample from
param_distribs = {
    # Try different numbers of trees (n_estimators) from 1 to 200
    "n_estimators": randint(low=1, high=200),
    # Try different numbers of features (max_features) between 1 and 8
    "max_features": randint(low=1, high=8),
}


def analyaze_best_model_error(full_pipeline, final_model):
    """
    This function tries to figure out which features (ingredients)
    are most important to the model’s decision.
    This is like asking: “What did the model pay attention to the most?
    Was it the number of bedrooms, or population density, or something else?”
    """

    # Tells us how much each feature contributes to predicting house prices
    feature_importances = final_model.feature_importances_

    # Extra features added by CombinedAttributesAdder
    extra_attribs = ["rooms_per_hhold", "pop_per_hhold", "bedrooms_per_room"]

    # Get category names from the one-hot encoder
    cat_encoder = full_pipeline.named_transformers_["cat"]
    cat_one_hot_attribs = list(cat_encoder.categories_[0])

    # Get numeric column names used in the 'num' pipeline
    num_attribs = full_pipeline.transformers_[0][2]

    # Collects all the feature names
    attributes = num_attribs + extra_attribs + cat_one_hot_attribs

    # Sorts them from most important to least
    sorted_feature_importances = sorted(
        zip(feature_importances, attributes), reverse=True
    )

    # Print the sorted list of (importance, feature name)
    print("Feature Importances:")
    for score, name in sorted_feature_importances:
        print(f"{name}: {score:.4f}")


def fine_tune_model(housing_prepared, housing_labels, full_pipeline):
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
    rnd_search = rnd_search.fit(housing_prepared, housing_labels)

    # Get the best model from the search
    final_model = rnd_search.best_estimator_

    # Analyze the best model's errors
    analyaze_best_model_error(full_pipeline, final_model)

    # Get the best model from the search
    final_model = rnd_search.best_estimator_

    return final_model, rnd_search
