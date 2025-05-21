import os
import sys

# To plot pretty figures
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import sklearn
from data_preparation import prepare_data
from evaluate_model import display_scores, evaluate_fine_tuned_model, evaluate_model
from fine_tune_model import fine_tune_model
from get_data import fetch_housing_data, load_housing_data
from strated_dataset import create_stratified_test_set, split_data
from train_model import train_model

# Set Matplotlib defaults
mpl.rc("axes", labelsize=14)
mpl.rc("xtick", labelsize=12)
mpl.rc("ytick", labelsize=12)

# Where to save the figures
PROJECT_ROOT_DIR = "."
CHAPTER_ID = "end_to_end_project"
IMAGES_PATH = os.path.join(PROJECT_ROOT_DIR, "images", CHAPTER_ID)
os.makedirs(IMAGES_PATH, exist_ok=True)


# Function to save figures
def save_fig(fig_id, tight_layout=True, fig_extension="png", resolution=300):
    path = os.path.join(IMAGES_PATH, fig_id + "." + fig_extension)
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)


DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml2/master/"
HOUSING_PATH = os.path.join("datasets", "housing")
HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"


def main():
    # Fetch the data
    fetch_housing_data(HOUSING_URL, HOUSING_PATH)

    # Load the data
    housing = load_housing_data(HOUSING_PATH)

    # Create a stratified test set
    housing, housing_labels, strat_test_set = create_stratified_test_set(housing)

    # Prepare data
    housing_prepared, full_pipeline = prepare_data(housing)

    # Train model
    forest_reg = train_model(housing_prepared, housing_labels)
    print(f"Model results {forest_reg}")

    # Evaluate model
    forest_rmse_scores = evaluate_model(forest_reg, housing_prepared, housing_labels)
    print(f"Model rmese scores {forest_rmse_scores}")

    # Display the cross-validation scores
    print("Random Forest Cross-Validation Scores:")
    display_scores(forest_rmse_scores)

    # Fine-Tune Model Using Randomized Search**
    final_model, rnd_search = fine_tune_model(
        housing_prepared, housing_labels, full_pipeline
    )
    # Print the best hyperparameters found
    print("\nBest hyperparameters found by Randomized Search:")
    print(rnd_search.best_params_)

    # Evaluate the Final Model on the Test Set
    # evaluate fine tuned model
    t_interval = evaluate_fine_tuned_model(final_model, full_pipeline, strat_test_set)
    print("95% Confidence Interval for Test RMSE (t-dist):", t_interval)

    # **9. Custom Feature Selection Transformer (Optional)**
    # This code defines a custom transformer for selecting the top k features based on pre-computed feature importances. This is shown in the exercise solutions as a potential addition to the pipeline.

    # ```python
    # # Helper function to get indices of top k values in an array
    # def indices_of_top_k(arr, k): #
    #     return np.sort(np.argpartition(np.array(arr), -k)[-k:]) #

    # # Custom Transformer for selecting top features
    # class TopFeatureSelector(BaseEstimator, TransformerMixin): #
    #     def __init__(self, feature_importances, k): #
    #         self.feature_importances = feature_importances #
    #         self.k = k #
    #     def fit(self, X, y=None): #
    #         self.feature_indices_ = indices_of_top_k(self.feature_importances, self.k) #
    #         return self #
    #     def transform(self, X): #
    #         return X[:, self.feature_indices_] #

    # To use this, you first need feature importances from a trained model
    # For example, from the Randomized Search best estimator:
    # feature_importances = rnd_search.best_estimator_.feature_importances_ #
    # k = 5 # example number of features to keep #
    # top_k_selector = TopFeatureSelector(feature_importances, k)
    # housing_prepared_top_k = top_k_selector.fit_transform(housing_prepared)


if __name__ == "__main__":
    main()
