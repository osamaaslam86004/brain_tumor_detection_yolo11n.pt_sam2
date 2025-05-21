import os

# To plot pretty figures
import matplotlib as mpl
import matplotlib.pyplot as plt

from data_preparation import prepare_data
from evaluate_model import display_scores, evaluate_fine_tuned_model, evaluate_model
from fine_tune_model import fine_tune_model
from get_data import fetch_housing_data, load_housing_data
from strated_dataset import create_stratified_test_set
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


def main(model_name="RandomForest"):  # Added model selection argument
    # Fetch the data
    fetch_housing_data(HOUSING_URL, HOUSING_PATH)

    # Load the data
    housing = load_housing_data(HOUSING_PATH)

    # Create a stratified test set
    housing, housing_labels, strat_test_set = create_stratified_test_set(housing)

    # Prepare data
    housing_prepared, full_pipeline = prepare_data(housing)

    # Train model
    model = train_model(housing_prepared, housing_labels)
    print(f"Model results: {model}")

    # Evaluate model
    model_rmse_scores = evaluate_model(model, housing_prepared, housing_labels)
    print(f"Model RMSE scores: {model_rmse_scores}")

    # Display the cross-validation scores
    print(f"{model_name} Cross-Validation Scores:")
    display_scores(model_rmse_scores)

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


if __name__ == "__main__":
    # Train and evaluate SVR
    main()
