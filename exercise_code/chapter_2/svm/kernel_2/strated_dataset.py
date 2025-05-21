# Required for splitting data
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

# Set random seed for reproducibility
np.random.seed(42)


def split_data(housing):

    # Perform stratified split based on income category
    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

    for train_index, test_index in split.split(housing, housing["income_cat"]):

        strat_train_set = housing.loc[train_index]
        strat_test_set = housing.loc[test_index]

    return strat_train_set, strat_test_set


# Create a Stratified Test Set
def create_stratified_test_set(housing):
    """
    This code creates an income category feature and then uses `StratifiedShuffleSplit` to create training and test sets that are representative of the income distribution in the full dataset. Finally, the income category is dropped from the sets.
    """

    # Create income categories to enable stratified sampling
    housing["income_cat"] = pd.cut(
        housing["median_income"],
        bins=[0.0, 1.5, 3.0, 4.5, 6.0, np.inf],
        labels=[1, 2, 3, 4, 5],
    )

    # Perform stratified split
    strat_train_set, strat_test_set = split_data(housing)

    # Drop the income_cat feature now that splitting is done
    for set_ in (strat_train_set, strat_test_set):
        set_.drop("income_cat", axis=1, inplace=True)

    # Separate features (housing) and labels (housing_labels) from the training set
    housing = strat_train_set.drop("median_house_value", axis=1)
    housing_labels = strat_train_set["median_house_value"].copy()

    return housing, housing_labels, strat_test_set
