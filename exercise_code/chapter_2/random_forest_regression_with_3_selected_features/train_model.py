# Required for training models
import joblib
from sklearn.ensemble import RandomForestRegressor


def train_model(x_train, y_train):
    """
    Train a Model (Random Forest Regressor)
    This code trains a `RandomForestRegressor`, which was found to perform well in the source.
    """

    # Define and train the RandomForestRegressor model
    forest_reg = RandomForestRegressor(n_estimators=100, random_state=42)

    # Fit the model to the training data
    forest_reg = forest_reg.fit(x_train, y_train)

    # save the model
    save_model(forest_reg)

    return forest_reg


def save_model(model):
    # Save the model to a file
    joblib.dump(model, "random_forest_model.pkl")
