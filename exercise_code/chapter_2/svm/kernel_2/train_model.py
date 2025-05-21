# Required for training models
import joblib
from sklearn.svm import SVR


def train_model(housing_prepared, housing_labels):
    """
    Train a Model (Random Forest Regressor or SVR)
    """

    model = SVR(kernel="linear", C=1.0)  # Default, can be tuned

    # Fit the model to the training data
    model.fit(housing_prepared, housing_labels)

    # save the model
    save_model(model)

    return model


def save_model(model):
    # Save the model to a file
    joblib.dump(model, "SVR_model.pkl")
