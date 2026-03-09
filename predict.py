import joblib
import numpy as np

heart_model = joblib.load("models/heart_model.pkl")
diabetes_model = joblib.load("models/diabetes_model.pkl")
parkinsons_model = joblib.load("models/parkinsons_model.pkl")


def prepare_input(data, model):
    expected = model.n_features_in_

    # fill missing values with 0
    if len(data) < expected:
        data = data + [0]*(expected-len(data))

    return np.array(data).reshape(1,-1)


def predict_heart(data):
    data = prepare_input(data, heart_model)
    return heart_model.predict(data)


def predict_diabetes(data):
    data = prepare_input(data, diabetes_model)
    return diabetes_model.predict(data)


def predict_parkinsons(data):
    data = prepare_input(data, parkinsons_model)
    return parkinsons_model.predict(data)