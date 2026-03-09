import joblib
import numpy as np

heart_model = joblib.load("models/heart_model.pkl")
diabetes_model = joblib.load("models/diabetes_model.pkl")
parkinsons_model = joblib.load("models/parkinsons_model.pkl")

def predict_heart(data):
    data = np.array(data).reshape(1,-1)
    return heart_model.predict(data)

def predict_diabetes(data):
    print("Expected features:", diabetes_model.n_features_in_)
    data = np.array(data).reshape(1,-1)
    return diabetes_model.predict(data)
    
def predict_parkinsons(data):
    data = np.array(data).reshape(1,-1)
    return parkinsons_model.predict(data)