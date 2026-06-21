import joblib
import numpy as np

# load model and preprocessing objects

model = joblib.load("saved_models/xgboost.pkl")
scaler = joblib.load("saved_models/scaler.pkl")
encoders = joblib.load("saved_models/label_encoders.pkl")


def preprocess_input(data):

    processed = []

    columns = [
        "Age",
        "Sex",
        "Job",
        "Housing",
        "Saving accounts",
        "Checking account",
        "Credit amount",
        "Duration",
        "Purpose"
    ]

    for col in columns:

        value = data[col]

        if col in encoders:
            value = encoders[col].transform([value])[0]

        processed.append(value)

    processed = np.array(processed).reshape(1, -1)

    processed = scaler.transform(processed)

    return processed


def predict_risk(data):

    processed = preprocess_input(data)

    prediction = model.predict(processed)[0]

    probability = model.predict_proba(processed)[0]

    return prediction, probability