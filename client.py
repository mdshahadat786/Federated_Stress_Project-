from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import pickle

def train_client(data, client_id):
    data.columns = data.columns.str.strip()   

    X = data.drop("Stress", axis=1)
    y = data["Stress"]

    model = RandomForestClassifier(n_estimators=100)
    model.fit(X, y)

    with open(f"client_{client_id}.pkl", "wb") as f:
        pickle.dump(model, f)

    return model