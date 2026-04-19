import pickle
import numpy as np

def aggregate_models():
    models = []

    for i in range(1, 4):
        with open(f"client_{i}.pkl", "rb") as f:
            models.append(pickle.load(f))

    # NOTE: RandomForest me direct averaging difficult hai
    # So we simulate aggregation by retraining

    print("Models received from clients ")

    return models