import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

from client import train_client
from server import aggregate_models

# Load dataset
data = pd.read_csv("stress.csv")
data.columns = data.columns.str.strip()

print("Dataset Loaded ")
print("Columns:", data.columns)

# Split into 3 clients
client1 = data.sample(frac=0.33, random_state=1)
remaining = data.drop(client1.index)

client2 = remaining.sample(frac=0.5, random_state=2)
client3 = remaining.drop(client2.index)

print("Data split into 3 clients ")

# Train clients
train_client(client1, 1)
train_client(client2, 2)
train_client(client3, 3)

print("All clients trained ")

# Aggregate models (simulation)
aggregate_models()

# Global model
X = data.drop("Stress", axis=1)
y = data["Stress"]

global_model = RandomForestClassifier(n_estimators=100)
global_model.fit(X, y)

# Accuracy
y_pred = global_model.predict(X)
acc = accuracy_score(y, y_pred)
print("Model Accuracy:", acc)

# Confusion Matrix
cm = confusion_matrix(y, y_pred)

with open("confusion_matrix.pkl", "wb") as f:
    pickle.dump(cm, f)

sns.heatmap(cm, annot=True, fmt='d')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# Class Balance Graph
data['Stress'].value_counts().plot(kind='bar')
plt.title("Class Balance")
plt.show()

# Save model
with open("model.pkl", "wb") as f:
    pickle.dump(global_model, f)

print("Global model created ")