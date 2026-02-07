import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Load data
df = pd.read_csv("behavioursurvey.csv")

# Remove missing data
df.dropna(inplace=True)

# Encode categorical features
df["Personality"].replace({"Extrovert": 0, "Introvert": 1}, inplace=True)
df["Stage_fear"].replace({"No": 0, "Yes": 1}, inplace=True)
df["Drained_after_socializing"].replace({"No": 0, "Yes": 1}, inplace=True)

# Prepare features and target
x = df.drop(columns=["id", "Personality"])
y = df["Personality"]

# Split data
X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Train model
model = KNeighborsClassifier()
model.fit(X_train, Y_train)

# Calculate accuracy
Y_pred = model.predict(X_test)
accuracy = accuracy_score(Y_pred, Y_test)

# Save model to pickle file
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

# Save feature names for later use
with open("feature_names.pkl", "wb") as f:
    pickle.dump(x.columns.tolist(), f)

print(f"Model saved as 'model.pkl'")
print(f"Feature names saved as 'feature_names.pkl'")
print(f"Model Accuracy: {accuracy:.4f}")
