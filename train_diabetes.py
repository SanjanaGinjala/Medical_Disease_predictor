import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

# load dataset
data = pd.read_csv("dataset/diabetes.csv")

# split features and target
X = data.drop("Outcome", axis=1)
y = data["Outcome"]

# train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# accuracy
accuracy = model.score(X_test, y_test)
print("Diabetes Model Accuracy:", accuracy)

# save model
joblib.dump(model, "models/diabetes_model.pkl")

print("Diabetes model saved")