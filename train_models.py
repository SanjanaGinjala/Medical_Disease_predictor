import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

data = pd.read_csv("dataset/heart.csv", header=None)

data.columns = [
    "age","sex","cp","trestbps","chol","fbs","restecg",
    "thalach","exang","oldpeak","slope","ca","thal","target"
]

# replace ? with NaN
data = data.replace("?", pd.NA)

# remove rows with missing values
data = data.dropna()

# convert everything to numeric
data = data.astype(float)
X = data.drop("target", axis=1)
y = data["target"]
scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train,X_test,y_train,y_test=train_test_split(
    X,y,test_size=0.2,random_state=42
)
models={
    "Logistic Regression":LogisticRegression(max_iter=5000),
    "Decision Tree":DecisionTreeClassifier(),
    "Random Forest":RandomForestClassifier(),
    "SVM":SVC(),
    "KNN": KNeighborsClassifier()
}

best_model=None
best_accuracy=0
print("Model Accuracies:\n")
for name,model in models.items():
    model.fit(X_train,y_train)
    predictions=model.predict(X_test)
    accuracy=accuracy_score(y_test,predictions)
    print(name,":",accuracy)
    if(accuracy>best_accuracy):
        best_accuracy=accuracy
        best_model=model

joblib.dump(best_model,"models/heart_model.pkl")
print("\n Best model saved as model.pkl")