import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Autolog MLflow
mlflow.autolog()

# Load Data
df = pd.read_csv("amazon_delivery_preprocessing.csv")
X = df.drop("is_delayed", axis=1)
y = df["is_delayed"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

with mlflow.start_run():
    rf = RandomForestClassifier(n_estimators=50, random_state=42)
    rf.fit(X_train, y_train)
    score = rf.score(X_test, y_test)
    print(f"Workflow CI succeeded! Accuracy: {score}")