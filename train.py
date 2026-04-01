import mlflow
import pandas as pd
from sklearn.linear_model import LogisticRegression

# Step 1: Connect MLflow
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("Demo_Experiment")

# Step 2: Create Data (Pandas)
data = pd.DataFrame({
    "x": [1, 2, 3, 4, 5, 6],
    "y": [0, 0, 0, 1, 1, 1]
})

X = data[["x"]]
y = data["y"]

# Step 3: Start MLflow run
with mlflow.start_run():
    # Train model
    model = LogisticRegression()
    model.fit(X, y)

    # Evaluate
    accuracy = model.score(X, y)

    # Log data
    mlflow.log_param("model", "LogisticRegression")
    mlflow.log_metric("accuracy", accuracy)

    # Register model
    mlflow.sklearn.log_model(
        model,
        "model",
        registered_model_name="My_Local_Model"
    )

print("Done ✅ Model logged and registered!")
