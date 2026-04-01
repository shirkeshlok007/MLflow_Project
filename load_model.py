import mlflow.pyfunc
import pandas as pd

model = mlflow.pyfunc.load_model("models:/My_Local_Model/1")

# FIX: use DataFrame with column name "x"
input_data = pd.DataFrame({"x": [10]})

prediction = model.predict(input_data)
print(prediction)
