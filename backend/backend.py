# backend.py

from fastapi import FastAPI
import joblib
import numpy as np
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import os
import uvicorn
from backend import app  # Ensure this matches your actual module structure




# Load the iris dataset
iris = load_iris()
X, y = iris.data, iris.target

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Load the trained model
model = joblib.load('irisTest.joblib')

# Calculate accuracy on the test set
model_accuracy = accuracy_score(y_test, model.predict(X_test))

# Initialize FastAPI app
app = FastAPI()

# Define FastAPI endpoints
@app.get("/")
async def read_root():
    return {"message": "Welcome to the model API!"}

@app.post("/predict/")
async def predict_species(data: dict):
    # Prediction logic
    features = np.array(data['features']).reshape(1, -1)
    prediction = model.predict(features)
    class_name = iris.target_names[prediction][0]
    # Include accuracy in the response
    return {"class": class_name, "accuracy": model_accuracy}


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
