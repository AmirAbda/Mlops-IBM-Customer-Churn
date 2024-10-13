from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
from services.preprocessing import preprocess_data
from services.training import train_model, evaluate_model, scale_data
from sklearn.model_selection import train_test_split

app = FastAPI()

DATA_FILEPATH = "data/data.xlsx"
TEST_SIZE = 0.3
RANDOM_STATE = 42

# Pydantic Model for input validation
class DataModel(BaseModel):
    data_path: str = DATA_FILEPATH

@app.post("/train-model")
async def train_model_endpoint(data: DataModel):
    try:
        # Preprocessing
        processed_df = preprocess_data(data.data_path)

        if processed_df is None:
            raise HTTPException(status_code=400, detail="Error occurred during preprocessing")

        # Split Data
        X = processed_df.drop(columns=['Churn Value'])
        y = processed_df['Churn Value']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)

        # Scaling
        X_train_scaled, X_test_scaled = scale_data(X_train, X_test)

        # Training and Evaluation
        model = train_model(X_train_scaled, y_train)
        evaluate_model(model, X_test_scaled, y_test)

        return {"message": "Model trained and evaluated successfully"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {e}")

# Health check endpoint
@app.get("/")
def read_root():
    return {"message": "API is running"}

