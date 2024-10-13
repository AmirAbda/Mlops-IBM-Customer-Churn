import pandas as pd
from sklearn.model_selection import train_test_split
from services.preprocessing import preprocess_data
from services.training import train_model, evaluate_model, scale_data
import sys
import os

# Add the parent directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Configuration
DATA_FILEPATH = "data/data.xlsx"
TEST_SIZE = 0.3
RANDOM_STATE = 42

def main():
    # Preprocessing
    processed_df = preprocess_data(DATA_FILEPATH)

    if processed_df is None:
        print("Error occurred during preprocessing. Exiting.")
        return

    # Split Data
    X = processed_df.drop(columns=['Churn Value'])
    y = processed_df['Churn Value']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)

    # Scaling
    X_train_scaled, X_test_scaled = scale_data(X_train, X_test)

    # Training and Evaluation
    model = train_model(X_train_scaled, y_train)  # Defaults to Logistic Regression
    evaluate_model(model, X_test_scaled, y_test)

if __name__ == "__main__":
    main()