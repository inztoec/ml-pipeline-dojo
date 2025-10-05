import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import os
import joblib # <-- NEW: Import the joblib library

def train_model():
    """
    Loads the processed data, trains a classification model, evaluates it,
    and saves the trained model to a file.
    """
    # 1. Load the processed data
    PROCESSED_DATA_PATH = os.path.join("data", "processed", "cancer_dataset.csv")    
    print(f"Loading data from {PROCESSED_DATA_PATH}...")
    df = pd.read_csv(PROCESSED_DATA_PATH)

    # 2. Separate features (X) and the target (y)
    X = df.drop('target', axis=1)
    y = df['target']
    print("Data separated into features (X) and target (y).")

    # 3. Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print("Data split into training and testing sets.")

    # 4. Initialize and train the model
    model = LogisticRegression(max_iter=10000)
    print("Training the Logistic Regression model...")
    model.fit(X_train, y_train)
    print("Model training complete.")

    # 5. Make predictions on the test set and evaluate
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    # --- NEW: Save the trained model ---
    MODELS_DIR = "models"
    os.makedirs(MODELS_DIR, exist_ok=True) # Create the 'models' directory
    MODEL_PATH = os.path.join(MODELS_DIR, "logistic_regression_v1.joblib")
    print(f"Saving trained model to {MODEL_PATH}")
    joblib.dump(model, MODEL_PATH)
    # --- END NEW ---

    print("\n--- Model Evaluation ---")
    print(f"Accuracy on the test set: {accuracy * 100:.2f}%")
    print("------------------------")
    
if __name__ == "__main__":
    train_model()
