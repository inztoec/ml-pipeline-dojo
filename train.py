import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Important: We are importing our OWN function from the other file!
from preprocess import preprocess_data

def train_model():
    """
    Trains a Linear Regression model on the preprocessed data.
    """
    print("--- Starting Model Training ---")

    # 1. Get the clean data by calling our preprocessing function
    X_train, X_test, y_train, y_test = preprocess_data()

    # 2. Create and train the model
    print("\nTraining the Linear Regression model...")
    model = LinearRegression()
    model.fit(X_train, y_train)
    print("Model training complete.")

    # 3. Make predictions on the test data
    predictions = model.predict(X_test)
    print("\nMade predictions on the test set:")
    print(predictions)

    # 4. Evaluate the model
    mse = mean_squared_error(y_test, predictions)
    print(f"\nModel Evaluation - Mean Squared Error: {mse:.2f}")

    print("--- Model Training Complete ---")
    return model

# This block allows us to run the script directly.
if __name__ == "__main__":
    trained_model = train_model()
    # In a real application, you would now save 'trained_model' to a file.
    # For example, using joblib:
    # import joblib
    # joblib.dump(trained_model, 'housing_model.pkl')
    # print("\nModel saved to housing_model.pkl")
