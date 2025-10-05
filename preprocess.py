import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
import os

def preprocess_data(data_path='data/housing_data.csv'):
    """
    Loads data, handles missing values, and splits it into
    training and testing sets.
    """
    print("--- Starting Preprocessing ---")

    # Load the data
    df = pd.read_csv(data_path)
    print(f"Loaded data with {df.shape[0]} rows.")
    print("Initial data with missing values:")
    print(df)

    # Separate features (X) and target (y)
    X = df.drop('Price', axis=1)
    y = df['Price']

    # Use SimpleImputer to fill in missing values with the mean
    imputer = SimpleImputer(strategy='mean')
    X_imputed = imputer.fit_transform(X)

    # Convert the imputed data back to a DataFrame to keep column names
    X = pd.DataFrame(X_imputed, columns=X.columns)
    print("\nData after handling missing values:")
    print(X)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"\nData split into training ({X_train.shape[0]} rows) and testing ({X_test.shape[0]} rows).")

    print("--- Preprocessing Complete ---")
    return X_train, X_test, y_train, y_test

# This block allows the script to be run directly for testing
if __name__ == "__main__":
    # The preprocessed data is returned but we don't need to use it here.
    # We are just running the function to see the print statements.
    preprocess_data()

