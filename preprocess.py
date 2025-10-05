import pandas as pd
from sklearn.datasets import load_breast_cancer
import os

def create_and_preprocess_dataframe() -> None:
    """Loads a built-in dataset, converts it to a pandas DataFrame, 
    and performs an initial inspection.
    """
    # 1. Load the dataset directly from scikit-learn
    # The 'bunch' object is like a dictionary containing data, feature names, etc.
    cancer_bunch = load_breast_cancer()
    
    print("Dataset loaded successfully from scikit-learn library.")

    # 2. Create a pandas DataFrame
    # The data itself is in the .data attribute
    # The column names are in the .feature_names attribute
    df = pd.DataFrame(cancer_bunch.data, columns=cancer_bunch.feature_names)
    
    # 3. Add the 'target' column (the answer: 0 for malignant, 1 for benign)
    df['target'] = cancer_bunch.target
    
    # 4. Perform our initial inspection!
    print("DataFrame created. Here's a quick summary:")
    print(df.info())

    print("\nFirst 5 rows of the new dataset:")
    print(df.head())
    
    # Define where to save the data and create the directory
    PROCESSED_DIR = "data/processed"
    PROCESSED_DATA_PATH = os.path.join(PROCESSED_DIR, "cancer_dataset.csv")
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    
    # Save the brand new, clean DataFrame to a CSV
    df.to_csv(PROCESSED_DATA_PATH, index=False)
    
    print(f"\nSuccessfully saved the new dataset to {PROCESSED_DATA_PATH}")


if __name__ == "__main__":
    create_and_preprocess_dataframe()

