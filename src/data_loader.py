import pandas as pd
from .config import DATASET_PATH

def load_data():
    df = pd.read_csv(DATASET_PATH)
    
    print("DATASET SHAPE:", df.shape)
    print("\nFirst 5 rows of the dataset:\n")
    print(df.head())
    
    return df    