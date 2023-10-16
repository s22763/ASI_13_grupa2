import pandas as pd

def download()->pd.DataFrame:
    return pd.read_csv('https://raw.githubusercontent.com/s22763/ASI_13_grupa2/main/diabetes_prediction_dataset.csv')