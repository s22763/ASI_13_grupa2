import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import wandb
import requests as r
def download():
    response = r.request("GET","http://127.0.0.1:8000/download").json()
    return pd.DataFrame.from_dict(response)

def init_wandb(configg:dict):
    wandb.init(
    project="asi-2",
    config=configg)
    
    
def preprocess(df:pd.DataFrame, random_state:int):
    #X = df.drop("diabetes", axis=1)  # Replace "target_column_name" with the actual name of your target column.
    y = df["diabetes"]
    X = df
    label_encoders = {}
    for column in X.select_dtypes(include=["object"]).columns: # replace "object" with "category"
        label_encoders[column] = LabelEncoder()
        X[column] = label_encoders[column].fit_transform(X[column])
    X_train, X_split, y_train, y_split = train_test_split(X, y, test_size=0.3, random_state=random_state)
    X_test, X_validate, y_test, y_validate = train_test_split(X_split, y_split, test_size=0.5, random_state=random_state)


    data = {
        "X_train": X_train.to_json(orient="records"),
        "X_test": X_test.to_json(orient="records"),
        "X_validate": X_validate.to_json(orient="records")
    }
    r.post("http://localhost:8000/preprocess", json=data)
    
    X_test = X_test.drop("diabetes", axis=1)
    X_validate = X_validate.drop('diabetes', axis=1)
    
    return X_train, X_test, X_validate, y_train, y_test, y_validate
