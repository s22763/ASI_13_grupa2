import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

def preprocess(df:pd.DataFrame):
    X = df.drop("diabetes", axis=1)  # Replace "target_column_name" with the actual name of your target column.
    y = df["diabetes"]
    label_encoders = {}
    for column in X.select_dtypes(include=["object"]).columns:
        label_encoders[column] = LabelEncoder()
        X[column] = label_encoders[column].fit_transform(X[column])
    return train_test_split(X, y, test_size=0.15, random_state=42) # return X_train, X_test, y_train, y_test