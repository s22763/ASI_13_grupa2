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
    X_train, X_split, y_train, y_split = train_test_split(X, y, test_size=0.3, random_state=random_state)
    X_test, X_validate, y_test, y_validate = train_test_split(X_split, y_split, test_size=0.5, random_state=random_state)
    
    return X_train, X_test, X_validate, y_train, y_test, y_validate
