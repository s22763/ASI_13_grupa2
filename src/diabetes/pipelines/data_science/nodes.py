import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, mean_squared_error
import pickle
import wandb

def create_model(X_train:pd.DataFrame, y_train:pd.DataFrame):
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)

    wandb.sklearn.plot_learning_curve(clf, X_train, y_train)
    return clf

def predict(X_validate:pd.DataFrame, y_validate:pd.DataFrame, clf:RandomForestClassifier):
    y_pred = clf.predict(X_validate)
    
    wandb.sklearn.plot_confusion_matrix(y_validate, y_pred)

    
    accuracy = accuracy_score(y_validate, y_pred)
    wandb.log({"accuracy": accuracy})
    return accuracy
    
def save_model(clf:RandomForestClassifier):
    pickle.dump(clf, open('model.pkl', 'wb'))
    