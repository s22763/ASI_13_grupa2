import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, mean_squared_error
from autogluon.tabular import TabularDataset, TabularPredictor
import pickle
import wandb

def create_model(X_train, y_train, model_type):

    clf = None
    if(model_type == "autogluon"):
        label = 'diabetes'
        clf = TabularPredictor(label)
        clf.fit(X_train)
    elif(model_type == "regression"):
        clf = LogisticRegression()
        clf.fit(X_train, y_train)
    else:               
        clf = RandomForestClassifier(n_estimators=100, random_state=42)
        clf.fit(X_train, y_train)
     
    coln = []
    for col in X_train.columns:
        coln.append(col)
        
    wandb.sklearn.plot_learning_curve(clf, X_train, y_train)
    wandb.sklearn.plot_feature_importances(clf, coln)
    
    return clf

def predict(X_validate, y_validate, clf):
    
    y_pred = clf.predict(X_validate)
    wandb.sklearn.plot_confusion_matrix(y_validate, y_pred, ['diabetic', 'not diabetic'])   
    accuracy = accuracy_score(y_validate, y_pred)
    wandb.log({"accuracy": accuracy})
    return accuracy
    
def save_model(clf:RandomForestClassifier):
    pickle.dump(clf, open('src\\fastapi\model.pkl', 'wb'))
    