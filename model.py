from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, mean_squared_error
import pickle

def model(X_train:pd.DataFrame, y_train:pd.DataFrame):
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    return clf

def predict(X_validate:pd.DataFrame, clf:RandomForestClassifier):
    y_pred = clf.predict(X_validate)
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy
    
def dump(clf:RandomForestClassifier):
    pickle.dump(clf, open('model.pkl', 'wb'))
