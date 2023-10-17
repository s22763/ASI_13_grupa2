from download import download
from preprocess import preprocess
from model import create_model, predict, save_model


df = download()
X_train, X_test, X_validate, y_train, y_test, y_validate = preprocess(df)

clf = create_model(X_train, y_train)
accuracy = predict(X_validate, y_validate, clf)
print("Accuracy:",accuracy)

save_model(clf)
