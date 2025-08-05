import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, classification_report

def test_model_accuracy():
    clf = joblib.load("iris_model.joblib")
    X_test = pd.read_csv("X_test.csv")
    y_test = pd.read_csv("y_test.csv")

    preds = clf.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"Model accuracy: {acc}")
    assert acc > 0.7, f"Model accuracy too low: {acc}"

    report = classification_report(y_test, preds)
    print("Classification Report:\n", report)
    with open("test_classification_report.txt", "w") as f:
        f.write(report)

