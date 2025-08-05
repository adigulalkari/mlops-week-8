import pandas as pd
import joblib
from sklearn.metrics import accuracy_score

def main():
    clf = joblib.load("iris_model.joblib")
    X_test = pd.read_csv("X_test.csv")
    y_test = pd.read_csv("y_test.csv")

    preds = clf.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"Test Accuracy: {acc:.4f}")

    with open("metrics.txt", "w") as f:
        f.write(f"accuracy: {acc:.4f}\n")

if __name__ == "__main__":
    main()

