import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
import joblib
import os

def main():
    data_path = os.getenv("DATA_PATH", "data/iris.csv")
    df = pd.read_csv(data_path)
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    clf = LogisticRegression(max_iter=200)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    print("Classification Report:\n", report)
    print(f"Accuracy: {acc:.4f}")

    joblib.dump(clf, "iris_model.joblib")
    pd.DataFrame(X_test).to_csv("X_test.csv", index=False)
    pd.DataFrame(y_test).to_csv("y_test.csv", index=False)

    print("✅ Training complete")

if __name__ == "__main__":
    main()
