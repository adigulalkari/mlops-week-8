# check_labels.py

import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
import argparse

def find_suspicious_labels(data_path, k=5, threshold=0.6):
    df = pd.read_csv(data_path)
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    knn = KNeighborsClassifier(n_neighbors=k + 1)
    knn.fit(X, y)
    _, indices = knn.kneighbors(X)

    flagged = []
    for i in range(len(df)):
        neighbors = indices[i][1:]
        neighbor_labels = y.iloc[neighbors]
        disagreement = sum(neighbor_labels != y.iloc[i]) / k
        if disagreement >= threshold:
            flagged.append(i)

    print(f"🚨 Found {len(flagged)} suspicious labels out of {len(df)}")
    return flagged

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--threshold", type=float, default=0.6)
    args = parser.parse_args()

    find_suspicious_labels(args.data_path, k=args.k, threshold=args.threshold)
