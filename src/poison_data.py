# poison_data.py

import pandas as pd
import numpy as np
import random
import argparse
import os

def poison_labels(input_path, output_path, poison_level):
    df = pd.read_csv(input_path)
    labels = df.iloc[:, -1]  # assume last column is label
    classes = labels.unique()

    num_poison = int(len(df) * poison_level)
    indices = np.random.choice(df.index, size=num_poison, replace=False)

    for idx in indices:
        current = df.loc[idx, df.columns[-1]]
        choices = [c for c in classes if c != current]
        df.loc[idx, df.columns[-1]] = random.choice(choices)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"✅ Poisoned {poison_level*100:.1f}% labels → saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-path", default="data/iris.csv")
    parser.add_argument("--output-path", default="data/iris_poisoned.csv")
    parser.add_argument("--poison-level", type=float, required=True)
    args = parser.parse_args()

    poison_labels(args.input_path, args.output_path, args.poison_level)
