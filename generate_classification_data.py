import numpy as np
import csv

def generate_binary_data(n_samples=100, seed=42):
    np.random.seed(seed)
    X = np.random.randn(n_samples, 2)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    return X, y

def save_csv(filename, X, y):
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        header = [f"x_{i}" for i in range(X.shape[1])] + ["y"]
        writer.writerow(header)
        for row, target in zip(X, y):
            writer.writerow(list(row) + [target])

if __name__ == "__main__":
    X, y = generate_binary_data()
    save_csv("data/small_test.csv", X, y)
