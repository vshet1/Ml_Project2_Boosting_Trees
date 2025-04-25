import csv
import numpy as np
import os
import argparse
import matplotlib.pyplot as plt

def generate_data(n_samples=100, noise=0.3, seed=42):
    np.random.seed(seed)
    X = np.random.randn(n_samples, 2)
    y_linear = X[:, 0] + X[:, 1] + noise * np.random.randn(n_samples)
    y = (y_linear > 0).astype(int)
    return X, y

def save_csv(filename, X, y):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["x_0", "x_1", "y"])
        for xi, yi in zip(X, y):
            writer.writerow([xi[0], xi[1], yi])

def plot_dataset(X, y, title="Generated Dataset"):
    plt.figure(figsize=(6, 5))
    for label in np.unique(y):
        plt.scatter(X[y == label, 0], X[y == label, 1],
                    label=f"Class {label}", alpha=0.6)
    plt.title(title)
    plt.xlabel("x_0")
    plt.ylabel("x_1")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate noisy binary classification data.")
    parser.add_argument("--samples", type=int, default=200, help="Number of samples")
    parser.add_argument("--noise", type=float, default=0.6, help="Amount of noise")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output", type=str, default="data/hard_test.csv", help="Output file path")
    parser.add_argument("--visualize", action="store_true", help="Show scatter plot of data")

    args = parser.parse_args()

    X, y = generate_data(n_samples=500, noise=0.4)
    save_csv(args.output, X, y)

    print(f"✅ Generated: {args.output}")
    print(f"Class 0: {np.sum(y==0)}, Class 1: {np.sum(y==1)}")

    if args.visualize:
        plot_dataset(X, y)
