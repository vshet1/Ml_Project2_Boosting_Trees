# visualizations/plot_prediction_results.py
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np

def plot_prediction_results(X, y_true, y_pred, title="Predictions on Training Data"):
    plt.figure(figsize=(7, 5))
    markers = ['o', 'x']
    colors = ['red', 'green']

    for label in np.unique(y_true):
        idx = np.where(y_true == label)
        plt.scatter(X[idx, 0], X[idx, 1],
                    c=[colors[label]],
                    label=f"True {label}",
                    marker=markers[label],
                    edgecolor=None)

    for label in np.unique(y_pred):
        idx = np.where(y_pred == label)
        plt.scatter(X[idx, 0], X[idx, 1],
                    facecolors='none',
                    edgecolors=colors[label],
                    marker='s',
                    label=f"Pred {label}")

    plt.legend()
    plt.title(title)
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.grid(True)
    plt.show()
