import csv
import matplotlib.pyplot as plt
import numpy as np
from model.GradientBoostingClassifier import GradientBoostingClassifier
from visualizations.plot_decision_boundary import plot_decision_boundary
from visualizations.plot_accuracy_curve import plot_accuracy_curve
from visualizations.plot_prediction_results import plot_prediction_results

def load_data(filepath):
    with open(filepath, "r") as f:
        reader = csv.DictReader(f)
        data = [row for row in reader]
    X = np.array([[float(row[k]) for k in row if k.startswith("x_")] for row in data])
    y = np.array([int(row["y"]) for row in data])  # Don't use median threshold here
    return X, y


def train_test_split_manual(X, y, test_ratio=0.2, seed=42):
    """Manual NumPy-based train/test split"""
    np.random.seed(seed)
    indices = np.random.permutation(len(X))
    split_idx = int(len(X) * (1 - test_ratio))
    train_idx, test_idx = indices[:split_idx], indices[split_idx:]
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]

def plot_confusion_matrix(cm, class_names=["Class 0", "Class 1"], title="Confusion Matrix"):
    fig, ax = plt.subplots()
    cax = ax.matshow(cm, cmap="Blues")
    plt.title(title, pad=20)
    fig.colorbar(cax)

    ax.set_xticklabels([""] + class_names)
    ax.set_yticklabels([""] + class_names)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    # Annotate each cell with count
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), va='center', ha='center', fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.show()
    
# === Load & Prepare Data ===
X, y = load_data("data/hard_test.csv")
print(f"Class 0 count: {np.sum(y == 0)}")
print(f"Class 1 count: {np.sum(y == 1)}")

X_train, X_test, y_train, y_test = train_test_split_manual(X, y, test_ratio=0.2)

# === Train Model ===
model = GradientBoostingClassifier(n_estimators=500, learning_rate=0.3)
model.fit(X_train, y_train)

# === Evaluate Model ===
train_preds = model.predict(X_train)
test_preds = model.predict(X_test)

train_acc = np.mean(train_preds == y_train)
test_acc = np.mean(test_preds == y_test)



print(f"Training Accuracy: {train_acc:.2f}")
print(f"Test Accuracy: {test_acc:.2f}")

# match_count = np.sum(test_preds == y_test)
# print(f"\n Matched {match_count} out of {len(y_test)} test samples")
# print(f"Prediction Error: {1 - test_acc:.2f}")

# ✅ Class-specific accuracy
class_0_acc = np.mean(test_preds[y_test == 0] == 0)
class_1_acc = np.mean(test_preds[y_test == 1] == 1)
print(f"Class 0 Accuracy: {class_0_acc:.2f}")
print(f"Class 1 Accuracy: {class_1_acc:.2f}")

# ✅ Optional prediction correctness summary
match_count = np.sum(test_preds == y_test)
print(f"\n✅ Matched {match_count} out of {len(y_test)} test samples")
print(f"🔍 Prediction Error: {1 - test_acc:.2f}")

# === Visualizations ===
if X.shape[1] == 2:
    plot_decision_boundary(model, X_train, y_train)

plot_accuracy_curve(model.accuracy_curve)
plot_prediction_results(X_train, y_train, train_preds, title="Train Set Predictions")
plot_prediction_results(X_test, y_test, test_preds, title="Test Set Predictions")

print("Test Predictions:", test_preds[:10])
print("Actual Labels   :", y_test[:10])
