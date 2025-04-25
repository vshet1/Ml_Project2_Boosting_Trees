import matplotlib.pyplot as plt

def plot_accuracy_curve(acc_curve):
    plt.plot(acc_curve, marker='o')
    plt.title("Training Accuracy per Iteration")
    plt.xlabel("Boosting Round")
    plt.ylabel("Accuracy")
    plt.grid(True)
    plt.show()
