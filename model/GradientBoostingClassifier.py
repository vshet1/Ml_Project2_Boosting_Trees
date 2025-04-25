import numpy as np

class DecisionStump:
    def __init__(self):
        self.feature_index = None
        self.threshold = None
        self.left_value = None
        self.right_value = None

    def fit(self, X, residuals):
        n_samples, n_features = X.shape
        best_loss = float("inf")
        for feature in range(n_features):
            thresholds = np.unique(X[:, feature])
            for threshold in thresholds:
                left = residuals[X[:, feature] <= threshold]
                right = residuals[X[:, feature] > threshold]
                left_mean = np.mean(left) if len(left) > 0 else 0
                right_mean = np.mean(right) if len(right) > 0 else 0
                loss = np.sum((left - left_mean)**2) + np.sum((right - right_mean)**2)
                if loss < best_loss:
                    best_loss = loss
                    self.feature_index = feature
                    self.threshold = threshold
                    self.left_value = left_mean
                    self.right_value = right_mean

    def predict(self, X):
        preds = np.where(X[:, self.feature_index] <= self.threshold,
                         self.left_value, self.right_value)
        return preds


class GradientBoostingClassifier:
    def __init__(self, n_estimators=100, learning_rate=0.1):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.models = []
        self.init_pred = None
        self.accuracy_curve = []

    def fit(self, X, y):
        y = y * 2 - 1  # Convert {0, 1} to {-1, 1}
        self.init_pred = np.zeros(len(y))
        y_pred = (self.init_pred > 0).astype(int)
        accuracy = np.mean(y_pred == ((y + 1) // 2))  # since y ∈ {-1, 1}
        self.accuracy_curve.append(accuracy)

        early_stop_rounds = 20
        best_acc = 0
        rounds_since_improvement = 0

        for i in range(self.n_estimators):
            residuals = y / (1 + np.exp(y * self.init_pred))
            stump = DecisionStump()
            stump.fit(X, residuals)
            self.models.append(stump)
            self.init_pred += self.learning_rate * stump.predict(X)
            
            y_pred = (self.init_pred > 0).astype(int)
            true_y = ((y + 1) // 2)
            accuracy = np.mean(y_pred == true_y)
            self.accuracy_curve.append(accuracy)
            
            if accuracy > best_acc:
                best_acc = accuracy
                rounds_since_improvement = 0
            else:
                rounds_since_improvement += 1
            
            if rounds_since_improvement >= early_stop_rounds:
                print(f"Early stopping at round {i} with best accuracy: {best_acc}")
                break
        
        # for i in range(self.n_estimators):
        #     residuals = y / (1 + np.exp(y * self.init_pred))
        #     stump = DecisionStump()
        #     stump.fit(X, residuals)
        #     self.models.append(stump)
        #     self.init_pred += self.learning_rate * stump.predict(X)
        #     # Track training accuracy
        #     y_pred = (self.init_pred > 0).astype(int)
        #     true_y = ((y + 1) // 2) if set(y) == {-1, 1} else y
        #     accuracy = np.mean(y_pred == true_y)
        #     self.accuracy_curve.append(accuracy)


    def predict(self, X):
        pred = np.zeros(X.shape[0])
        for model in self.models:
            pred += self.learning_rate * model.predict(X)

        probs = 1 / (1 + np.exp(-pred))        # Logistic sigmoid
        return (probs > 0.5).astype(int)

# import numpy as np

# class Depth2Tree:
#     def __init__(self):
#         self.split_feature_1 = None
#         self.threshold_1 = None
#         self.split_feature_2 = None
#         self.threshold_2 = None
#         self.leaf_values = {}

#     def fit(self, X, residuals):
#         n_samples, n_features = X.shape
#         best_loss = float('inf')

#         for f1 in range(n_features):
#             for t1 in np.unique(X[:, f1]):
#                 left_idx = X[:, f1] <= t1
#                 right_idx = ~left_idx

#                 for f2 in range(n_features):
#                     for t2 in np.unique(X[:, f2]):
#                         loss = 0
#                         leafs = {}

#                         for quadrant in [(True, True), (True, False), (False, True), (False, False)]:
#                             q_idx = (X[:, f1] <= t1 if quadrant[0] else X[:, f1] > t1) & \
#                                     (X[:, f2] <= t2 if quadrant[1] else X[:, f2] > t2)
#                             if np.any(q_idx):
#                                 leaf_mean = np.mean(residuals[q_idx])
#                                 leafs[quadrant] = leaf_mean
#                                 loss += np.sum((residuals[q_idx] - leaf_mean) ** 2)

#                         if loss < best_loss:
#                             best_loss = loss
#                             self.split_feature_1 = f1
#                             self.threshold_1 = t1
#                             self.split_feature_2 = f2
#                             self.threshold_2 = t2
#                             self.leaf_values = leafs

#     def predict(self, X):
#         preds = np.zeros(X.shape[0])
#         for i in range(X.shape[0]):
#             f1 = X[i, self.split_feature_1] <= self.threshold_1
#             f2 = X[i, self.split_feature_2] <= self.threshold_2
#             preds[i] = self.leaf_values[(f1, f2)]
#         return preds


# class GradientBoostingClassifier:
#     def __init__(self, n_estimators=100, learning_rate=0.1, subsample=0.8, early_stop_rounds=20):
#         self.n_estimators = n_estimators
#         self.learning_rate = learning_rate
#         self.subsample = subsample
#         self.early_stop_rounds = early_stop_rounds

#         self.models = []
#         self.init_pred = None
#         self.accuracy_curve = []

#     def fit(self, X, y):
#         y = y * 2 - 1  # Convert {0, 1} to {-1, 1}
#         n_samples = X.shape[0]
#         self.init_pred = np.zeros(n_samples)
#         y_pred = (self.init_pred > 0).astype(int)
#         true_y = ((y + 1) // 2)

#         accuracy = np.mean(y_pred == true_y)
#         self.accuracy_curve.append(accuracy)

#         best_acc = 0
#         rounds_since_improvement = 0

#         for i in range(self.n_estimators):
#             # Logistic loss gradient
#             residuals = y / (1 + np.exp(y * self.init_pred))

#             # Row Subsampling
#             sample_idx = np.random.choice(n_samples, int(self.subsample * n_samples), replace=False)
#             X_sample = X[sample_idx]
#             residuals_sample = residuals[sample_idx]

#             # Train Depth2Tree instead of DecisionStump
#             tree = Depth2Tree()
#             tree.fit(X_sample, residuals_sample)

#             self.models.append(tree)
#             self.init_pred += self.learning_rate * tree.predict(X)

#             # Accuracy on full training data
#             y_pred = (self.init_pred > 0).astype(int)
#             accuracy = np.mean(y_pred == true_y)
#             self.accuracy_curve.append(accuracy)

#             # Early stopping
#             if accuracy > best_acc:
#                 best_acc = accuracy
#                 rounds_since_improvement = 0
#             else:
#                 rounds_since_improvement += 1

#             if rounds_since_improvement >= self.early_stop_rounds:
#                 print(f"Early stopping at round {i + 1} with best training accuracy: {best_acc:.4f}")
#                 break

#     def predict(self, X):
#         pred = np.zeros(X.shape[0])
#         for model in self.models:
#             pred += self.learning_rate * model.predict(X)

#         probs = 1 / (1 + np.exp(-pred))  # Logistic sigmoid
#         return (probs > 0.5).astype(int)
