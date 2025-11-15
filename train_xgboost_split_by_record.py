import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score,
    ConfusionMatrixDisplay,
)
from sklearn.model_selection import GroupKFold
from joblib import dump
from data_pre_xgboost_overlapping import load_and_process_data  # must return (features, labels, file_ids)

# ==============================
#  CUSTOM STRATIFIED GROUP K-FOLD
# ==============================
class StratifiedGroupKFold:
    """
    Custom implementation of Stratified Group K-Fold cross-validation.
    Ensures:
      - each fold has roughly the same class distribution (stratification)
      - no group (recording) appears in both training and test sets
    """

    def __init__(self, n_splits=5, random_state=None):
        self.n_splits = n_splits
        self.random_state = np.random.RandomState(random_state)

    def split(self, X, y, groups):
        # Compute label distribution per group
        unique_groups, group_indices = np.unique(groups, return_inverse=True)
        n_groups = len(unique_groups)
        n_classes = len(np.unique(y))

        # Count how many samples of each class per group
        group_class_counts = np.zeros((n_groups, n_classes), dtype=int)
        for i, g in enumerate(group_indices):
            group_class_counts[g, y[i]] += 1

        # Target samples per fold (approx equal)
        fold_counts = np.zeros((self.n_splits, n_classes), dtype=int)
        fold_groups = [[] for _ in range(self.n_splits)]

        # Shuffle groups
        order = np.arange(n_groups)
        self.random_state.shuffle(order)

        # Greedy assignment of groups to folds
        for g in order:
            best_fold = np.argmin(np.sum((fold_counts + group_class_counts[g]) ** 2, axis=1))
            fold_counts[best_fold] += group_class_counts[g]
            fold_groups[best_fold].append(unique_groups[g])

        # Yield folds
        for i in range(self.n_splits):
            test_mask = np.isin(groups, fold_groups[i])
            train_idx = np.where(~test_mask)[0]
            test_idx = np.where(test_mask)[0]
            yield train_idx, test_idx


# ==============================
#  MAIN TRAINING FUNCTION
# ==============================
def train_xgboost_stratified_group_kfold(folder_path, model_output_path, n_splits=5):
    """
    Train and evaluate an XGBoost model using Stratified Group K-Fold CV.
    Ensures class balance across folds while keeping recordings grouped.
    """

    try:
        # === Load and preprocess data ===
        print("Loading and preprocessing data...")
        data, labels, file_ids = load_and_process_data(folder_path)

        unique_labels = np.unique(labels)
        label_mapping = {
            i: action
            for i, action in enumerate(
                ["resting_sit", "resting_st", "sit_stand", "spin_l", "spin_r", "walking_"]
            )
        }
        print(f"\nLabel Mapping: {label_mapping}")
        print(f"Total samples: {len(data)}, Unique recordings: {len(np.unique(file_ids))}\n")

        # === Prepare stratified group folds ===
        print("Initializing Stratified Group K-Fold...")
        sgkf = StratifiedGroupKFold(n_splits=n_splits, random_state=42)

        accuracies, f1_scores = [], []
        sum_conf_matrices = np.zeros((len(unique_labels), len(unique_labels)))

        # === Cross-validation loop ===
        for fold, (train_idx, test_idx) in enumerate(sgkf.split(data, labels, groups=file_ids), start=1):
            print(f"\n================ Fold {fold}/{n_splits} ================")

            X_train, X_test = data[train_idx], data[test_idx]
            y_train, y_test = labels[train_idx], labels[test_idx]

            print(f"Training samples: {len(X_train)}, Testing samples: {len(X_test)}")
            print(f"Unique train files: {len(np.unique(file_ids[train_idx]))}, "
                  f"Unique test files: {len(np.unique(file_ids[test_idx]))}")

            # === Train XGBoost model ===
            xgb_model = xgb.XGBClassifier(
                n_estimators=300,
                learning_rate=0.05,
                max_depth=6,
                subsample=0.8,
                colsample_bytree=0.8,
                objective="multi:softmax",
                num_class=len(unique_labels),
                eval_metric="mlogloss",
                random_state=42
            )
            xgb_model.fit(X_train, y_train)

            # === Evaluate fold ===
            y_pred = xgb_model.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
            accuracies.append(acc)
            f1_scores.append(f1)

            print(f"Fold {fold} Accuracy: {acc:.3f}, Weighted F1: {f1:.3f}")

            # === Classification report for this fold ===
            action_labels = [label_mapping[i] for i in unique_labels]
            print("\nClassification Report — Fold", fold)
            print(classification_report(y_test, y_pred, target_names=action_labels, zero_division=0))

            # Also accumulate confusion matrix for average visualization later
            cm = confusion_matrix(y_test, y_pred, labels=unique_labels)
            cm_sum = cm.sum(axis=1, keepdims=True)
            cm_sum[cm_sum == 0] = 1  # avoid divide by zero
            cm_norm = cm.astype(float) / cm_sum
            sum_conf_matrices += cm_norm

        # === Averages across folds ===
        print("\n================ Summary Across 10 Folds ================")
        print(f"Mean Accuracy: {np.mean(accuracies):.3f} ± {np.std(accuracies):.3f}")
        print(f"Mean Weighted F1: {np.mean(f1_scores):.3f} ± {np.std(f1_scores):.3f}")
        print("======================================================\n")

        # === Average confusion matrix ===
        mean_cm = sum_conf_matrices / n_splits
        action_labels = [label_mapping[i] for i in unique_labels]

        disp = ConfusionMatrixDisplay(confusion_matrix=mean_cm, display_labels=action_labels)
        disp.plot(cmap=plt.cm.Blues, xticks_rotation=45, values_format=".2f")
        plt.title("Average Normalized Confusion Matrix — Stratified Group K-Fold")
        plt.tight_layout()
        plt.show()

        # === Retrain final model on all data ===
        print("Retraining final model on full dataset...")
        final_model = xgb.XGBClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="multi:softmax",
            num_class=len(unique_labels),
            eval_metric="mlogloss",
            random_state=42
        )
        final_model.fit(data, labels)

        print(f"Saving final model to {model_output_path}...")
        dump(final_model, model_output_path)
        print("Final model saved successfully.\n")
        return {
            "mean_acc": np.mean(accuracies),
            "std_acc": np.std(accuracies),
            "mean_f1": np.mean(f1_scores),
            "std_f1": np.std(f1_scores),
        }

    except Exception as e:
        print(f"Error during training: {e}")


# === Run training ===
if __name__ == "__main__":
    folder_path = "C:/Users/CHEN/Desktop/תואר שני/תזה/LSTM RNN algorithm/training_data_new"
    model_output_path = "xgboost_model_stratified_group_kfold.joblib"
    train_xgboost_stratified_group_kfold(folder_path, model_output_path, n_splits=10)
