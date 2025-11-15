# file: train_xgboost_stratified_group_kfold.py
import numpy as np
import xgboost as xgb
from joblib import dump
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from data_pre_xgboost_overlapping import load_and_process_data


# ===================================
#  CUSTOM STRATIFIED GROUP K-FOLD
# ===================================
class StratifiedGroupKFold:
    """Stratified Group K-Fold CV ensuring balanced class distribution and group exclusivity."""

    def __init__(self, n_splits=5, random_state=None):
        self.n_splits = n_splits
        self.random_state = np.random.RandomState(random_state)

    def split(self, X, y, groups):
        unique_groups, group_indices = np.unique(groups, return_inverse=True)
        n_groups = len(unique_groups)
        n_classes = len(np.unique(y))

        group_class_counts = np.zeros((n_groups, n_classes), dtype=int)
        for i, g in enumerate(group_indices):
            group_class_counts[g, y[i]] += 1

        fold_counts = np.zeros((self.n_splits, n_classes), dtype=int)
        fold_groups = [[] for _ in range(self.n_splits)]

        order = np.arange(n_groups)
        self.random_state.shuffle(order)

        for g in order:
            best_fold = np.argmin(np.sum((fold_counts + group_class_counts[g]) ** 2, axis=1))
            fold_counts[best_fold] += group_class_counts[g]
            fold_groups[best_fold].append(unique_groups[g])

        for i in range(self.n_splits):
            test_mask = np.isin(groups, fold_groups[i])
            train_idx = np.where(~test_mask)[0]
            test_idx = np.where(test_mask)[0]
            yield train_idx, test_idx


# ===================================
#  MAIN TRAINING FUNCTION
# ===================================
def train_xgboost_stratified_group_kfold(
    folder_path=None,
    model_output_path="xgboost_model_stratified_group_kfold.joblib",
    n_splits=5,
    data=None,
    labels=None,
    file_ids=None,
):
    """
    Train and evaluate an XGBoost model using Stratified Group K-Fold CV.
    If data, labels, and file_ids are provided, skips preprocessing.

    Args:
        folder_path: path to folder with data (if preprocessing is needed)
        model_output_path: path to save the final trained model
        n_splits: number of folds (default 5)
        data, labels, file_ids: optional preprocessed arrays

    Returns:
        dict: {'mean_acc', 'std_acc', 'mean_f1', 'std_f1'}
    """
    try:
        # === Data loading ===
        if data is None or labels is None or file_ids is None:
            print("Loading and preprocessing data...")
            data, labels, file_ids = load_and_process_data(folder_path)
        else:
            print("Using preprocessed data provided by caller...")

        unique_labels = np.unique(labels)
        label_mapping = {
            0: "resting_sit",
            1: "resting_st",
            2: "sit_stand",
            3: "spin_l",
            4: "spin_r",
            5: "walking_",
        }

        print(f"Total samples: {len(data)}, Unique recordings: {len(np.unique(file_ids))}")
        print(f"Label mapping: {label_mapping}\n")

        # === Cross-validation ===
        sgkf = StratifiedGroupKFold(n_splits=n_splits, random_state=42)
        accuracies, f1_scores = [], []

        for fold, (train_idx, test_idx) in enumerate(sgkf.split(data, labels, groups=file_ids), start=1):
            print(f"\n================ Fold {fold}/{n_splits} ================")
            X_train, X_test = data[train_idx], data[test_idx]
            y_train, y_test = labels[train_idx], labels[test_idx]

            print(f"Training samples: {len(X_train)}, Testing samples: {len(X_test)}")

            # === Model definition ===
            model = xgb.XGBClassifier(
                n_estimators=300,
                learning_rate=0.05,
                max_depth=6,
                subsample=0.8,
                colsample_bytree=0.8,
                objective="multi:softmax",
                num_class=len(unique_labels),
                eval_metric="mlogloss",
                random_state=42,
                n_jobs=-1,
            )

            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            acc = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
            accuracies.append(acc)
            f1_scores.append(f1)

            print(f"Fold {fold} → Accuracy: {acc:.3f}, Weighted F1: {f1:.3f}")
            print(classification_report(y_test, y_pred, zero_division=0))

        # === Average metrics ===
        mean_acc, mean_f1 = np.mean(accuracies), np.mean(f1_scores)
        std_acc, std_f1 = np.std(accuracies), np.std(f1_scores)
        print("\n================ Summary ================")
        print(f"Mean Accuracy: {mean_acc:.3f} ± {std_acc:.3f}")
        print(f"Mean Weighted F1: {mean_f1:.3f} ± {std_f1:.3f}")
        print("=========================================\n")

        # === Retrain on full dataset ===
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
            random_state=42,
            n_jobs=-1,
        )
        final_model.fit(data, labels)
        dump(final_model, model_output_path)
        print(f"✅ Final model saved to {model_output_path}\n")

        return {
            "mean_acc": mean_acc,
            "std_acc": std_acc,
            "mean_f1": mean_f1,
            "std_f1": std_f1,
        }

    except Exception as e:
        print(f"❌ Error during training: {e}")
        return None


# === Run standalone training ===
if __name__ == "__main__":
    folder_path = "C:/Users/CHEN/Desktop/תואר שני/תזה/LSTM RNN algorithm/training_data_new"
    model_output_path = "xgboost_model_stratified_group_kfold.joblib"
    metrics = train_xgboost_stratified_group_kfold(
        folder_path=folder_path,
        model_output_path=model_output_path,
        n_splits=10,
    )
    print("\nTraining completed.")
