import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score,
    ConfusionMatrixDisplay,
)
from joblib import dump
from mne.decoding import CSP

# UPDATED import: preprocessing returns filter-bank EEG
from data_pre_extra_features import load_and_process_data


# ============================================================
#  CUSTOM STRATIFIED GROUP K-FOLD (UNCHANGED)
# ============================================================

class StratifiedGroupKFold:
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
            yield np.where(~test_mask)[0], np.where(test_mask)[0]


# ============================================================
#  FBCSP PARAMETERS
# ============================================================

N_CSP_COMPONENTS = 2   # per band


# ============================================================
#  MAIN TRAINING FUNCTION
# ============================================================

def train_xgboost_stratified_group_kfold(folder_path, model_output_path, n_splits=5):

    print("Loading and preprocessing data...")
    X_base, X_eeg, X_eeg_fb, y, file_ids = load_and_process_data(folder_path)

    unique_labels = np.unique(y)
    label_mapping = {
        i: action
        for i, action in enumerate(
            ["resting_sit", "resting_st", "sit_stand", "spin_l", "spin_r", "walking"]
        )
    }

    print(f"Total samples: {len(X_base)}")
    print(f"Unique recordings: {len(np.unique(file_ids))}")
    print(f"Number of filter bands: {X_eeg_fb.shape[1]}")

    sgkf = StratifiedGroupKFold(n_splits=n_splits, random_state=42)

    accuracies, f1_scores = [], []
    sum_conf_matrices = np.zeros((len(unique_labels), len(unique_labels)))

    # ========================================================
    #  CROSS-VALIDATION LOOP
    # ========================================================

    for fold, (train_idx, test_idx) in enumerate(
        sgkf.split(X_base, y, groups=file_ids), start=1
    ):
        print(f"\n================ Fold {fold}/{n_splits} ================")

        X_train_base = X_base[train_idx]
        X_test_base = X_base[test_idx]

        y_train, y_test = y[train_idx], y[test_idx]

        # ====================================================
        #  FBCSP — FIT ONLY ON TRAINING DATA
        # ====================================================

        X_train_csp_list = []
        X_test_csp_list = []

        for band_i in range(X_eeg_fb.shape[1]):
            X_train_band = np.array([X_eeg_fb[i][band_i] for i in train_idx])
            X_test_band = np.array([X_eeg_fb[i][band_i] for i in test_idx])

            csp = CSP(
                n_components=N_CSP_COMPONENTS,
                reg=None,
                log=True,
                norm_trace=False
            )

            X_train_band_csp = csp.fit_transform(X_train_band, y_train)
            X_test_band_csp = csp.transform(X_test_band)

            X_train_csp_list.append(X_train_band_csp)
            X_test_csp_list.append(X_test_band_csp)

        # Concatenate CSP features from all bands
        X_train_fbcsp = np.concatenate(X_train_csp_list, axis=1)
        X_test_fbcsp = np.concatenate(X_test_csp_list, axis=1)

        # ====================================================
        #  CONCATENATE FEATURES
        # ====================================================

        X_train = np.concatenate([X_train_base, X_train_fbcsp], axis=1)
        X_test = np.concatenate([X_test_base, X_test_fbcsp], axis=1)

        print(f"Final feature dimension: {X_train.shape[1]}")

        # ====================================================
        #  TRAIN XGBOOST
        # ====================================================

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

        # ====================================================
        #  EVALUATION
        # ====================================================

        y_pred = xgb_model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

        accuracies.append(acc)
        f1_scores.append(f1)

        print(f"Fold {fold} Accuracy: {acc:.3f}, Weighted F1: {f1:.3f}")

        action_labels = [label_mapping[i] for i in unique_labels]
        print(classification_report(y_test, y_pred, target_names=action_labels, zero_division=0))

        cm = confusion_matrix(y_test, y_pred, labels=unique_labels)
        cm_norm = cm.astype(float) / np.maximum(cm.sum(axis=1, keepdims=True), 1)
        sum_conf_matrices += cm_norm

    # ========================================================
    #  SUMMARY
    # ========================================================

    print("\n================ Summary Across Folds ================")
    print(f"Mean Accuracy: {np.mean(accuracies):.3f} ± {np.std(accuracies):.3f}")
    print(f"Mean Weighted F1: {np.mean(f1_scores):.3f} ± {np.std(f1_scores):.3f}")

    mean_cm = sum_conf_matrices / n_splits
    action_labels = [label_mapping[i] for i in unique_labels]

    disp = ConfusionMatrixDisplay(confusion_matrix=mean_cm, display_labels=action_labels)
    disp.plot(cmap=plt.cm.Blues, xticks_rotation=45, values_format=".2f")
    plt.title("Average Normalized Confusion Matrix — Stratified Group K-Fold (FBCSP)")
    plt.tight_layout()
    plt.show()

    # ========================================================
    #  FINAL MODEL (TRAIN ON ALL DATA)
    # ========================================================

    print("Training final model on full dataset (FBCSP)...")

    X_csp_all_list = []

    for band_i in range(X_eeg_fb.shape[1]):
        X_band_all = np.array([X_eeg_fb[i][band_i] for i in range(len(y))])

        csp = CSP(
            n_components=N_CSP_COMPONENTS,
            reg=None,
            log=True,
            norm_trace=False
        )

        X_band_csp_all = csp.fit_transform(X_band_all, y)
        X_csp_all_list.append(X_band_csp_all)

    X_fbcsp_all = np.concatenate(X_csp_all_list, axis=1)
    X_all = np.concatenate([X_base, X_fbcsp_all], axis=1)

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

    final_model.fit(X_all, y)

    dump(
        {
            "model": final_model,
            "fbcsp_bands": X_eeg_fb.shape[1],
            "csp_components_per_band": N_CSP_COMPONENTS
        },
        model_output_path
    )

    print(f"Final model + FBCSP saved to {model_output_path}")

    return {
        "mean_acc": np.mean(accuracies),
        "std_acc": np.std(accuracies),
        "mean_f1": np.mean(f1_scores),
        "std_f1": np.std(f1_scores),
    }


# ============================================================
#  RUN
# ============================================================

if __name__ == "__main__":
    folder_path = "C:/Users/CHEN/Desktop/תואר שני/תזה/LSTM RNN algorithm/training_data_all_records_all_subjects"
    model_output_path = "xgboost_model_stratified_group_kfold_with_fbcsp.joblib"

    train_xgboost_stratified_group_kfold(
        folder_path,
        model_output_path,
        n_splits=10
    )
