# file: best_config_of_preprocessing.py
import itertools
import pandas as pd
import numpy as np
from joblib import dump
from data_pre_xgboost_overlapping import load_and_process_data
from model_training_for_optimatize import train_xgboost_stratified_group_kfold, StratifiedGroupKFold
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score
import os
import time


def evaluate_combinations(folder_path, model_output_prefix):
    """
    Evaluate all combinations of wavelet, decomposition levels, window size, and sampling rate.
    Uses 0 overlap and 10 folds with StratifiedGroupKFold CV.
    """
    # Search grid
    wavelet_functions = ["db4", "sym5", "coif3", "bior4.4"]
    levels = [3, 4, 5]
    window_secs = [1, 2, 3, 5, 20]
    sampling_rates = [128]

    results = []

    for wavelet, level, window_sec, sr in itertools.product(
        wavelet_functions, levels, window_secs, sampling_rates
    ):
        print(f"\n=== Evaluating {wavelet}, levels={level}, window={window_sec}s, sr={sr}Hz ===")

        try:
            # Load and preprocess data
            data, labels, file_ids = load_and_process_data(
                folder_path=folder_path,
                wavelet=wavelet,
                levels=level,
                window_sec=window_sec,
                overlap=0.0,
                sampling_rate=sr,
            )
            print(f"Data shape: {data.shape}, Labels: {len(labels)}")

            # Initialize group-aware CV
            sgkf = StratifiedGroupKFold(n_splits=10, random_state=42)
            accuracies, f1_scores = [], []

            start_time = time.time()

            for fold, (train_idx, test_idx) in enumerate(sgkf.split(data, labels, groups=file_ids), start=1):
                X_train, X_test = data[train_idx], data[test_idx]
                y_train, y_test = labels[train_idx], labels[test_idx]

                model = XGBClassifier(
                    n_estimators=300,
                    learning_rate=0.05,
                    max_depth=6,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    objective="multi:softmax",
                    eval_metric="mlogloss",
                    random_state=42
                )

                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

                acc = accuracy_score(y_test, y_pred)
                f1 = f1_score(y_test, y_pred, average="weighted")
                accuracies.append(acc)
                f1_scores.append(f1)

                print(f"Fold {fold}: Accuracy = {acc:.4f}, Weighted F1 = {f1:.4f}")

            mean_acc = np.mean(accuracies)
            mean_f1 = np.mean(f1_scores)
            duration = time.time() - start_time

            print(f"✅ Mean Accuracy: {mean_acc:.4f} | Mean F1: {mean_f1:.4f} | Runtime: {duration/60:.2f} min")

            results.append({
                "wavelet": wavelet,
                "levels": level,
                "window_sec": window_sec,
                "sampling_rate": sr,
                "mean_acc": mean_acc,
                "std_acc": np.std(accuracies),
                "mean_f1": mean_f1,
                "std_f1": np.std(f1_scores),
                "runtime_sec": duration,
            })

        except Exception as e:
            print(f"❌ Failed for {wavelet}, L{level}, W{window_sec}s, SR={sr}Hz — {e}")
            continue

    # Summarize
    if not results:
        raise RuntimeError("No valid configurations succeeded — check dataset or parameters.")

    df = pd.DataFrame(results)
    df.sort_values(by=["mean_f1", "mean_acc"], ascending=False, inplace=True)
    df.to_csv("wavelet_xgboost_results2.csv", index=False)

    print("\n🏆 Top 5 Configurations:")
    print(df.head(5).to_string(index=False))
    return df


if __name__ == "__main__":
    folder_path = "C:/Users/CHEN/Desktop/תואר שני/תזה/LSTM RNN algorithm/training_data_new"
    model_output_prefix = "xgboost_wavelet_opt2"
    df = evaluate_combinations(folder_path, model_output_prefix)
