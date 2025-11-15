import numpy as np
import pandas as pd
import pywt
from sklearn.preprocessing import LabelEncoder
import glob
import os
import re


# === Wavelet feature extraction utilities ===
def compute_features(coeffs):
    """Compute statistical features from wavelet coefficients (length-normalized)."""
    coeffs = np.asarray(coeffs)
    n = len(coeffs)
    energy = np.sum(coeffs ** 2) / n
    entropy = -np.sum((coeffs ** 2) * np.log(coeffs ** 2 + 1e-10)) / n
    mean = np.mean(coeffs)
    std = np.std(coeffs)
    return energy, entropy, mean, std


def extract_wavelet_features(df, wavelet='db4', levels=4):
    """Extract wavelet-based features for each EEG channel."""
    feature_vector = []
    for channel in df.columns:
        signal = df[channel].values
        coeffs = pywt.wavedec(signal, wavelet, level=levels)
        cA_features = compute_features(coeffs[0])
        cD_features = [compute_features(cD) for cD in coeffs[1:]]
        cD_aggregated = np.mean(cD_features, axis=0)
        feature_vector.extend(cA_features)
        feature_vector.extend(cD_aggregated)
    return np.array(feature_vector)


def is_artifact_segment(segment, baseline_std, high_factor=100.0, low_factor=0.00):
    """Detect if a segment contains artifact waves (too high or too low variance)."""
    channel_stds = segment.std(axis=0)
    too_high = np.any(channel_stds > high_factor * baseline_std)
    too_low = np.any(channel_stds < low_factor * baseline_std)
    return too_high or too_low


# === Main preprocessing function ===
def load_and_process_data(
    folder_path,
    wavelet='bior4.4',
    levels=3,
    window_sec=2.0,
    overlap=0.0,
    sampling_rate=128,
):
    """
    Load EEG data from CSV files, apply overlapping-window segmentation, and extract features.

    Args:
        folder_path: Path to folder containing EEG CSV files.
        wavelet: Wavelet type for feature extraction.
        levels: Number of decomposition levels for DWT.
        window_sec: Window size in seconds (default 2.0 s).
        overlap: Overlap ratio between consecutive windows (0.0–0.99).
        sampling_rate: EEG sampling rate in Hz (default 128 Hz).

    Returns:
        feature_matrix: Matrix of extracted features (samples × features).
        labels: Numeric class labels.
    """
    feature_matrix, labels = [], []
    file_ids = []

    valid_eeg_channels = [
        "EEG.F3", "EEG.F4", "EEG.FC5", "EEG.FC6", "EEG.AF3", "EEG.F7",
        "EEG.T7", "EEG.P7", "EEG.O1", "EEG.O2", "EEG.P8", "EEG.T8",
        "EEG.F8", "EEG.AF4"
    ]

    file_paths = glob.glob(os.path.join(folder_path, "*.csv"))
    print(f"Found {len(file_paths)} CSV files.")

    # Derived parameters
    window_samples = int(window_sec * sampling_rate)
    step_samples = int(window_samples * (1 - overlap))
    step_samples = max(step_samples, 1)

    for file_path in file_paths:
        try:
            df = pd.read_csv(file_path, skiprows=1)
        except Exception as e:
            print(f"Error reading file {file_path}: {e}")
            continue

        # Extract label from filename
        match = re.search(r"(resting_sit|resting_st|sit_stand|spin_l|spin_r|walking_)", file_path, re.IGNORECASE)
        if not match:
            print(f"Skipping file (no valid label found): {file_path}")
            continue
        label = match.group(1).lower()

        eeg_channels = [col for col in df.columns if col in valid_eeg_channels]
        if not eeg_channels:
            print(f"No valid EEG channels in file: {file_path}")
            continue
        df = df[eeg_channels].dropna()

        global_median_std = np.median(df.std(axis=0))
        kept, skipped = 0, 0
        n_samples = len(df)

        # Generate overlapping windows (50 % overlap)
        for start in range(0, n_samples - window_samples + 1, step_samples):
            end = start + window_samples
            segment = df.iloc[start:end]

            if is_artifact_segment(segment, baseline_std=global_median_std):
                skipped += 1
                continue

            features = extract_wavelet_features(segment, wavelet=wavelet, levels=levels)
            feature_matrix.append(features)
            labels.append(label)
            file_ids.append(os.path.basename(file_path))

            kept += 1

        print(f"{os.path.basename(file_path)} → windows kept: {kept}, skipped: {skipped}")

    if not feature_matrix:
        raise ValueError("No valid data found after preprocessing.")

    label_encoder = LabelEncoder()
    numeric_labels = label_encoder.fit_transform(labels)

    print(f"Total valid samples: {len(feature_matrix)}")
    return np.array(feature_matrix), np.array(numeric_labels), np.array(file_ids)



# === Example usage ===
if __name__ == "__main__":
    folder_path = "C:/Users/CHEN/Desktop/תואר שני/תזה/LSTM RNN algorithm/training_data_new"

    try:
        features, labels, file_ids = load_and_process_data(
            folder_path,
            window_sec=2.0,
            overlap=0.0,
            sampling_rate=128
        )

        print("Feature extraction completed successfully.")
        print(f"Feature matrix shape: {features.shape}")
        print(f"Labels distribution: {np.bincount(labels)}")
    except Exception as e:
        print(f"Error during preprocessing: {e}")
