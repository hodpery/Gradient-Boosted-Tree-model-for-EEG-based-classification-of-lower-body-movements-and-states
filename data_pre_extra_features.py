import numpy as np
import pandas as pd
import pywt
from sklearn.preprocessing import LabelEncoder
from scipy.signal import welch, butter, filtfilt
import glob
import os
import re


# ============================================================
# === Wavelet feature extraction utilities (UNCHANGED)
# ============================================================

def compute_features(coeffs):
    coeffs = np.asarray(coeffs)
    n = len(coeffs)
    energy = np.sum(coeffs ** 2) / n
    entropy = -np.sum((coeffs ** 2) * np.log(coeffs ** 2 + 1e-10)) / n
    mean = np.mean(coeffs)
    std = np.std(coeffs)
    return energy, entropy, mean, std


def extract_wavelet_features(df, wavelet='bior4.4', levels=3):
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


# ============================================================
# === μ / β bandpower utilities (UNCHANGED)
# ============================================================

def compute_bandpower(signal, fs, band):
    f, psd = welch(signal, fs=fs, nperseg=len(signal))
    band_mask = (f >= band[0]) & (f <= band[1])
    return np.trapz(psd[band_mask], f[band_mask])


# ============================================================
# === Artifact rejection (UNCHANGED)
# ============================================================

def is_artifact_segment(segment, baseline_std, high_factor=100.0, low_factor=0.00):
    channel_stds = segment.std(axis=0)
    too_high = np.any(channel_stds > high_factor * baseline_std)
    too_low = np.any(channel_stds < low_factor * baseline_std)
    return too_high or too_low


# ============================================================
# === FBCSP filter-bank utilities (NEW)
# ============================================================

FBCSP_BANDS = [
    (4, 8),     # theta
    (8, 12),    # mu
    (12, 16),
    (16, 20),
    (20, 30),   # beta
]


def bandpass_filter(signal, fs, low, high, order=4):
    nyq = fs / 2.0
    b, a = butter(order, [low / nyq, high / nyq], btype="band")
    return filtfilt(b, a, signal)


def create_filter_bank(eeg_window, fs):
    """
    eeg_window: (channels, samples)
    returns: (n_bands, channels, samples)
    """
    fb = []
    for low, high in FBCSP_BANDS:
        filtered = np.array([
            bandpass_filter(ch, fs, low, high)
            for ch in eeg_window
        ])
        fb.append(filtered)
    return np.array(fb)


# ============================================================
# === Main preprocessing function (ADAPTED)
# ============================================================

def load_and_process_data(
    folder_path,
    wavelet='bior4.4',
    levels=3,
    window_sec=2.0,
    overlap=0.0,
    sampling_rate=128,
):
    """
    Returns:
        X_features     : DWT + μ/β features
        X_eeg_windows  : Raw EEG windows (N, channels, samples) → CSP
        X_eeg_fb       : Filter-bank EEG (N, bands, channels, samples) → FBCSP
        y              : Numeric labels
        file_ids       : Recording IDs
    """

    feature_matrix = []
    eeg_windows = []
    eeg_fb_windows = []
    labels = []
    file_ids = []

    valid_eeg_channels = [
        "EEG.F3", "EEG.F4", "EEG.FC5", "EEG.FC6",
        "EEG.AF3", "EEG.F7", "EEG.T7", "EEG.P7",
        "EEG.O1", "EEG.O2", "EEG.P8", "EEG.T8",
        "EEG.F8", "EEG.AF4"
    ]

    file_paths = glob.glob(os.path.join(folder_path, "*.csv"))
    print(f"Found {len(file_paths)} CSV files.")

    window_samples = int(window_sec * sampling_rate)
    step_samples = max(int(window_samples * (1 - overlap)), 1)

    for file_path in file_paths:
        try:
            df = pd.read_csv(file_path, skiprows=1)
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            continue

        match = re.search(
            r"(resting_sit|resting_st|sit_stand|spin_l|spin_r|walking)",
            file_path,
            re.IGNORECASE
        )
        if not match:
            continue

        label = match.group(1).lower()

        eeg_channels = [c for c in df.columns if c in valid_eeg_channels]
        if not eeg_channels:
            continue

        df = df[eeg_channels].dropna()
        baseline_std = np.median(df.std(axis=0))

        kept, skipped = 0, 0

        for start in range(0, len(df) - window_samples + 1, step_samples):
            segment = df.iloc[start:start + window_samples]

            if is_artifact_segment(segment, baseline_std):
                skipped += 1
                continue

            # =============================
            # 1️⃣ DWT FEATURES
            # =============================
            dwt_features = extract_wavelet_features(
                segment, wavelet=wavelet, levels=levels
            )

            # =============================
            # 2️⃣ μ / β BANDPOWER
            # =============================
            bandpower_features = []
            for ch in segment.columns:
                sig = segment[ch].values
                mu = compute_bandpower(sig, sampling_rate, (8, 12))
                beta = compute_bandpower(sig, sampling_rate, (13, 30))
                bandpower_features.extend([mu, beta])

            feature_matrix.append(
                np.concatenate([dwt_features, bandpower_features])
            )

            # =============================
            # 3️⃣ RAW EEG WINDOW (CSP)
            # =============================
            eeg_window = segment.values.T  # (channels, samples)
            eeg_windows.append(eeg_window)

            # =============================
            # 4️⃣ FILTER-BANK EEG (FBCSP)
            # =============================
            eeg_fb_windows.append(
                create_filter_bank(eeg_window, sampling_rate)
            )

            labels.append(label)
            file_ids.append(os.path.basename(file_path))
            kept += 1

        print(f"{os.path.basename(file_path)} → kept: {kept}, skipped: {skipped}")

    if not feature_matrix:
        raise ValueError("No valid EEG windows found.")

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(labels)

    print(f"Total valid samples: {len(feature_matrix)}")
    print(f"Filter-bank bands: {len(FBCSP_BANDS)}")

    return (
        np.array(feature_matrix),
        np.array(eeg_windows),
        np.array(eeg_fb_windows),
        np.array(y),
        np.array(file_ids)
    )


# ============================================================
# === Example usage
# ============================================================

if __name__ == "__main__":
    folder_path = "C:/Users/CHEN/Desktop/תואר שני/תזה/LSTM RNN algorithm/training_data_all_records_all_subjects"

    X, X_eeg, X_eeg_fb, y, file_ids = load_and_process_data(
        folder_path,
        window_sec=2.0,
        overlap=0.0,
        sampling_rate=128
    )

    print("Preprocessing completed.")
    print(f"Feature matrix shape: {X.shape}")
    print(f"EEG windows shape (CSP): {X_eeg.shape}")
    print(f"Filter-bank EEG shape (FBCSP): {X_eeg_fb.shape}")
    print(f"Label distribution: {np.bincount(y)}")
