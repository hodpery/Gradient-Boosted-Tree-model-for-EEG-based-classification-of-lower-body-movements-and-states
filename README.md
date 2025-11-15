# EEG Lower-Body Movement Dataset and Machine Learning Classification Pipeline

This repository contains an EEG dataset and a full machine-learning pipeline for classifying lower-body movements using DWT-based statistical features and a Gradient Boosted Tree (GBT) model. The dataset was collected using the EMOTIV EPOC X EEG headset and includes six movement classes performed across multiple sessions. All preprocessing, model training, evaluation, and optimization scripts are included.

---

## 📂 Repository Overview

📦 dataset/                # EEG recordings (.csv), segmented windows, metadata  
📦 scripts/
│    ├── train_xgboost_split_by_record.py
│    ├── data_pre_xgboost_overlapping.py
│    ├── best_config_of_preprocessing.py
│    ├── model_training_for_optimize.py
│
📦 results/
│    ├── confusion_matrices/
│    ├── wavelet_optimization_table.csv
│    ├── feature_importance_plots/
│
📄 README.md
📄 .gitignore

---

# 🧠 Dataset Description

### **Recording Device**  
- EMOTIV EPOC X (14 EEG channels + 2 reference electrodes)  
- Sampling Rate: 128 Hz  

### **Subject Information**  
The dataset was recorded from **one subject**, which is intentional since EEG movement-related patterns vary strongly across individuals. In practical exoskeleton systems, personalized calibration is required; therefore, this dataset is intended as a subject-specific training example.

---

# 🦵 Recorded Lower-Body Movements

Each movement was recorded in **30 sessions**, each lasting **60 seconds**:

1. Resting Stand  
2. Resting Sit  
3. Walking  
4. Standing–Sitting Transition  
5. Spin Left  
6. Spin Right  

Total data:  
- 180 recordings  
- 10,800 seconds of EEG  

CSV includes timestamp, counter, and 14 µV EEG channels.

---

# 🧩 Preprocessing Pipeline

### 1. Segmentation  
- 2-second non-overlapping windows  
- Empirically chosen as the optimal balance between performance and responsiveness  

### 2. Feature Extraction (DWT)  
Best configuration (from optimization experiments):  
- Wavelet: **Bior4.4**  
- Levels: **3**  
- Window length: **2 seconds**

Extracted features:  
- Energy  
- Entropy  
- Mean  
- Standard deviation  

---

# 🤖 Machine Learning

### Classifier  
**Gradient Boosted Tree (GBT)** chosen for:  
- High performance with moderate datasets  
- Robustness to noise  
- Good generalization compared to neural networks  

### Performance  
- 6 classes → **0.742 ± 0.035**  
- 6 classes + 90% overlap → **0.781 ± 0.043**  
- 3 classes → **0.878 ± 0.037**  
- 2 classes → **0.949 ± 0.035**

---

# 📜 Script Descriptions

### `train_xgboost_split_by_record.py`  
Main training script. Ensures **splitting by full recording** to avoid leakage.

### `data_pre_xgboost_overlapping.py`  
Creates **overlapping 2-second windows** for overlap experiments (20%, 50%, 90%).

### `best_config_of_preprocessing.py`  
Applies the **final chosen DWT configuration** and generates training-ready features.

### `model_training_for_optimize.py`  
Runs full optimization:  
- db4, bior4.4, coif3, sym5  
- Window sizes 0.5–20 sec  
- Levels 1–6  
Saves comparison table.

---

# 📈 Results Included

- Confusion matrices (0% overlap & 90% overlap)  
- Full wavelet optimization table  
- Feature importance plots  

---

# 👉 Usage

### Preprocess with final config:
```bash
python scripts/best_config_of_preprocessing.py
```

### Train main model:
```bash
python scripts/train_xgboost_split_by_record.py
```

### Create overlapping dataset:
```bash
python scripts/data_pre_xgboost_overlapping.py
```

### Run wavelet optimization:
```bash
python scripts/model_training_for_optimize.py
```

---

# 📚 Citation

If you use this dataset or code, please cite:

```
Pery, H. (2025). Gradient Boosted Tree model for EEG-based classification 
of lower-body movements and states. GitHub Repository. 
https://github.com/hodpery/Gradient-Boosted-Tree-model-for-EEG-based-classification-of-lower-body-movements-and-states

```

---

# 🛡️ License
Released under the MIT License.

