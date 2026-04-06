# SL-FEC
````markdown
# SL-FEC: Staged Learning Framework for Fine-grained Emotion Classification

## 📌 Overview
SL-FEC (Staged Learning Framework for Fine-grained Emotion Classification) is a two-stage training framework designed to improve fine-grained emotion classification performance. It decomposes the task into:

1. **Stage 1 (Coarse-grained)**: Binary emotion classification (positive vs. negative)
2. **Stage 2 (Fine-grained)**: Multi-class emotion classification

This coarse-to-fine paradigm improves representation structure and enhances discrimination among subtle emotion categories.

---

## 🗂️ Datasets

We use the following publicly available datasets:

### 1. Empathetic Dialogues (ED)
- Paper: *Towards Empathetic Open-domain Conversation Models*
- Link: https://github.com/facebookresearch/EmpatheticDialogues

### 2. GoEmotions
- Paper: *GoEmotions: A Dataset of Fine-Grained Emotions*
- Link: https://github.com/google-research/google-research/tree/master/goemotions

---

## 🤖 Pretrained Models

We use **RoBERTa / BERT-based models** from HuggingFace:

- RoBERTa-base:
  - https://huggingface.co/roberta-base

- BERT-base-uncased:
  - https://huggingface.co/bert-base-uncased

You can download and place the model locally or load it directly via HuggingFace Transformers.

---

## ⚙️ Environment

### 🔧 Framework
- Python: 3.9+
- PyTorch: **2.1.0**
- Transformers: **4.36.0**
- Scikit-learn: **1.3.0**
- Pandas: **2.0.3**
- NumPy: **1.24.4**

### ⚡ CUDA
- CUDA Version: **12.1**
- cuDNN: **8.x**

### 🖥️ GPU
- GPU: **NVIDIA RTX 3090 / A100 (recommended)**
- VRAM: **≥ 24GB recommended (≥ 12GB minimum with gradient accumulation)**

---

## 🚀 Training Pipeline

### Stage 1: Binary Classification
- Objective: Learn coarse emotion polarity
- Loss: CrossEntropyLoss
- Epochs: 1

### Stage 2: Multi-class Classification
- Objective: Fine-grained emotion classification
- Loss:
  - CrossEntropy (with class weights)
  - Optional: Focal Loss (for long-tail distribution)
- Strategy:
  - Freeze lower encoder layers initially
  - Gradually unfreeze all layers
  - Early stopping based on Macro-F1

---

## 📊 Key Features

- ✅ Two-stage training (coarse → fine)
- ✅ Class imbalance handling (log-weight + focal loss)
- ✅ Gradient accumulation for memory efficiency
- ✅ Mixed precision training (AMP)
- ✅ Layer freezing/unfreezing strategy
- ✅ Early stopping based on Macro-F1

---

## 📈 Evaluation Metrics

- Accuracy (ACC)
- Weighted F1 (WF1)
- Macro F1 (MF1) ← **main metric**

---

## 📦 Usage

### 1. Install dependencies
```bash
pip install torch transformers pandas scikit-learn tqdm
````

### 2. Prepare dataset

Place dataset files in:

```
data/
 ├── train.csv
 ├── valid.csv
 └── test.csv
```

### 3. Run training

```bash
python train.py
```

---

## 📌 Notes

* Recommended to enable **mixed precision (AMP)** for faster training.
* If GPU memory is limited:

  * Reduce `batch_size`
  * Reduce `max_len`
  * Increase `grad_accum_steps`

---

## 📄 Citation

If you find this work useful, please cite:

```
@article{slfec2026,
  title={SL-FEC: Staged Learning Framework for Fine-grained Emotion Classification},
  author={Your Name},
  year={2026}
}
```

---

## 📬 Contact

For questions or collaboration, please open an issue or contact the author.

```
```
