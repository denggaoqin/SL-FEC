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

We use **RoBERTa models** from HuggingFace:

- RoBERTa-base:
  - https://huggingface.co/roberta-base

You can download and place the model locally or load it directly via HuggingFace Transformers.

---

## ⚙️ Environment

### 🔧 Framework
- Python: 3.12
- PyTorch: **1.11.6**

### 🖥️ GPU
- GPU: **NVIDIA RTX 3090 / A100 (recommended)**
- VRAM: **≥ 24GB recommended**


## 📈 Evaluation Metrics

- Accuracy (ACC)
- Weighted F1 (WF1)
- Macro F1 (MF1) ← **main metric**

---
data/
 ├── train.csv
 ├── valid.csv
 └── test.csv
```

### 3. Run training

```bash
python train.py
```


## 📬 Contact

For questions or collaboration, please open an issue or contact the author.

```
d202581751@hust.edu.com
```
