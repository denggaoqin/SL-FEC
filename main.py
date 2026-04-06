import os
import math
import random
import numpy as np
import torch
import pandas as pd

from torch import nn
from torch.utils.data import Dataset, DataLoader

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup
)

from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm

# --------------------------
# Environment & AMP
# --------------------------
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
from torch.cuda.amp import autocast, GradScaler
scaler = GradScaler()

# --------------------------
# Configuration
# --------------------------
MODEL = "your/models--roberta-base"
train_path = "your_train_path_here.csv"
valid_path = "your_valid_path_here.csv"  
test_path  = "your_test_path_here.csv"  

seed = 42

# Stage 1
binary_epochs = 1
lr_binary = 2e-5

# Stage 2
multi_epochs = 6
lr_multi = 2e-5
weight_decay = 0.01

batch_size = 64
grad_accum_steps = 1      
max_len = 128             
warmup_ratio = 0.05      

freeze_top_k = 4         
freeze_epochs = 2         

# Long-tail loss 
use_focal = True
focal_gamma = 0.5
focal_weight = 0.05       

# Early stopping 
use_early_stopping = True
patience = 3
save_best_path = "best_ed_multiclass.pt"

num_classes = 32

# --------------------------
# Set random seed
# --------------------------
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

set_seed(seed)

# --------------------------
# Load data
# --------------------------
df_train = pd.read_csv(train_path)
df_valid = pd.read_csv(valid_path)
df_test  = pd.read_csv(test_path)

# --------------------------
# Label -> binary
# --------------------------
positive_labels = {1, 3, 5, 6, 9, 12, 13, 15, 16, 20, 23, 27, 28, 29, 30, 31}

def get_binary_label(x):
    return 1 if int(x) in positive_labels else 0

df_train["binary"] = df_train["label"].apply(get_binary_label)
df_valid["binary"] = df_valid["label"].apply(get_binary_label)
df_test["binary"]  = df_test["label"].apply(get_binary_label)

# --------------------------
# Dataset
# --------------------------
class EmotionDataset(Dataset):
    def __init__(self, df, tokenizer, max_len=128):
        self.texts = df["text"].tolist()
        self.labels = df["label"].astype(int).tolist()
        self.binary_labels = df["binary"].astype(int).tolist()
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            str(self.texts[idx]),
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt"
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "label": torch.tensor(self.labels[idx], dtype=torch.long),
            "binary": torch.tensor(self.binary_labels[idx], dtype=torch.long),
        }

# --------------------------
# Two-stage model
# --------------------------
class TwoStageEmotionModel(nn.Module):
    def __init__(self, model_path, num_labels=32):
        super().__init__()
        self.binary_model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=2)
        self.multi_model  = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=num_labels)

    def forward(self, input_ids, attention_mask, labels=None, stage="binary"):
        if stage == "binary":
            return self.binary_model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        else:
            return self.multi_model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

# --------------------------
# Focal Loss
# --------------------------
class FocalLoss(nn.Module):
    def __init__(self, gamma=1.5, reduction="mean"):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction
        self.ce_none = nn.CrossEntropyLoss(reduction="none")

    def forward(self, logits, targets):
        ce = self.ce_none(logits, targets)
        pt = torch.exp(-ce)
        focal = (1 - pt) ** self.gamma * ce
        if self.reduction == "mean":
            return focal.mean()
        elif self.reduction == "sum":
            return focal.sum()
        return focal

# --------------------------
# class-balanced
# --------------------------
def compute_class_weights_log(labels, num_classes, eps=1e-6):
    counts = [0] * num_classes
    for l in labels:
        counts[int(l)] += 1
    freqs = torch.tensor(counts, dtype=torch.float) + eps
    weights = 1.0 / torch.log(freqs + 1.0)
    weights = weights / weights.sum() * num_classes
    return weights

# --------------------------
# Freeze / unfreeze 
# --------------------------
def freeze_encoder_except_topk(base_model, top_k=2, freeze=True):
    if not hasattr(base_model, "encoder") or not hasattr(base_model.encoder, "layer"):
        # fallback: freeze/unfreeze all parameters
        for p in base_model.parameters():
            p.requires_grad = not freeze
        return

    layers = list(base_model.encoder.layer)
    n = len(layers)
    for i, layer in enumerate(layers):
        if freeze:
            req = (i >= n - top_k)
        else:
            req = True
        for p in layer.parameters():
            p.requires_grad = req

def set_classifier_trainable(hf_model, trainable=True):
    # HuggingFace classification head is usually .classifier
    if hasattr(hf_model, "classifier"):
        for p in hf_model.classifier.parameters():
            p.requires_grad = trainable

# --------------------------
# Transfer encoder weights
# --------------------------
def transfer_encoder_weights(model):
    try:
        src = model.binary_model.base_model
        dst = model.multi_model.base_model
        dst.load_state_dict(src.state_dict(), strict=False)
        print(">> transferred encoder weights from binary.base_model -> multi.base_model")
    except Exception as e:
        print("!! transfer skipped:", e)

# --------------------------
# Evaluation
# --------------------------
@torch.no_grad()
def evaluate_multiclass(model, dataloader, device):
    model.eval()
    all_labels, all_preds = [], []
    for batch in dataloader:
        ids = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        out = model(input_ids=ids, attention_mask=mask, labels=None, stage="multi")
        preds = out.logits.argmax(dim=-1)

        all_labels.extend(labels.cpu().tolist())
        all_preds.extend(preds.cpu().tolist())

    acc = accuracy_score(all_labels, all_preds)
    wf1 = f1_score(all_labels, all_preds, average="weighted")
    mf1 = f1_score(all_labels, all_preds, average="macro")
    return acc, wf1, mf1

@torch.no_grad()
def evaluate_binary(model, dataloader, device):
    model.eval()
    all_labels, all_preds = [], []
    for batch in dataloader:
        ids = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        labels = batch["binary"].to(device)

        out = model(input_ids=ids, attention_mask=mask, labels=None, stage="binary")
        preds = out.logits.argmax(dim=-1)

        all_labels.extend(labels.cpu().tolist())
        all_preds.extend(preds.cpu().tolist())

    return accuracy_score(all_labels, all_preds)

# --------------------------
# Train one epoch
# --------------------------
def train_one_epoch(
    model,
    dataloader,
    optimizer,
    scheduler,
    device,
    stage="binary",
    grad_accum_steps=1,
    ce_loss_fn=None,
    focal_loss_fn=None,
    focal_w=0.15,
    use_focal=False
):
    model.train()
    optimizer.zero_grad(set_to_none=True)

    total_loss = 0.0
    step_count = 0
    opt_steps = 0

    pbar = tqdm(dataloader, desc=f"Train({stage})", ncols=120)
    for step_idx, batch in enumerate(pbar, start=1):
        ids = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        labels = batch["binary" if stage == "binary" else "label"].to(device)

        with autocast():
            out = model(input_ids=ids, attention_mask=mask, labels=None, stage=stage)
            logits = out.logits

            if stage == "binary":
                loss = nn.CrossEntropyLoss()(logits, labels)
            else:
                ce = ce_loss_fn(logits, labels)
                if use_focal and focal_loss_fn is not None:
                    focal = focal_loss_fn(logits, labels)
                    loss = ce + focal_w * focal
                else:
                    loss = ce

            loss = loss / grad_accum_steps

        scaler.scale(loss).backward()

        total_loss += float(loss.detach().cpu())
        step_count += 1

        if step_idx % grad_accum_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            opt_steps += 1
            if scheduler is not None:
                scheduler.step()

        pbar.set_postfix({"loss": total_loss / max(1, step_count), "opt_steps": opt_steps})

    # Handle remaining steps if not divisible by grad_accum_steps
    if (len(dataloader) % grad_accum_steps) != 0:
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        if scheduler is not None:
            scheduler.step()

# --------------------------
# Main pipeline
# --------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device)

tokenizer = AutoTokenizer.from_pretrained(MODEL)

train_ds = EmotionDataset(df_train, tokenizer, max_len=max_len)
valid_ds = EmotionDataset(df_valid, tokenizer, max_len=max_len)
test_ds  = EmotionDataset(df_test,  tokenizer, max_len=max_len)

train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=2, pin_memory=True)
valid_loader = DataLoader(valid_ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

model = TwoStageEmotionModel(MODEL, num_labels=num_classes).to(device)

# --------------------------
# Stage 1: Binary
# --------------------------
print("\n=== Stage 1: Binary pretraining ===")
bin_optimizer = torch.optim.AdamW(
    model.binary_model.parameters(),
    lr=lr_binary,
    weight_decay=weight_decay
)

bin_total_opt_steps = math.ceil(len(train_loader) / grad_accum_steps) * max(1, binary_epochs)
bin_warmup_steps = int(warmup_ratio * bin_total_opt_steps)

bin_scheduler = get_linear_schedule_with_warmup(
    bin_optimizer,
    num_warmup_steps=bin_warmup_steps,
    num_training_steps=bin_total_opt_steps
)

for e in range(binary_epochs):
    train_one_epoch(
        model, train_loader,
        bin_optimizer, bin_scheduler,
        device,
        stage="binary",
        grad_accum_steps=grad_accum_steps
    )
    acc_bin = evaluate_binary(model, valid_loader, device)
    print(f"[Binary Epoch {e+1}] Val ACC={acc_bin:.4f}")

# --------------------------
# Transfer encoder weights -> multi
# --------------------------
transfer_encoder_weights(model)

# --------------------------
# Stage 2: Multi-class
# --------------------------
print("\n=== Stage 2: Multi-class fine-tuning ===")

class_weights = compute_class_weights_log(df_train["label"].tolist(), num_classes).to(device)
ce_loss_fn = nn.CrossEntropyLoss(weight=class_weights)
focal_loss_fn = FocalLoss(gamma=focal_gamma) if use_focal else None

# Freeze strategy: freeze all encoder layers except top-k initially
base = model.multi_model.base_model
freeze_encoder_except_topk(base, top_k=freeze_top_k, freeze=True)
set_classifier_trainable(model.multi_model, trainable=True)
print(f">> Frozen encoder layers except top {freeze_top_k} (initial), classifier trainable.")

multi_optimizer = torch.optim.AdamW(
    model.multi_model.parameters(),
    lr=lr_multi,
    weight_decay=weight_decay
)

multi_total_opt_steps = math.ceil(len(train_loader) / grad_accum_steps) * max(1, multi_epochs)
multi_warmup_steps = int(warmup_ratio * multi_total_opt_steps)

multi_scheduler = get_linear_schedule_with_warmup(
    multi_optimizer,
    num_warmup_steps=multi_warmup_steps,
    num_training_steps=multi_total_opt_steps
)

best_mf1 = -1.0
bad_epochs = 0

for epoch in range(multi_epochs):
    # Unfreeze all encoder layers at specified epoch
    if epoch == freeze_epochs:
        freeze_encoder_except_topk(base, top_k=freeze_top_k, freeze=False)
        print(f">> Unfroze ALL encoder layers at epoch {epoch+1} (keep same optimizer).")

    train_one_epoch(
        model, train_loader,
        multi_optimizer, multi_scheduler,
        device,
        stage="multi",
        grad_accum_steps=grad_accum_steps,
        ce_loss_fn=ce_loss_fn,
        focal_loss_fn=focal_loss_fn,
        focal_w=focal_weight,
        use_focal=use_focal
    )

    val_acc, val_wf1, val_mf1 = evaluate_multiclass(model, valid_loader, device)
    print(f"[Multi Epoch {epoch+1}] Val ACC={val_acc:.4f} WF1={val_wf1:.4f} MF1={val_mf1:.4f}")

    # Early stopping / save best model based on macro-F1
    if val_mf1 > best_mf1 + 1e-6:
        best_mf1 = val_mf1
        bad_epochs = 0
        torch.save(model.multi_model.state_dict(), save_best_path)
        print(f">> Saved best checkpoint: {save_best_path} (best MF1={best_mf1:.4f})")
    else:
        bad_epochs += 1

    if use_early_stopping and bad_epochs >= patience:
        print(f">> Early stopping: MF1 no improvement for {patience} epochs.")
        break

# --------------------------
# Final Test 
# --------------------------
print("\n=== Final Test (load best checkpoint) ===")
if os.path.exists(save_best_path):
    model.multi_model.load_state_dict(torch.load(save_best_path, map_location=device))
    print(f">> Loaded best: {save_best_path}")
else:
    print(">> Best checkpoint not found, using current weights.")

test_acc, test_wf1, test_mf1 = evaluate_multiclass(model, test_loader, device)
print("\n====== Final Test Results ======")
print(f"Test ACC         : {test_acc:.4f}")
print(f"Test Weighted-F1 : {test_wf1:.4f}")
print(f"Test Macro-F1    : {test_mf1:.4f}")