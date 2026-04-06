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
# 环境 & AMP
# --------------------------
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
from torch.cuda.amp import autocast, GradScaler
scaler = GradScaler()

# --------------------------
# 配置（按建议默认值）
# --------------------------
MODEL = "/data/qdg/TACO-main-copy/src/roberta-base/models--roberta-base/snapshots/e2da8e2f811d1448a5b465c236feacd80ffbac7b"
train_path = "/data/qdg/TACO-main-copy/data/go_emotion/train.csv"
valid_path = "/data/qdg/TACO-main-copy/data/go_emotion/valid.csv"
test_path  = "/data/qdg/TACO-main-copy/data/go_emotion/test.csv"

seed = 42

# Stage 1 (binary)
binary_epochs = 1
lr_binary = 2e-5

# Stage 2 (multi)
multi_epochs = 12
lr_multi = 1e-5                 # 建议：从 1e-5 起步
weight_decay = 0.01             # 建议：0.01 常用
warmup_ratio = 0.05             # 建议：0.03~0.06
max_len = 96                    # 建议：96 或 128
batch_size = 16
grad_accum_steps = 1            # 建议：有效 batch 变大（16*2=32）

# 冻结策略（不重建 optimizer，只改 requires_grad）
freeze_top_k = 4
freeze_epochs = 4               # 建议：略拉长

# 长尾损失（默认不开，按需要再开）
use_focal = False
focal_gamma = 1.5
focal_weight = 0.2              # 建议：0.1~0.3

# use_focal = True
# focal_gamma = 1.5
# focal_weight = 0.2              # 建议：0.1~0.3

# Early stopping（按 macro-F1）
use_early_stopping = True
patience = 3                    # 连续 3 个 epoch mf1 不涨就停
save_best_path = "best_multiclass.pt"

# --------------------------
# 固定随机种子
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
# 加载数据
# --------------------------
df_train = pd.read_csv(train_path)
df_valid = pd.read_csv(valid_path)
df_test  = pd.read_csv(test_path)

# --------------------------
# GoEmotion positive labels
# --------------------------
positive_labels = {0, 1, 4, 5, 7, 8, 13, 15, 17, 18, 20, 21, 22, 23, 26}

def get_binary_label(x):
    return 1 if x in positive_labels else 0

df_train["binary"] = df_train["label"].apply(get_binary_label)
df_valid["binary"] = df_valid["label"].apply(get_binary_label)
df_test["binary"]  = df_test["label"].apply(get_binary_label)

# --------------------------
# Dataset
# --------------------------
class EmotionDataset(Dataset):
    def __init__(self, df, tokenizer, max_len=96):
        self.texts = df["text"].tolist()
        self.labels = df["label"].tolist()
        self.binary_labels = df["binary"].tolist()
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoded = self.tokenizer(
            str(self.texts[idx]),
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt"
        )
        return {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "label": torch.tensor(self.labels[idx], dtype=torch.long),
            "binary": torch.tensor(self.binary_labels[idx], dtype=torch.long),
        }

# --------------------------
# Two-stage model
# --------------------------
class TwoStageEmotionModel(nn.Module):
    def __init__(self, model_path, num_labels=27):
        super().__init__()
        self.binary_model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=2)
        self.multi_model  = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=num_labels)

    def forward(self, input_ids, attention_mask, labels=None, stage="binary"):
        if stage == "binary":
            return self.binary_model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        else:
            return self.multi_model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

# --------------------------
# Focal Loss（可选）
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
# class weights（建议：log 版，更温和更稳）
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
# 冻结/解冻 helper
# --------------------------
def freeze_encoder_except_topk(base_model, top_k=2, freeze=True):
    """
    freeze=True: 只让 top_k 层 requires_grad=True，其余 False
    freeze=False: 所有 encoder 层 requires_grad=True（解冻）
    """
    if not hasattr(base_model, "encoder") or not hasattr(base_model.encoder, "layer"):
        for p in base_model.parameters():
            p.requires_grad = not freeze
        return

    layers = list(base_model.encoder.layer)
    n = len(layers)
    for i, layer in enumerate(layers):
        if freeze:
            should_require = (i >= n - top_k)
        else:
            should_require = True
        for p in layer.parameters():
            p.requires_grad = should_require

def set_head_trainable(model, stage="multi", trainable=True):
    if stage == "multi":
        head = model.multi_model.classifier
    else:
        head = model.binary_model.classifier
    for p in head.parameters():
        p.requires_grad = trainable

# --------------------------
# transfer encoder weights
# --------------------------
def transfer_encoder_weights(model):
    try:
        src = model.binary_model.base_model
        dst = model.multi_model.base_model
        dst.load_state_dict(src.state_dict(), strict=False)
        print(">> transferred encoder weights")
    except Exception as e:
        print(f">> transfer skipped: {e}")

# --------------------------
# eval
# --------------------------
@torch.no_grad()
def evaluate_multiclass(model, dataloader, device):
    model.eval()
    all_labels, all_preds = [], []
    for batch in dataloader:
        ids = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        out = model(input_ids=ids, attention_mask=mask, stage="multi")
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

        out = model(input_ids=ids, attention_mask=mask, stage="binary")
        preds = out.logits.argmax(dim=-1)

        all_labels.extend(labels.cpu().tolist())
        all_preds.extend(preds.cpu().tolist())
    return accuracy_score(all_labels, all_preds)

# --------------------------
# Train one epoch（支持梯度累积；binary 不重复 forward）
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
    focal_w=0.2,
    use_focal=False
):
    model.train()
    optimizer.zero_grad(set_to_none=True)

    total_loss = 0.0
    steps = 0
    opt_steps = 0

    pbar = tqdm(dataloader, desc=f"Train({stage})", ncols=120)
    for step_idx, batch in enumerate(pbar, start=1):
        ids  = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)

        labels = batch["binary" if stage == "binary" else "label"].to(device)

        with autocast():
            # 统一走 forward（binary 不重复 forward）
            out = model(input_ids=ids, attention_mask=mask, labels=None, stage=stage)
            logits = out.logits

            if stage == "binary":
                # 二分类用原生 CE（不加权，先稳）
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
        steps += 1

        if step_idx % grad_accum_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            opt_steps += 1
            if scheduler is not None:
                scheduler.step()

        avg_loss = total_loss / max(1, steps)
        pbar.set_postfix({"loss": avg_loss, "opt_steps": opt_steps})

    # 如果最后还剩余未 step 的梯度（当 len(dataloader) 不能整除 grad_accum_steps）
    if (len(dataloader) % grad_accum_steps) != 0:
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        if scheduler is not None:
            scheduler.step()

# --------------------------
# main
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

model = TwoStageEmotionModel(MODEL, num_labels=27).to(device)

# --------------------------
# Stage 1: Binary
# --------------------------
print("\n=== Stage 1: Binary pretraining ===")
bin_optimizer = torch.optim.AdamW(
    model.binary_model.parameters(),
    lr=lr_binary,
    weight_decay=weight_decay
)

# 计算优化器步数（考虑梯度累积）
bin_total_opt_steps = math.ceil(len(train_loader) / grad_accum_steps) * max(1, binary_epochs)
bin_warmup_steps = int(warmup_ratio * bin_total_opt_steps)
bin_scheduler = get_linear_schedule_with_warmup(bin_optimizer, bin_warmup_steps, bin_total_opt_steps)

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
# transfer weights
# --------------------------
transfer_encoder_weights(model)

# --------------------------
# Stage 2: Multi-class fine-tune
# --------------------------
print("\n=== Stage 2: Multi-class fine-tune ===")

num_classes = 27
class_weights = compute_class_weights_log(df_train["label"].tolist(), num_classes).to(device)
ce_loss_fn = nn.CrossEntropyLoss(weight=class_weights)

focal_loss_fn = FocalLoss(gamma=focal_gamma) if use_focal else None

# 冻结：只训练 top_k encoder 层 + head
base = model.multi_model.base_model
freeze_encoder_except_topk(base, top_k=freeze_top_k, freeze=True)
set_head_trainable(model, stage="multi", trainable=True)
print(f">> Frozen encoder layers except top {freeze_top_k} (head trainable)")

# 重要：不在解冻时重建 optimizer（保留 Adam 动量）
# 这里直接把 multi_model 全参数交给 optimizer；冻结层 requires_grad=False 时不会产生有效梯度
multi_optimizer = torch.optim.AdamW(
    model.multi_model.parameters(),
    lr=lr_multi,
    weight_decay=weight_decay
)

multi_total_opt_steps = math.ceil(len(train_loader) / grad_accum_steps) * max(1, multi_epochs)
multi_warmup_steps = int(warmup_ratio * multi_total_opt_steps)

multi_scheduler = get_linear_schedule_with_warmup(
    multi_optimizer,
    multi_warmup_steps,
    multi_total_opt_steps
)

best_mf1 = -1.0
bad_epochs = 0

for epoch in range(multi_epochs):
    # 到点解冻（不重建 optimizer，只改 requires_grad）
    if epoch == freeze_epochs:
        freeze_encoder_except_topk(base, top_k=freeze_top_k, freeze=False)
        print(f">> Unfroze ALL encoder layers at epoch {epoch+1}")

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

    # 保存 best（按 macro-F1）
    if val_mf1 > best_mf1 + 1e-6:
        best_mf1 = val_mf1
        bad_epochs = 0
        torch.save(model.multi_model.state_dict(), save_best_path)
        print(f">> Saved best multi_model to: {save_best_path} (best MF1={best_mf1:.4f})")
    else:
        bad_epochs += 1

    if use_early_stopping and bad_epochs >= patience:
        print(f">> Early stopping triggered (no MF1 improvement for {patience} epochs).")
        break

# --------------------------
# Final test（加载 best）
# --------------------------
print("\n=== Load best checkpoint and evaluate on test ===")
if os.path.exists(save_best_path):
    model.multi_model.load_state_dict(torch.load(save_best_path, map_location=device))
    print(f">> Loaded: {save_best_path}")
else:
    print(">> No best checkpoint found, evaluating current model.")

test_acc, test_wf1, test_mf1 = evaluate_multiclass(model, test_loader, device)

print("\n====== Final Test Results ======")
print(f"Test ACC         : {test_acc:.4f}")
print(f"Test Weighted-F1 : {test_wf1:.4f}")
print(f"Test Macro-F1    : {test_mf1:.4f}")
