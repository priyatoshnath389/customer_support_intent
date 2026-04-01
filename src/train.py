import os
import joblib
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification, get_linear_schedule_with_warmup
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# ─────────────────────────────────────────────
# CONFIG  (change these if your paths differ)
# ─────────────────────────────────────────────
TRAIN_CSV   = "data/processed/train.csv"
TEST_CSV    = "data/processed/test.csv"
MODEL_DIR   = "models"
MAX_LEN     = 32       # based on Day 2 EDA — customer support texts are short
BATCH_SIZE  = 32
EPOCHS      = 5
LR          = 2e-5
SEED        = 42

os.makedirs(MODEL_DIR, exist_ok=True)
torch.manual_seed(SEED)
np.random.seed(SEED)

# ─────────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────────
print("Loading data...")
train_df = pd.read_csv(TRAIN_CSV)
test_df  = pd.read_csv(TEST_CSV)

# Quick sanity check
assert "text" in train_df.columns and "intent" in train_df.columns, \
    "CSV must have 'text' and 'intent' columns"

print(f"  Train samples : {len(train_df)}")
print(f"  Test  samples : {len(test_df)}")
print(f"  Intents found : {train_df['intent'].nunique()}")

# ─────────────────────────────────────────────
# 2. LABEL ENCODING
# ─────────────────────────────────────────────
print("\nEncoding labels...")
label_encoder = LabelEncoder()
train_df["label"] = label_encoder.fit_transform(train_df["intent"])
test_df["label"]  = label_encoder.transform(test_df["intent"])

joblib.dump(label_encoder, os.path.join(MODEL_DIR, "label_encoder.pkl"))
print("  Saved → models/label_encoder.pkl")

NUM_LABELS = len(label_encoder.classes_)
print(f"\n  Label map ({NUM_LABELS} classes):")
for i, cls in enumerate(label_encoder.classes_):
    print(f"    {i:2d} → {cls}")

# ─────────────────────────────────────────────
# 3. TOKENIZER
# ─────────────────────────────────────────────
print("\nLoading tokenizer...")
tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
joblib.dump(tokenizer, os.path.join(MODEL_DIR, "tokenizer.pkl"))

# ─────────────────────────────────────────────
# 4. PYTORCH DATASET
# ─────────────────────────────────────────────
class IntentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=MAX_LEN):
        self.texts     = list(texts)
        self.labels    = list(labels)
        self.tokenizer = tokenizer
        self.max_len   = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            str(self.texts[idx]),
            add_special_tokens=True,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        return {
            "input_ids":      encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "labels":         torch.tensor(self.labels[idx], dtype=torch.long),
        }

train_dataset = IntentDataset(train_df["text"], train_df["label"], tokenizer)
test_dataset  = IntentDataset(test_df["text"],  test_df["label"],  tokenizer)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0, pin_memory=True)
test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)

print(f"\n  Train batches : {len(train_loader)}")
print(f"  Test  batches : {len(test_loader)}")

# ─────────────────────────────────────────────
# 5. MODEL
# ─────────────────────────────────────────────
print("\nLoading DistilBERT...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"  Running on : {device}")

model = DistilBertForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
    num_labels=NUM_LABELS,
)
model.to(device)

total_params     = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"  Total params    : {total_params:,}")
print(f"  Trainable params: {trainable_params:,}")

# ─────────────────────────────────────────────
# 6. OPTIMIZER + SCHEDULER
# ─────────────────────────────────────────────
optimizer = AdamW(model.parameters(), lr=LR, weight_decay=0.01)

total_steps = len(train_loader) * EPOCHS
warmup_steps = int(0.1 * total_steps)  # 10% warmup — stabilises early training

scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=warmup_steps,
    num_training_steps=total_steps,
)

# ─────────────────────────────────────────────
# 7. TRAINING + VALIDATION LOOP
# ─────────────────────────────────────────────
def run_epoch(model, loader, optimizer, scheduler, device, is_train=True):
    model.train() if is_train else model.eval()

    total_loss = 0
    all_preds, all_labels = [], []

    context = torch.enable_grad() if is_train else torch.no_grad()
    with context:
        for batch in tqdm(loader, desc="Train" if is_train else "Eval ", leave=False):
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels         = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss    = outputs.loss
            logits  = outputs.logits

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()

            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(loader)
    acc      = accuracy_score(all_labels, all_preds)
    f1       = f1_score(all_labels, all_preds, average="weighted")
    return avg_loss, acc, f1, all_labels, all_preds


history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": [], "train_f1": [], "val_f1": []}
best_val_f1 = 0.0

print("\n" + "─" * 55)
print(f"{'Epoch':>6} | {'Train Loss':>10} | {'Val Loss':>8} | {'Train F1':>8} | {'Val F1':>7}")
print("─" * 55)

for epoch in range(1, EPOCHS + 1):
    tr_loss, tr_acc, tr_f1, _, _           = run_epoch(model, train_loader, optimizer, scheduler, device, is_train=True)
    vl_loss, vl_acc, vl_f1, vl_lbls, vl_preds = run_epoch(model, test_loader,  optimizer, scheduler, device, is_train=False)

    history["train_loss"].append(tr_loss)
    history["val_loss"].append(vl_loss)
    history["train_acc"].append(tr_acc)
    history["val_acc"].append(vl_acc)
    history["train_f1"].append(tr_f1)
    history["val_f1"].append(vl_f1)

    print(f"{epoch:>6} | {tr_loss:>10.4f} | {vl_loss:>8.4f} | {tr_f1:>8.4f} | {vl_f1:>7.4f}")

    # Save best model
    if vl_f1 > best_val_f1:
        best_val_f1 = vl_f1
        model.save_pretrained(os.path.join(MODEL_DIR, "distilbert_intent"))
        tokenizer.save_pretrained(os.path.join(MODEL_DIR, "distilbert_intent"))
        print(f"         ✓ New best saved (F1={best_val_f1:.4f})")

print("─" * 55)
print(f"\nBest Validation F1 : {best_val_f1:.4f}")

# ─────────────────────────────────────────────
# 8. FINAL METRICS
# ─────────────────────────────────────────────
print(f"\nFinal Test Accuracy : {vl_acc:.4f}")
print(f"Final Test F1       : {vl_f1:.4f}")

# ─────────────────────────────────────────────
# 9. PLOTS — Loss curve + Confusion Matrix
# ─────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Loss curve
axes[0].plot(range(1, EPOCHS + 1), history["train_loss"], marker="o", label="Train loss")
axes[0].plot(range(1, EPOCHS + 1), history["val_loss"],   marker="o", label="Val loss")
axes[0].set_title("Training vs Validation Loss")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Loss")
axes[0].legend()
axes[0].grid(alpha=0.3)

# Confusion matrix
cm = confusion_matrix(vl_lbls, vl_preds)
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=label_encoder.classes_,
    yticklabels=label_encoder.classes_,
    ax=axes[1],
)
axes[1].set_title("Confusion Matrix (Test Set)")
axes[1].set_xlabel("Predicted")
axes[1].set_ylabel("Actual")
axes[1].tick_params(axis="x", rotation=45)
axes[1].tick_params(axis="y", rotation=0)

plt.tight_layout()
plt.savefig(os.path.join(MODEL_DIR, "training_results.png"), dpi=150, bbox_inches="tight")
plt.show()
print("\nPlot saved → models/training_results.png")
print("\nDone. All artifacts saved under models/")