import json
import os
import random

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(os.environ["LLM_TEE_DATA_ROOT"], 'models--prajjwal1--bert-tiny', 'snapshots', '6f75de8b60a9f8a2fdf7b69cbd86d9e64bcb3837')
TRAIN_DATA_FILE = os.path.join(CURRENT_DIR, "asb_train.json")
GENERATED_DIR = os.environ.get("ASB_TRAINING_OUTPUT_DIR", os.path.join(CURRENT_DIR, "generated"))
SAVE_WEIGHT_PATH = os.path.join(GENERATED_DIR, "checker_tinybert_17b.pth")
SAVE_METADATA_PATH = os.path.join(GENERATED_DIR, "checker_tinybert_17b.meta.json")

BERT_LR = 5e-5
HEAD_LR = 2e-4
WEIGHT_DECAY = 0.05
EPOCHS = 30
BATCH_SIZE = 16
MAX_LEN = 128
SEED = 42
PATIENCE = 5


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


seed_everything(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class FocalLoss(torch.nn.Module):
    def __init__(self, positive_alpha, gamma=2.0):
        super().__init__()
        self.positive_alpha = float(positive_alpha)
        self.gamma = float(gamma)

    def forward(self, logits, targets):
        ce_loss = F.cross_entropy(logits, targets, reduction="none")
        pt = torch.exp(-ce_loss)
        alpha_t = torch.where(
            targets == 1,
            torch.full_like(pt, self.positive_alpha),
            torch.full_like(pt, 1.0 - self.positive_alpha),
        )
        return (alpha_t * (1.0 - pt).pow(self.gamma) * ce_loss).mean()


class ASBShieldDataset(Dataset):
    def __init__(self, rows, tokenizer):
        self.rows = rows
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        item = self.rows[index]
        intent = str(item["intent"]).lower().strip(" .,!?")
        action = str(item["action"]).lower().replace("_", " ").strip(" .,!?")
        encoding = self.tokenizer(
            f"Intent: {intent}",
            f"Execution Payload: {action}",
            max_length=MAX_LEN,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "labels": torch.tensor(int(item["hard_label"]), dtype=torch.long),
        }


def grouped_train_validation_split(rows):
    """Hold out one of the four training tasks per agent for validation."""
    intents_by_agent = {}
    for item in rows:
        agent = str(item.get("agent_name", ""))
        raw_intent = str(item.get("raw_intent", item.get("intent", "")))
        intents_by_agent.setdefault(agent, set()).add(raw_intent)

    validation_keys = set()
    split_manifest = {}
    for agent in sorted(intents_by_agent):
        intents = sorted(intents_by_agent[agent])
        random.Random(f"{SEED}:{agent}").shuffle(intents)
        if len(intents) < 2:
            raise ValueError(f"Agent {agent} has fewer than two training tasks")
        validation_intents = intents[-1:]
        training_intents = intents[:-1]
        validation_keys.update((agent, intent) for intent in validation_intents)
        split_manifest[agent] = {
            "train_intents": training_intents,
            "validation_intents": validation_intents,
        }

    train_rows, validation_rows = [], []
    for item in rows:
        key = (
            str(item.get("agent_name", "")),
            str(item.get("raw_intent", item.get("intent", ""))),
        )
        (validation_rows if key in validation_keys else train_rows).append(item)

    return train_rows, validation_rows, split_manifest


def confusion_from_scores(labels, safe_scores, threshold):
    tp = fp = tn = fn = 0
    for label, score in zip(labels, safe_scores):
        prediction = 1 if score >= threshold else 0
        if label == 1 and prediction == 1:
            tp += 1
        elif label == 0 and prediction == 0:
            tn += 1
        elif label == 0 and prediction == 1:
            fp += 1
        else:
            fn += 1
    safe_recall = tp / (tp + fn) if tp + fn else 0.0
    attack_recall = tn / (tn + fp) if tn + fp else 0.0
    balanced_accuracy = (safe_recall + attack_recall) / 2.0
    return {
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "safe_recall": safe_recall,
        "attack_recall": attack_recall,
        "balanced_accuracy": balanced_accuracy,
    }


def collect_scores(model, loader):
    labels, safe_scores = [], []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            batch_labels = batch["labels"].to(device)
            logits = model(ids, attention_mask=mask).logits
            scores = F.softmax(logits, dim=1)[:, 1]
            labels.extend(batch_labels.cpu().tolist())
            safe_scores.extend(scores.cpu().tolist())
    return labels, safe_scores


def choose_threshold(labels, safe_scores):
    candidates = [index / 100.0 for index in range(5, 96)]
    ranked = []
    for threshold in candidates:
        metrics = confusion_from_scores(labels, safe_scores, threshold)
        ranked.append((metrics["balanced_accuracy"], -abs(threshold - 0.5), threshold, metrics))
    _, _, threshold, metrics = max(ranked)
    return threshold, metrics


def train():
    os.makedirs(GENERATED_DIR, exist_ok=True)
    with open(TRAIN_DATA_FILE, "r", encoding="utf-8") as handle:
        full_data = json.load(handle)
    if not full_data:
        raise ValueError("Training data is empty")

    train_rows, validation_rows, split_manifest = grouped_train_validation_split(full_data)
    train_intents = {item.get("raw_intent", item["intent"]) for item in train_rows}
    validation_intents = {item.get("raw_intent", item["intent"]) for item in validation_rows}
    if train_intents & validation_intents:
        raise ValueError("User-task leakage between model-train and validation sets")

    count_safe = sum(int(item["hard_label"]) == 1 for item in train_rows)
    count_attack = len(train_rows) - count_safe
    if not count_safe or not count_attack:
        raise ValueError("Both safe and attack classes are required")
    positive_alpha = count_attack / (count_safe + count_attack)

    print(f"Train rows: {len(train_rows)}; validation rows: {len(validation_rows)}")
    print(f"Train attack(0)/safe(1): {count_attack}/{count_safe}")
    print(f"Class-balanced focal positive alpha: {positive_alpha:.6f}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_PATH, num_labels=2
    ).to(device)
    train_loader = DataLoader(
        ASBShieldDataset(train_rows, tokenizer),
        batch_size=BATCH_SIZE,
        shuffle=True,
    )
    validation_loader = DataLoader(
        ASBShieldDataset(validation_rows, tokenizer),
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    loss_fn = FocalLoss(positive_alpha=positive_alpha, gamma=2.0).to(device)
    optimizer = AdamW([
        {
            "params": [p for n, p in model.named_parameters() if "bert" in n],
            "lr": BERT_LR,
            "weight_decay": WEIGHT_DECAY,
        },
        {
            "params": [p for n, p in model.named_parameters() if "classifier" in n],
            "lr": HEAD_LR,
            "weight_decay": 0.0,
        },
    ])

    best_metric = -1.0
    patience_counter = 0
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch}"):
            optimizer.zero_grad()
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            logits = model(ids, attention_mask=mask).logits
            loss = loss_fn(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        labels, safe_scores = collect_scores(model, validation_loader)
        threshold, metrics = choose_threshold(labels, safe_scores)
        score = metrics["balanced_accuracy"]
        print(
            f"Epoch {epoch}: loss={total_loss / max(len(train_loader), 1):.6f}; "
            f"val balanced_acc={score:.4f}; threshold={threshold:.2f}; "
            f"safe_recall={metrics['safe_recall']:.4f}; "
            f"attack_recall={metrics['attack_recall']:.4f}"
        )

        if score > best_metric:
            best_metric = score
            patience_counter = 0
            torch.save(model.state_dict(), SAVE_WEIGHT_PATH)
            with open(SAVE_METADATA_PATH, "w", encoding="utf-8") as handle:
                json.dump({
                    "model_path": MODEL_PATH,
                    "max_len": MAX_LEN,
                    "safe_class_id": 1,
                    "safe_threshold": threshold,
                    "validation_metrics": metrics,
                    "validation_split": split_manifest,
                    "seed": SEED,
                }, handle, ensure_ascii=False, indent=2)
            print(f"Saved new best checkpoint: {SAVE_WEIGHT_PATH}")
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"Early stopping; best balanced accuracy={best_metric:.4f}")
                break


if __name__ == "__main__":
    train()
