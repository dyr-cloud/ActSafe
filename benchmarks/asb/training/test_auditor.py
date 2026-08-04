import json
import os

import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
TEST_FILE = os.path.join(CURRENT_DIR, "asb_auditor_test.json")
MODEL_PATH = os.path.join(os.environ["LLM_TEE_DATA_ROOT"], 'models--prajjwal1--bert-tiny', 'snapshots', '6f75de8b60a9f8a2fdf7b69cbd86d9e64bcb3837')
WEIGHTS_PATH = os.environ.get("ASB_AUDITOR_WEIGHTS", os.path.join(CURRENT_DIR, "generated", "checker_tinybert_17b.pth"))
METADATA_PATH = os.environ.get("ASB_AUDITOR_METADATA", os.path.join(CURRENT_DIR, "generated", "checker_tinybert_17b.meta.json"))
OUTPUT_FILE = os.environ.get("ASB_AUDITOR_TEST_OUTPUT", os.path.join(CURRENT_DIR, "generated", "auditor_test_result.json"))

with open(METADATA_PATH, "r", encoding="utf-8") as handle:
    metadata = json.load(handle)

MAX_LEN = int(metadata["max_len"])
SAFE_CLASS_ID = int(metadata["safe_class_id"])
THRESHOLD = float(metadata["safe_threshold"])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_PATH, num_labels=2
).to(device)
model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=device))
model.eval()

with open(TEST_FILE, "r", encoding="utf-8") as handle:
    test_data = json.load(handle)

tp = fp = tn = fn = 0
false_safe = []
false_block = []

for item in tqdm(test_data, desc="Testing ASB Auditor"):
    intent = str(item["intent"]).lower().strip(" .,!?")
    action = str(item["action"]).lower().replace("_", " ").strip(" .,!?")
    true_label = int(item["hard_label"])

    tokens = tokenizer(
        f"Intent: {intent}",
        f"Execution Payload: {action}",
        max_length=MAX_LEN,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        logits = model(
            tokens["input_ids"], attention_mask=tokens["attention_mask"]
        ).logits
        safe_score = F.softmax(logits, dim=1)[0, SAFE_CLASS_ID].item()

    prediction = 1 if safe_score >= THRESHOLD else 0
    if true_label == 1 and prediction == 1:
        tp += 1
    elif true_label == 0 and prediction == 0:
        tn += 1
    elif true_label == 0 and prediction == 1:
        fp += 1
        false_safe.append({
            "source_case_id": item.get("source_case_id"),
            "intent": item.get("raw_intent", item["intent"]),
            "action": item["action"],
            "safe_score": safe_score,
        })
    else:
        fn += 1
        false_block.append({
            "source_case_id": item.get("source_case_id"),
            "intent": item.get("raw_intent", item["intent"]),
            "action": item["action"],
            "safe_score": safe_score,
        })

total = len(test_data)
accuracy = (tp + tn) / total if total else 0.0
safe_recall = tp / (tp + fn) if tp + fn else 0.0
attack_recall = tn / (tn + fp) if tn + fp else 0.0
balanced_accuracy = (safe_recall + attack_recall) / 2.0

# Defense terminology: an attack accepted as safe is a security false
# negative; a benign operation blocked is a defense false positive.
security_fnr = fp / (tn + fp) if tn + fp else 0.0
defense_fpr = fn / (tp + fn) if tp + fn else 0.0

result = {
    "total": total,
    "threshold": THRESHOLD,
    "confusion_safe_as_positive": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
    "accuracy": accuracy,
    "balanced_accuracy": balanced_accuracy,
    "safe_recall": safe_recall,
    "attack_recall": attack_recall,
    "security_fnr_attack_accepted": security_fnr,
    "defense_fpr_benign_blocked": defense_fpr,
    "false_safe_cases": false_safe,
    "false_block_cases": false_block,
}

os.makedirs(os.path.dirname(os.path.abspath(OUTPUT_FILE)), exist_ok=True)
with open(OUTPUT_FILE, "w", encoding="utf-8") as handle:
    json.dump(result, handle, ensure_ascii=False, indent=2)

print("\n" + "=" * 72)
print("[ASB TinyBERT Auditor Test]")
print(f"Samples: {total}; threshold={THRESHOLD:.2f}")
print(f"Accuracy:          {accuracy * 100:.2f}%")
print(f"Balanced accuracy: {balanced_accuracy * 100:.2f}%")
print(f"Safe recall:       {safe_recall * 100:.2f}%")
print(f"Attack recall:     {attack_recall * 100:.2f}%")
print(f"Security FNR (attack accepted): {security_fnr * 100:.2f}% ({fp}/{tn + fp})")
print(f"Defense FPR (benign blocked):   {defense_fpr * 100:.2f}% ({fn}/{tp + fn})")
print(f"Result: {os.path.abspath(OUTPUT_FILE)}")
print("=" * 72)
