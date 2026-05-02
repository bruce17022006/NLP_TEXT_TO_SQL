import torch
import os
from transformers import T5Tokenizer, T5ForConditionalGeneration, get_linear_schedule_with_warmup
from datasets import load_dataset
from torch.utils.data import DataLoader

# ===============================
# 🔥 SETUP
# ===============================
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

# ===============================
# 🔥 LOAD DATASET
# ===============================
dataset = load_dataset("json", data_files="students_dataset.json")["train"]
dataset = dataset.shuffle(seed=42)

split = dataset.train_test_split(test_size=0.1)
train_data = split["train"]
val_data = split["test"]

print("Train size:", len(train_data))
print("Validation size:", len(val_data))

# ===============================
# 🔥 MODEL
# ===============================
model_name = "google/flan-t5-small"
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name).to(device)

# ===============================
# 🔥 PREPROCESS
# ===============================
def preprocess(example):
    inputs = tokenizer(example["input"], max_length=64, padding="max_length", truncation=True)
    outputs = tokenizer(example["output"], max_length=64, padding="max_length", truncation=True)

    return {
        "input_ids": inputs["input_ids"],
        "attention_mask": inputs["attention_mask"],
        "labels": outputs["input_ids"]
    }

train_data = train_data.map(preprocess, remove_columns=train_data.column_names)
val_data = val_data.map(preprocess, remove_columns=val_data.column_names)

# ===============================
# 🔥 COLLATE
# ===============================
def collate_fn(batch):
    input_ids = torch.tensor([x["input_ids"] for x in batch])
    attention_mask = torch.tensor([x["attention_mask"] for x in batch])
    labels = torch.tensor([x["labels"] for x in batch])

    labels[labels == tokenizer.pad_token_id] = -100

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }

train_loader = DataLoader(train_data, batch_size=2, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_data, batch_size=2, shuffle=False, collate_fn=collate_fn)

# ===============================
# 🔥 TRAIN SETUP
# ===============================
EPOCHS = 6
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=10,
    num_training_steps=len(train_loader) * EPOCHS
)

# ===============================
# 🚀 TRAIN LOOP
# ===============================
for epoch in range(EPOCHS):
    print(f"\nEpoch {epoch+1}")
    model.train()
    total_loss = 0

    for batch in train_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss

        if torch.isnan(loss) or torch.isinf(loss):
            continue

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()
        scheduler.step()

        total_loss += loss.item()

    print("Train Loss:", total_loss / len(train_loader))

    # 🔍 Validation
    model.eval()
    val_loss = 0

    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            val_loss += outputs.loss.item()

    print("Validation Loss:", val_loss / len(val_loader))

# ===============================
# 💾 SAVE
# ===============================
model.save_pretrained("./fine_tune_model")
tokenizer.save_pretrained("./fine_tune_model")

print("✅ Training Complete!")