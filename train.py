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

# Clear any leftover memory from the previous crash
torch.cuda.empty_cache()

# ===============================
# 🔥 LOAD DATASET
# ===============================
dataset = load_dataset("json", data_files="students_dataset.json")["train"]
dataset = dataset.shuffle(seed=42)

split = dataset.train_test_split(test_size=0.1)
train_data = split["train"]
val_data = split["test"]

# ===============================
# 🔥 MODEL
# ===============================
model_name = "google/flan-t5-small"
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name).to(device)

# ===============================
# 🔥 PREPROCESS (Optimized Length)
# ===============================
def preprocess(example):
    # Reduced slightly from 256 to 192 to save VRAM while keeping multi-table support
    inputs = tokenizer(example["input"], max_length=192, padding="max_length", truncation=True)
    outputs = tokenizer(example["output"], max_length=192, padding="max_length", truncation=True)

    return {
        "input_ids": inputs["input_ids"],
        "attention_mask": inputs["attention_mask"],
        "labels": outputs["input_ids"]
    }

train_data = train_data.map(preprocess, remove_columns=train_data.column_names)
val_data = val_data.map(preprocess, remove_columns=val_data.column_names)

# ===============================
# 🔥 COLLATE & DATALOADERS
# ===============================
def collate_fn(batch):
    input_ids = torch.tensor([x["input_ids"] for x in batch])
    attention_mask = torch.tensor([x["attention_mask"] for x in batch])
    labels = torch.tensor([x["labels"] for x in batch])
    labels[labels == tokenizer.pad_token_id] = -100
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

# Set batch_size to 1 to fit in 4GB VRAM
train_loader = DataLoader(train_data, batch_size=1, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_data, batch_size=1, shuffle=False, collate_fn=collate_fn)

# ===============================
# 🔥 TRAIN SETUP
# ===============================
EPOCHS = 8
GRADIENT_ACCUMULATION_STEPS = 4  # Simulates a batch size of 4
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=50,
    num_training_steps=(len(train_loader) // GRADIENT_ACCUMULATION_STEPS) * EPOCHS
)

# ===============================
# 🚀 TRAIN LOOP (With Memory Optimization)
# ===============================
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    optimizer.zero_grad() # Initialize gradients once at start of epoch

    for i, batch in enumerate(train_loader):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        # Forward pass
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        
        # Scale loss to account for accumulation steps
        loss = outputs.loss / GRADIENT_ACCUMULATION_STEPS
        loss.backward()

        # Only update weights every N steps[cite: 7]
        if (i + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad() # Reset gradients[cite: 7]
            
        total_loss += outputs.loss.item()

    print(f"Epoch {epoch+1} | Average Loss: {total_loss / len(train_loader)}")
    
    # Clear cache after each epoch to prevent fragmentation[cite: 7]
    torch.cuda.empty_cache()

# ===============================
# 💾 SAVE
# ===============================
model.save_pretrained("./fine_tune_model")
tokenizer.save_pretrained("./fine_tune_model")
print("✅ Training Complete! Memory limits respected.")