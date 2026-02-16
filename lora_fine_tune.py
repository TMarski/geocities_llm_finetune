import os
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model

# --------------------------
# 1. Config
# --------------------------
BASE_MODEL = r"\home\models\Qwen3-4B-Instruct-2507"
DATA_PATH = r"\home\data\geocities.jsonl"
OUTPUT_DIR = "\home\models\fine-tunes\Qwen3-4B-Instruct_Geocities"

MAX_SEQ_LEN = 1024
BATCH_SIZE = 1           # per-device batch size
GRAD_ACCUM_STEPS = 16    # effective batch size = BATCH_SIZE * GRAD_ACCUM_STEPS
EPOCHS = 2
LR = 2e-4

# --------------------------
# 2. Dataset
# --------------------------
# JSONL with {"text": "..."} per line
dataset = load_dataset(
    "json",
    data_files={"train": DATA_PATH},
)["train"]

def filter_example(example):
    t = example.get("text", "")
    return isinstance(t, str) and len(t.strip()) > 20

dataset = dataset.filter(filter_example)

# --------------------------
# 3. Tokenizer
# --------------------------
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

def tokenize_fn(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=MAX_SEQ_LEN,
        padding=False,
    )

tokenized = dataset.map(
    tokenize_fn,
    batched=True,
    remove_columns=dataset.column_names,
)

# --------------------------
# 4. Model + LoRA
# --------------------------
# Load base model (full precision or bf16). The warning about torch_dtype being
# deprecated is just a warning; it's fine to use here.
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.bfloat16,
)

# LoRA config for Qwen/LLaMA-style blocks
lora_config = LoraConfig(
    r=64,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# --------------------------
# 5. Data collator
# --------------------------
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,           # causal LM, not masked LM
)

# --------------------------
# 6. TrainingArguments / Trainer
# --------------------------
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM_STEPS,
    learning_rate=LR,
    lr_scheduler_type="cosine",
    warmup_ratio=0.03,
    logging_steps=10,
    save_strategy="epoch",
    bf16=True,              # 4080 supports bf16; set False if you hit issues
    tf32=True,
    report_to="none",       # no wandb/tensorboard spam
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized,
    data_collator=data_collator,
)

# --------------------------
# 7. Train
# --------------------------
trainer.train()

# --------------------------
# 8. Save LoRA adapter + tokenizer
# --------------------------
trainer.model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print("Done. LoRA adapter saved to", OUTPUT_DIR)
