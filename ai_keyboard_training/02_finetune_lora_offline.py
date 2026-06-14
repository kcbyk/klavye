# -*- coding: utf-8 -*-
"""
02_finetune_lora_offline.py
Offline LoRA fine-tuning for SmolLM2-135M — CPU-first, no HF API usage.
Save under: ai_keyboard_training/02_finetune_lora_offline.py
Run (Windows CMD): set TRANSFORMERS_OFFLINE=1 && set HF_DATASETS_OFFLINE=1 && python ai_keyboard_training\02_finetune_lora_offline.py
Run (PowerShell): $env:TRANSFORMERS_OFFLINE='1'; $env:HF_DATASETS_OFFLINE='1'; python .\ai_keyboard_training\02_finetune_lora_offline.py
"""
from pathlib import Path
import os, sys, json, warnings, io
warnings.filterwarnings("ignore")
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
# enforce offline behavior
os.environ["TRANSFORMERS_OFFLINE"] = os.environ.get("TRANSFORMERS_OFFLINE", "1")
os.environ["HF_DATASETS_OFFLINE"] = os.environ.get("HF_DATASETS_OFFLINE", "1")

# ensure utf-8 stdout on Windows
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

# --- imports ---
import subprocess

def install(pkg):
    subprocess.run([sys.executable, "-m", "pip", "install", pkg], check=False)

# (Optional) check essentials
required = ("peft","datasets","transformers","accelerate","torch")
for mod in required:
    try:
        __import__(mod)
    except ImportError:
        print(f"[WARNING] Python package '{mod}' missing. Install manually if needed (offline install required).")

import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, DataCollatorForLanguageModeling, Trainer
from peft import get_peft_model, LoraConfig, TaskType, PeftModel

# --- config ---
ROOT = Path(__file__).resolve().parent
MODEL_ID = "HuggingFaceTB/SmolLM2-135M-Instruct"  # metadata only
LOCAL_MODEL = ROOT / "outputs" / "smollm2_base"
OUTPUT_LORA = ROOT / "outputs" / "smollm2_lora"
OUTPUT_MERGED = ROOT / "outputs" / "smollm2_merged"
DATA_TRAIN = ROOT / "data" / "turkish_chat_train.jsonl"
DATA_EVAL  = ROOT / "data" / "turkish_chat_eval.jsonl"

# device: choose cuda if available
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_FP16 = False if DEVICE.type == "cpu" else True

# LoRA config
LORA_CONFIG = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=8,
    lora_alpha=32,
    target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
    lora_dropout=0.05,
    bias="none",
    inference_mode=False,
)

# training params (conservative for CPU)
MAX_SEQ_LEN = 128
BATCH_SIZE = 2
GRAD_ACC = 8
EPOCHS = 3
LR = 2e-4
WARMUP_STEPS = 50

# prepare paths
for p in (LOCAL_MODEL, OUTPUT_LORA, OUTPUT_MERGED):
    p.mkdir(parents=True, exist_ok=True)

# --- verify local model exists (NO INTERNET) ---
if not (LOCAL_MODEL / "config.json").exists():
    print("ERROR: Local model not found at:", LOCAL_MODEL)
    print("Place the pretrained SmolLM2 model files (config.json, model.* or model.safetensors, tokenizer files) into:")
    print("  ", LOCAL_MODEL)
    sys.exit(2)

# load tokenizer & model from local
print("Loading tokenizer & model from local folder:", LOCAL_MODEL)
tokenizer = AutoTokenizer.from_pretrained(str(LOCAL_MODEL), local_files_only=True)
model = AutoModelForCausalLM.from_pretrained(str(LOCAL_MODEL), torch_dtype=torch.float32, local_files_only=True)
# ensure padding token
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# attach LoRA
print("Applying LoRA adapter...")
model = get_peft_model(model, LORA_CONFIG)
model.print_trainable_parameters()

# --- dataset loader (expects jsonl where each line is a JSON object) ---
def load_jsonl(path: Path):
    if not path.exists():
        print(f"ERROR: dataset file missing: {path}")
        sys.exit(1)
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for ln in f:
            if ln.strip():
                rows.append(json.loads(ln))
    return rows

train_raw = load_jsonl(DATA_TRAIN)
eval_raw = load_jsonl(DATA_EVAL)

def extract_text(sample: dict) -> str:
    if sample.get("full_text") is not None:
        return sample["full_text"]
    if sample.get("text") is not None:
        return sample["text"]
    if sample.get("prompt") is not None and sample.get("completion") is not None:
        return sample["prompt"] + " " + sample["completion"]
    if sample.get("instruction") is not None and sample.get("output") is not None:
        inp = sample.get("input", "") or ""
        return f"{sample['instruction']} {inp} {sample['output']}"
    return ""

def tokenize(examples):
    texts = [extract_text(s) for s in examples["data"]]
    tokens = tokenizer(
        texts,
        truncation=True,
        max_length=MAX_SEQ_LEN,
        padding="max_length",
    )
    tokens["labels"] = tokens["input_ids"].copy()
    return tokens

train_ds = Dataset.from_dict({"data": train_raw})
eval_ds  = Dataset.from_dict({"data": eval_raw})

train_ds = train_ds.map(tokenize, batched=True, batch_size=32, remove_columns=["data"])
eval_ds  = eval_ds.map(tokenize, batched=True, batch_size=32, remove_columns=["data"])

print(f"Train samples: {len(train_ds)}, Eval: {len(eval_ds)}")

# --- training args (CPU-friendly) ---
training_args = TrainingArguments(
    output_dir=str(OUTPUT_LORA),
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACC,
    learning_rate=LR,
    warmup_steps=WARMUP_STEPS,
    weight_decay=0.01,
    fp16=USE_FP16,
    logging_steps=10,
    evaluation_strategy="no",
    save_strategy="no",
    dataloader_num_workers=0,
    report_to="none",
    remove_unused_columns=False,
    label_names=["labels"],
)

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=eval_ds,
    data_collator=data_collator,
)

print("Starting training on device:", DEVICE)
try:
    trainer.train()
    trainer.save_model(str(OUTPUT_LORA))
    print("LoRA adapter saved to:", OUTPUT_LORA)
except Exception as e:
    print("Training failed:", repr(e))
    print("If OOM or other issues, reduce BATCH_SIZE or GRAD_ACC, or train fewer steps.")
    sys.exit(3)

# Merge adapter into base to create a standalone model (merged)
print("Merging adapter into base model...")
base_model = AutoModelForCausalLM.from_pretrained(str(LOCAL_MODEL), torch_dtype=torch.float32, local_files_only=True)
lora_model = PeftModel.from_pretrained(base_model, str(OUTPUT_LORA), local_files_only=True)
merged = lora_model.merge_and_unload()
merged.eval()
merged.save_pretrained(str(OUTPUT_MERGED))
tokenizer.save_pretrained(str(OUTPUT_MERGED))
print("Merged model saved to:", OUTPUT_MERGED)
print("Done. Next: export to tflite (quantize) — see next script.")