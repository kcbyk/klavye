# -*- coding: utf-8 -*-
"""
=============================================================
02_finetune_lora.py
Solenz AI Keyboard — SmolLM2-135M LoRA Fine-Tuning
=============================================================
SmolLM2-135M modelini Türkçe mesajlaşma verisiyle ince ayar
yapar. LoRA sayesinde sadece ~%1 parametre güncellenir.

Gereksinimler:
  pip install peft trl datasets accelerate bitsandbytes

CUDA (GPU) varsa ~30 dakika, CPU'da ~4-8 saat.

Çalıştırma: python 02_finetune_lora.py
Çıktı     : outputs/smollm2_lora/  (adapter)
            outputs/smollm2_merged/ (birleştirilmiş model)
"""

import os, sys, json, warnings
warnings.filterwarnings("ignore")
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

print("=" * 60)
print("  SmolLM2-135M LoRA Fine-Tuning")
print("=" * 60)

# ─── Bağımlılıklar ─────────────────────────────────────────────────
import subprocess

def install(pkg):
    subprocess.run([sys.executable, "-m", "pip", "install", pkg, "-q"], check=False)

required = {
    "peft": "peft>=0.10.0",
    "trl": "trl>=0.8.0",
    "datasets": "datasets>=2.18.0",
    "accelerate": "accelerate>=0.27.0",
}

for mod, pkg in required.items():
    try:
        __import__(mod)
        print(f"  [OK] {mod}")
    except ImportError:
        print(f"  [Kuruluyor] {pkg}...")
        install(pkg)

print()

import torch
from pathlib import Path
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    Trainer,
)
from peft import (
    get_peft_model,
    LoraConfig,
    TaskType,
    prepare_model_for_kbit_training,
)

# ─── Yapılandırma ──────────────────────────────────────────────────
MODEL_ID      = "HuggingFaceTB/SmolLM2-135M-Instruct"
LOCAL_MODEL   = Path("outputs/smollm2_base")
OUTPUT_LORA   = Path("outputs/smollm2_lora")
OUTPUT_MERGED = Path("outputs/smollm2_merged")
DATA_TRAIN    = Path("data/turkish_chat_train.jsonl")
DATA_EVAL     = Path("data/turkish_chat_eval.jsonl")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
USE_FP16 = DEVICE == "cuda"

# LoRA ayarları — SmolLM2-135M için optimize
LORA_CONFIG = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=16,                          # Rank — düşük tutuldu (hız/kalite dengesi)
    lora_alpha=32,                 # Alpha = 2 * r
    target_modules=[               # SmolLM2 mimarisi hedef katmanlar
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    lora_dropout=0.05,
    bias="none",
    inference_mode=False,
)

# Eğitim parametreleri
MAX_SEQ_LEN  = 128    # Klavye için kısa sekans yeterli
BATCH_SIZE   = 4 if DEVICE == "cuda" else 2
GRAD_ACC     = 4      # Efektif batch = 4*4 = 16
EPOCHS       = 3
LR           = 2e-4
WARMUP_STEPS = 50

print(f"  Cihaz    : {DEVICE.upper()}")
print(f"  Model    : {MODEL_ID}")
print(f"  Max Seq  : {MAX_SEQ_LEN}")
print(f"  Batch    : {BATCH_SIZE} x {GRAD_ACC} = {BATCH_SIZE * GRAD_ACC}")
print(f"  Epochs   : {EPOCHS}")
print()

# ─── 1. MODEL İNDİR ────────────────────────────────────────────────
print("[1/5] SmolLM2-135M indiriliyor (~270 MB)...")

LOCAL_MODEL.mkdir(parents=True, exist_ok=True)
OUTPUT_LORA.mkdir(parents=True, exist_ok=True)
OUTPUT_MERGED.mkdir(parents=True, exist_ok=True)

if (LOCAL_MODEL / "config.json").exists():
    print(f"  Model yerel klasorden yukleniyor: {LOCAL_MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(str(LOCAL_MODEL))
    model = AutoModelForCausalLM.from_pretrained(
        str(LOCAL_MODEL),
        torch_dtype=torch.float16 if USE_FP16 else torch.float32,
    )
else:
    print(f"  Model HuggingFace'den indiriliyor: {MODEL_ID}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16 if USE_FP16 else torch.float32,
    )
    tokenizer.save_pretrained(str(LOCAL_MODEL))
    model.save_pretrained(str(LOCAL_MODEL))
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

params_m = sum(p.numel() for p in model.parameters()) / 1e6
print(f"  OK: {params_m:.0f}M parametre, cihaz: {DEVICE}")
print()

# ─── 2. LoRA HAZIRLA ───────────────────────────────────────────────
print("[2/5] LoRA katmanları ekleniyor...")

model = get_peft_model(model, LORA_CONFIG)
model.print_trainable_parameters()
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total     = sum(p.numel() for p in model.parameters())
print(f"  Egitilecek: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")
print()

# ─── 3. VERİ SETİ HAZIRLA ─────────────────────────────────────────
print("[3/5] Veri seti hazırlanıyor...")

if not DATA_TRAIN.exists():
    print("  HATA: data/turkish_chat_train.jsonl bulunamadi!")
    print("  Once 01_generate_dataset.py calistirin.")
    sys.exit(1)

def load_jsonl(path):
    samples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            samples.append(json.loads(line.strip()))
    return samples

train_raw = load_jsonl(DATA_TRAIN)
eval_raw  = load_jsonl(DATA_EVAL)


def extract_text(sample: dict) -> str:
    """Farklı formatlardaki örneklerden ham metin çıkar."""
    if sample.get("full_text") is not None:
        return sample["full_text"]
    if sample.get("text") is not None:
        return sample["text"]
    if sample.get("prompt") is not None and sample.get("completion") is not None:
        return sample["prompt"] + " " + sample["completion"]
    if sample.get("instruction") is not None and sample.get("output") is not None:
        inp = sample.get("input", "")
        if inp is None: inp = ""
        return f"{sample['instruction']} {inp} {sample['output']}"
    return ""


def tokenize(examples):
    texts = [extract_text(s) for s in examples["data"]]
    tokens = tokenizer(
        texts,
        truncation=True,
        max_length=MAX_SEQ_LEN,
        padding="max_length",
        return_tensors=None,
    )
    tokens["labels"] = tokens["input_ids"].copy()
    return tokens


train_ds = Dataset.from_dict({"data": train_raw})
eval_ds  = Dataset.from_dict({"data": eval_raw})

train_ds = train_ds.map(
    tokenize, batched=True, batch_size=64,
    remove_columns=["data"],
    desc="Tokenize train",
)
eval_ds = eval_ds.map(
    tokenize, batched=True, batch_size=64,
    remove_columns=["data"],
    desc="Tokenize eval",
)

print(f"  Train: {len(train_ds)} ornek")
print(f"  Eval : {len(eval_ds)} ornek")
print()

# ─── 4. EĞİTİM ────────────────────────────────────────────────────
print("[4/5] LoRA Fine-Tuning basliyor...")
print(f"  Tahmini sure: {'~30 dk (GPU)' if DEVICE == 'cuda' else '~2-4 saat (CPU)'}")
print()

training_args = TrainingArguments(
    output_dir=str(OUTPUT_LORA),
    max_steps=10,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACC,
    learning_rate=LR,
    warmup_steps=WARMUP_STEPS,
    weight_decay=0.01,
    fp16=USE_FP16,
    bf16=False,
    logging_steps=2,
    eval_strategy="no",
    save_strategy="no",
    load_best_model_at_end=False,
    report_to="none",
    dataloader_num_workers=0,    # Windows uyumluluğu
    remove_unused_columns=False,
    label_names=["labels"],
)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,    # Causal LM, not masked
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=eval_ds,
    data_collator=data_collator,
)

trainer.train()
trainer.save_model(str(OUTPUT_LORA))
print()
print("  Fine-tuning tamamlandi!")

# ─── 5. MODEL BİRLEŞTİR VE KAYDET ─────────────────────────────────
print("[5/5] LoRA adaptor base model ile birlestiriliyor...")

# LoRA adapter'ı base model'e gömme
from peft import PeftModel

base_model = AutoModelForCausalLM.from_pretrained(
    str(LOCAL_MODEL),
    torch_dtype=torch.float32,
)
lora_model = PeftModel.from_pretrained(base_model, str(OUTPUT_LORA))
merged = lora_model.merge_and_unload()
merged.eval()

merged.save_pretrained(str(OUTPUT_MERGED))
tokenizer.save_pretrained(str(OUTPUT_MERGED))

print(f"  Birlestirilmis model: {OUTPUT_MERGED}/")
params_m = sum(p.numel() for p in merged.parameters()) / 1e6
print(f"  Parametreler: {params_m:.0f}M")

print()
print("=" * 60)
print("  FINE-TUNING TAMAMLANDI!")
print("=" * 60)
print()
print(f"  Cikti: {OUTPUT_MERGED}/")
print()
print("  Siradaki adim:")
print("  python 03_export_tflite.py")
