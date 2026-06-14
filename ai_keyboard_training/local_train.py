# -*- coding: utf-8 -*-
"""
=============================================================
local_train.py
Solenz AI Keyboard — TAM YEREL Fine-Tuning (API Gerektirmez)
=============================================================
SmolLM2-135M modelini TAMAMEN OFFLINE, kendi donanımında
Türkçe klavye verisiyle eğitir.

Ön koşul:
  - outputs/smollm2_base/  klasöründe model mevcut (zaten var)
  - data/turkish_chat_train.jsonl ve eval.jsonl mevcut (zaten var)

Çalıştırma:
  python local_train.py

Çıktı:
  outputs/smollm2_lora/    <- LoRA adapter ağırlıkları
  outputs/smollm2_merged/  <- Birleştirilmiş, export'a hazır model
"""

import os
import sys
import json
import warnings
import gc
import time

# Tüm HuggingFace ve Torch uyarılarını sustur
warnings.filterwarnings("ignore")
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_OFFLINE"] = "1"          # İnternet bağlantısı DEVRE DIŞI
os.environ["TRANSFORMERS_OFFLINE"] = "1"     # HuggingFace API'ye gitme

# Windows için UTF-8 çıktı
sys.stdout = __import__("io").TextIOWrapper(
    sys.stdout.buffer, encoding="utf-8", errors="replace"
)

# ─── Başlık ────────────────────────────────────────────────────────
print("=" * 65)
print("  Solenz AI Keyboard — Yerel Fine-Tuning (Offline)")
print("=" * 65)
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
from peft import get_peft_model, LoraConfig, TaskType, PeftModel

# ─── Yol Ayarları ─────────────────────────────────────────────────
BASE_DIR      = Path(__file__).parent
LOCAL_MODEL   = BASE_DIR / "outputs" / "smollm2_base"
OUTPUT_LORA   = BASE_DIR / "outputs" / "smollm2_lora"
OUTPUT_MERGED = BASE_DIR / "outputs" / "smollm2_merged"
DATA_TRAIN    = BASE_DIR / "data" / "turkish_chat_train.jsonl"
DATA_EVAL     = BASE_DIR / "data" / "turkish_chat_eval.jsonl"

# ─── Donanım Tespiti ──────────────────────────────────────────────
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
USE_FP16 = False  # CPU'da FP16 desteklenmez, FP32 kullan

if DEVICE == "cuda":
    gpu = torch.cuda.get_device_name(0)
    vram = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"  Cihaz    : GPU — {gpu} ({vram:.1f} GB)")
    USE_FP16 = True
else:
    import multiprocessing
    cpu_count = multiprocessing.cpu_count()
    print(f"  Cihaz    : CPU ({cpu_count} çekirdek)")
    print("  Not      : CPU eğitimi ~1-3 saat sürebilir.")
    print("             Bilgisayarı kapatmayın!")

print()

# ─── Eğitim Parametreleri ─────────────────────────────────────────
# CPU için optimize edilmiş — küçük batch, daha az epoch ama yeterli
MAX_SEQ_LEN  = 64     # Klavye için 64 token fazlasıyla yeterli
BATCH_SIZE   = 2      # CPU'da bellek tasarrufu için 2
GRAD_ACC     = 8      # Efektif batch = 2*8 = 16 (sanal)
EPOCHS       = 3
LR           = 2e-4
WARMUP_RATIO = 0.05

# GPU varsa daha agresif ayarlar
if DEVICE == "cuda":
    BATCH_SIZE  = 8
    GRAD_ACC    = 2
    MAX_SEQ_LEN = 128

print(f"  Model    : SmolLM2-135M (yerel: {LOCAL_MODEL})")
print(f"  Seq len  : {MAX_SEQ_LEN}")
print(f"  Batch    : {BATCH_SIZE} x {GRAD_ACC} = {BATCH_SIZE * GRAD_ACC} (efektif)")
print(f"  Epochs   : {EPOCHS}")
print(f"  LR       : {LR}")
print()

# ─── GEREKSİNİM KONTROLÜ ─────────────────────────────────────────
print("[0/5] Ön kontroller...")

if not LOCAL_MODEL.exists() or not (LOCAL_MODEL / "config.json").exists():
    print(f"\n  HATA: Model bulunamadı: {LOCAL_MODEL}")
    print("  Çözüm: Aşağıdaki scripti bir KERE çalıştırın (internet gerektirir):")
    print()
    print("    python -c \"")
    print("    from transformers import AutoModelForCausalLM, AutoTokenizer")
    print("    m = 'HuggingFaceTB/SmolLM2-135M-Instruct'")
    print("    AutoTokenizer.from_pretrained(m).save_pretrained('outputs/smollm2_base')")
    print("    AutoModelForCausalLM.from_pretrained(m, torch_dtype='auto').save_pretrained('outputs/smollm2_base')")
    print("    \"")
    sys.exit(1)

if not DATA_TRAIN.exists():
    print(f"\n  HATA: Veri seti bulunamadı: {DATA_TRAIN}")
    print("  Çözüm: python 01_generate_dataset.py")
    sys.exit(1)

OUTPUT_LORA.mkdir(parents=True, exist_ok=True)
OUTPUT_MERGED.mkdir(parents=True, exist_ok=True)
print("  Tüm dosyalar mevcut. Başlanıyor...")
print()

# ─── 1. MODEL VE TOKENİZER YÜKLE ─────────────────────────────────
print("[1/5] Model yerel klasörden yükleniyor...")
t0 = time.time()

tokenizer = AutoTokenizer.from_pretrained(
    str(LOCAL_MODEL),
    local_files_only=True,   # İnternet YOK
)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

model = AutoModelForCausalLM.from_pretrained(
    str(LOCAL_MODEL),
    torch_dtype=torch.float32,  # CPU için her zaman float32
    local_files_only=True,       # İnternet YOK
)

# CPU'da gradient checkpointing belleği korur ama yavaşlatır, kapat
if DEVICE == "cpu":
    model.gradient_checkpointing = False
else:
    model.gradient_checkpointing_enable()

elapsed = time.time() - t0
params_m = sum(p.numel() for p in model.parameters()) / 1e6
print(f"  OK: {params_m:.0f}M parametre — {elapsed:.1f}s")
print()

# ─── 2. LoRA UYGULA ───────────────────────────────────────────────
print("[2/5] LoRA katmanları ekleniyor...")

lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=8,                        # CPU'da rank=8 yeterli ve hızlı
    lora_alpha=16,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",  # Attention
        "gate_proj", "up_proj", "down_proj",       # FFN
    ],
    lora_dropout=0.05,
    bias="none",
    inference_mode=False,
)

model = get_peft_model(model, lora_config)

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total     = sum(p.numel() for p in model.parameters())
print(f"  Eğitilebilir : {trainable:,} / {total:,} parametre")
print(f"  Oran         : {100*trainable/total:.2f}%")
print()

# ─── 3. VERİ SETİ YÜKLE VE HAZIRLA ─────────────────────────────────
print("[3/5] Veri seti hazırlanıyor...")


def load_jsonl(path: Path) -> list:
    """JSONL dosyasını oku."""
    samples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    return samples


def extract_text(sample: dict) -> str:
    """
    Farklı JSONL formatlarından tutarlı metin çıkar.
    Mevcut veri setindeki 3 format:
      1. {"full_text": "..."}
      2. {"text": "..."}
      3. {"prompt": "...", "completion": "..."}
      4. {"instruction": "...", "input": "...", "output": "..."}
    """
    # Format 1: full_text
    if sample.get("full_text"):
        return sample["full_text"].strip()

    # Format 2: text
    if sample.get("text"):
        return sample["text"].strip()

    # Format 3: prompt + completion
    if sample.get("prompt") and sample.get("completion"):
        return f"{sample['prompt']} {sample['completion']}".strip()

    # Format 4: instruction + input + output
    if sample.get("instruction") and sample.get("output"):
        inp = sample.get("input", "") or ""
        return f"{sample['input']} {sample['output']}".strip() if inp else sample["output"].strip()

    return ""


def tokenize_batch(examples: dict) -> dict:
    """
    Örnekleri tokenize et.
    Causal LM için labels = input_ids (self-supervised).
    """
    texts = [extract_text(s) for s in examples["data"]]

    # Boş metinleri filtrele
    texts = [t if t else tokenizer.eos_token for t in texts]

    encoded = tokenizer(
        texts,
        truncation=True,
        max_length=MAX_SEQ_LEN,
        padding="max_length",
        return_tensors=None,
    )

    # Causal LM: girdi ve etiket aynı, padding tokenları -100 ile maskele
    labels = []
    for ids, mask in zip(encoded["input_ids"], encoded["attention_mask"]):
        label_row = []
        for token_id, m in zip(ids, mask):
            # Padding tokenlarını kayıp hesabından çıkar
            label_row.append(token_id if m == 1 else -100)
        labels.append(label_row)

    encoded["labels"] = labels
    return encoded


train_raw = load_jsonl(DATA_TRAIN)
eval_raw  = load_jsonl(DATA_EVAL)

train_ds = Dataset.from_dict({"data": train_raw})
eval_ds  = Dataset.from_dict({"data": eval_raw})

train_ds = train_ds.map(
    tokenize_batch,
    batched=True,
    batch_size=64,
    remove_columns=["data"],
    desc="Train tokenize",
)
eval_ds = eval_ds.map(
    tokenize_batch,
    batched=True,
    batch_size=64,
    remove_columns=["data"],
    desc="Eval tokenize",
)

print(f"  Train : {len(train_ds)} örnek")
print(f"  Eval  : {len(eval_ds)} örnek")
print()

# ─── 4. EĞİTİM ────────────────────────────────────────────────────
print("[4/5] Fine-Tuning başlıyor...")

if DEVICE == "cpu":
    tahmini_sure = len(train_ds) * EPOCHS / 10  # ~10 örnek/sn CPU'da
    print(f"  Tahmini süre : ~{tahmini_sure/60:.0f}-{tahmini_sure/60*2:.0f} dakika (CPU)")
else:
    print("  Tahmini süre : ~10-20 dakika (GPU)")
print()

training_args = TrainingArguments(
    output_dir=str(OUTPUT_LORA),

    # Epoch ve adım ayarları
    num_train_epochs=EPOCHS,

    # Batch ayarları
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACC,

    # Öğrenme hızı
    learning_rate=LR,
    lr_scheduler_type="cosine",
    warmup_ratio=WARMUP_RATIO,

    # Optimizasyon
    weight_decay=0.01,
    max_grad_norm=1.0,
    optim="adamw_torch",

    # Precision — CPU'da her zaman FP32
    fp16=USE_FP16 and DEVICE == "cuda",
    bf16=False,

    # Loglama
    logging_steps=5,
    logging_first_step=True,

    # Değerlendirme ve kaydetme
    eval_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=1,           # Disk tasarrufu
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,

    # Raporlama — HIÇBIR harici servis yok
    report_to="none",

    # Windows uyumluluğu
    dataloader_num_workers=0,     # Windows'ta multiprocessing sorunlarını önler
    remove_unused_columns=False,
    label_names=["labels"],

    # CPU optimizasyonu — use_cpu yeni isim (no_cuda kaldırıldı)
    use_cpu=(DEVICE == "cpu"),
)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,   # Causal LM
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=eval_ds,
    data_collator=data_collator,
    tokenizer=tokenizer,
)

print("  Eğitim başlıyor... (loglar aşağıda)")
print("-" * 65)
train_result = trainer.train()
print("-" * 65)

# Metrikleri kaydet
trainer.save_metrics("train", train_result.metrics)
trainer.save_model(str(OUTPUT_LORA))
tokenizer.save_pretrained(str(OUTPUT_LORA))

final_loss = train_result.metrics.get("train_loss", "N/A")
total_time = train_result.metrics.get("train_runtime", 0)
print(f"\n  Fine-Tuning Tamamlandı!")
print(f"  Son loss   : {final_loss}")
print(f"  Süre       : {total_time/60:.1f} dakika")
print()

# ─── 5. LORA + BASE MODEL BİRLEŞTİR ─────────────────────────────
print("[5/5] LoRA adaptörü base model ile birleştiriliyor...")
print("  (Bu adım ~2-5 dakika sürebilir)")

# Önce belleği temizle
del trainer
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()

# Base model'i temiz yükle
base_model = AutoModelForCausalLM.from_pretrained(
    str(LOCAL_MODEL),
    torch_dtype=torch.float32,
    local_files_only=True,
)

# LoRA adapter'ı yükle ve birleştir
peft_model = PeftModel.from_pretrained(base_model, str(OUTPUT_LORA))
merged = peft_model.merge_and_unload()   # Adapter ağırlıkları içine gömülür
merged.eval()

# Kaydet
merged.save_pretrained(str(OUTPUT_MERGED))
tokenizer.save_pretrained(str(OUTPUT_MERGED))

merged_size_mb = sum(
    f.stat().st_size for f in OUTPUT_MERGED.iterdir()
    if f.is_file()
) / 1e6

print(f"  Kaydedildi : {OUTPUT_MERGED}")
print(f"  Boyut      : ~{merged_size_mb:.0f} MB")

# ─── ÖZET ─────────────────────────────────────────────────────────
print()
print("=" * 65)
print("  FINE-TUNING TAMAMLANDI!")
print("=" * 65)
print()
print(f"  Merged model : {OUTPUT_MERGED}/")
print()
print("  Sıradaki adım:")
print("    python 03_export_tflite.py")
print()
print("  veya direkt export için:")
print("    python local_export.py")
