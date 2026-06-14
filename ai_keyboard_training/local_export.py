# -*- coding: utf-8 -*-
"""
=============================================================
local_export.py
Solenz AI Keyboard — INT8 ONNX Export (Android Hazır)
=============================================================
SmolLM2-135M (fine-tuned) modelini:
  1. ONNX FP32 formatına çevirir
  2. INT8 Dynamic Quantization uygular (~45-55 MB)
  3. Tokenizer'ı kopyalar
  4. FlorisBoard assets/ klasörüne otomatik taşır

Gereksinimler:
  pip install onnx onnxruntime optimum[onnxruntime]

Çalıştırma:
  python local_export.py

Çıktı:
  outputs/tflite/ghost_text_model.onnx   (~50 MB, INT8)
  outputs/tflite/tokenizer/              (tokenizer dosyaları)
"""

import os
import sys
import shutil
import time
import gc
import warnings

warnings.filterwarnings("ignore")
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

sys.stdout = __import__("io").TextIOWrapper(
    sys.stdout.buffer, encoding="utf-8", errors="replace"
)

print("=" * 65)
print("  Solenz AI Keyboard — INT8 ONNX Export")
print("=" * 65)
print()

import torch
import numpy as np
from pathlib import Path

# ─── Model Yolu Önceliği: fine-tuned > base ───────────────────────
BASE_DIR      = Path(__file__).parent
MERGED_PATH   = BASE_DIR / "outputs" / "smollm2_merged"
BASE_PATH     = BASE_DIR / "outputs" / "smollm2_base"
OUTPUT_DIR    = BASE_DIR / "outputs" / "tflite"
ONNX_FP32     = BASE_DIR / "outputs" / "_temp_fp32.onnx"
ONNX_INT8     = OUTPUT_DIR / "ghost_text_model.onnx"
TOK_DIR       = OUTPUT_DIR / "tokenizer"
SEQ_LEN       = 64   # Klavye için 64 token ideal (hız/kalite dengesi)

if MERGED_PATH.exists() and any(MERGED_PATH.iterdir()):
    MODEL_PATH = MERGED_PATH
    print(f"  Model    : Fine-tuned ({MERGED_PATH.name})")
elif BASE_PATH.exists() and any(BASE_PATH.iterdir()):
    MODEL_PATH = BASE_PATH
    print(f"  Model    : Base ({BASE_PATH.name}) — fine-tuning yapılmadıysa bu kullanılır")
else:
    print("  HATA: Model bulunamadı!")
    print("  Önce local_train.py çalıştırın.")
    sys.exit(1)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
TOK_DIR.mkdir(parents=True, exist_ok=True)

# ─── Import Kontrolü ──────────────────────────────────────────────
try:
    import onnx
    import onnxruntime as ort
    from onnxruntime.quantization import quantize_dynamic, QuantType
except ImportError:
    print("  HATA: ONNX kütüphaneleri eksik!")
    print("  Kurmak için:")
    print("    pip install onnx onnxruntime")
    sys.exit(1)

from transformers import AutoModelForCausalLM, AutoTokenizer

# ─── 1. MODEL YÜKLE ──────────────────────────────────────────────
print("[1/5] Model yükleniyor...")
t0 = time.time()

tokenizer = AutoTokenizer.from_pretrained(str(MODEL_PATH), local_files_only=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    str(MODEL_PATH),
    torch_dtype=torch.float32,   # ONNX export için float32 zorunlu
    local_files_only=True,
)
model = model.float().eval()

params_m = sum(p.numel() for p in model.parameters()) / 1e6
print(f"  OK: {params_m:.1f}M parametre — {time.time()-t0:.1f}s")
print()

# ─── 2. TORCH INFERENCE TESTİ ────────────────────────────────────
print("[2/5] Türkçe tamamlama testi (PyTorch)...")

TEST_PROMPTS = [
    "merhaba nasılsın",
    "yarın buluşalım",
    "çok teşekkür",
    "toplantı saat",
    "iyi geceler",
]

with torch.no_grad():
    for prompt in TEST_PROMPTS:
        enc = tokenizer(
            prompt,
            return_tensors="pt",
            max_length=SEQ_LEN,
            padding="max_length",
            truncation=True,
        )
        t_inf = time.perf_counter()
        out = model(**enc)
        ms = (time.perf_counter() - t_inf) * 1000

        # Son gerçek token'ın logits'ini al
        seq_end = int(enc["attention_mask"].sum()) - 1
        logits = out.logits[0, seq_end, :]
        probs = torch.softmax(logits, dim=-1)
        top_id = int(torch.argmax(probs))
        top_tok = tokenizer.decode([top_id], skip_special_tokens=True).strip()
        conf = float(probs[top_id]) * 100

        print(f"  '{prompt}' → '{top_tok}' ({conf:.1f}%, {ms:.0f}ms)")

print()

# ─── 3. ONNX FP32 EXPORT ─────────────────────────────────────────
print("[3/5] ONNX FP32 export ediliyor...")
print("  (2-5 dakika sürebilir, lütfen bekleyin...)")


class SmolLMONNXWrapper(torch.nn.Module):
    """
    ONNX için sadeleştirilmiş wrapper.
    input_ids + attention_mask alır, full logits döndürür.
    KV-Cache yok = Android uyumlu, basit, güvenilir.
    """

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(
        self,
        input_ids: torch.Tensor,       # [batch, seq_len] int64
        attention_mask: torch.Tensor,  # [batch, seq_len] int64
    ) -> torch.Tensor:
        output = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,           # KV-Cache KAPALI
        )
        return output.logits           # [batch, seq_len, vocab_size]


wrapper = SmolLMONNXWrapper(model).eval()

dummy_ids  = torch.zeros(1, SEQ_LEN, dtype=torch.long)
dummy_mask = torch.ones(1, SEQ_LEN, dtype=torch.long)

t0 = time.time()
with torch.no_grad():
    torch.onnx.export(
        wrapper,
        args=(dummy_ids, dummy_mask),
        f=str(ONNX_FP32),
        opset_version=17,           # ONNX Runtime 1.16+ uyumlu
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids":      {0: "batch"},
            "attention_mask": {0: "batch"},
            "logits":         {0: "batch"},
        },
        do_constant_folding=True,   # Sabit ifadeleri katla → küçük boyut
        export_params=True,
        dynamo=False,               # Klasik exporter — daha kararlı
    )

# ONNX modelini doğrula
try:
    onnx_model = onnx.load(str(ONNX_FP32))
    onnx.checker.check_model(onnx_model)
    del onnx_model
    gc.collect()
except Exception as e:
    print(f"  UYARI: ONNX doğrulama uyarısı: {e}")
    print("  Model yine de kullanılabilir olabilir, devam ediliyor...")

fp32_mb = ONNX_FP32.stat().st_size / 1e6
print(f"  OK: {fp32_mb:.0f} MB FP32 — {time.time()-t0:.0f}s")

# Belleği temizle
del model, wrapper
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()
print()

# ─── 4. INT8 DYNAMIC QUANTIZATION ────────────────────────────────
print("[4/5] INT8 Dynamic Quantization uygulanıyor...")
print("  (Tüm ağırlıklar INT8'e dönüştürülüyor...)")
t0 = time.time()

quantize_dynamic(
    model_input=str(ONNX_FP32),
    model_output=str(ONNX_INT8),
    weight_type=QuantType.QInt8,
    # Aşağıdaki operatörler INT8'e dönüştürülür:
    # MatMul, Attention, LSTM vb. (transformer için yeterli)
)

int8_mb = ONNX_INT8.stat().st_size / 1e6
reduction = (1 - int8_mb / fp32_mb) * 100
print(f"  OK: {int8_mb:.1f} MB INT8 (-%{reduction:.0f} küçülme) — {time.time()-t0:.0f}s")

if int8_mb > 100:
    print(f"\n  UYARI: Model {int8_mb:.0f} MB, 100 MB'dan büyük!")
    print("  Android'de Split APK gerekebilir.")
    print("  Çözüm: SEQ_LEN=32 ile yeniden deneyin.")
elif int8_mb > 50:
    print(f"  BILGI: Model {int8_mb:.0f} MB — hedef 50 MB, biraz büyük ama çalışır.")
else:
    print(f"  MÜKEMMEL: Model {int8_mb:.1f} MB — hedef 50 MB'ın altında!")

# Geçici FP32 dosyasını sil
ONNX_FP32.unlink(missing_ok=True)
print()

# ─── 5. TOKENİZER + FLORISBOARD ASSETS ───────────────────────────
print("[5/5] Tokenizer kaydediliyor ve assets'e kopyalanıyor...")

# Tokenizer'ı outputs/tflite/tokenizer/ altına kaydet
tokenizer.save_pretrained(str(TOK_DIR))
tok_files = [f.name for f in TOK_DIR.iterdir() if f.is_file()]
print(f"  Tokenizer dosyaları: {', '.join(tok_files)}")

# FlorisBoard assets/ klasörünü bul
import glob as _glob
candidates = _glob.glob(
    os.path.join(
        str(BASE_DIR.parent),
        "florisboard-main",
        "florisboard-main",
        "app",
        "src",
        "main",
        "assets",
    )
)

if not candidates:
    # Daha geniş arama
    candidates = _glob.glob(
        r"C:\Users\senol\**\assets",
        recursive=True,
    )
    candidates = [c for c in candidates if "florisboard" in c.lower()]

ASSETS = Path(candidates[0]) if candidates else None
COPIED = False

if ASSETS and ASSETS.exists():
    # Model kopyala
    dst_model = ASSETS / "ghost_text_model.onnx"
    shutil.copy2(str(ONNX_INT8), str(dst_model))
    print(f"  Model  → {dst_model} ({dst_model.stat().st_size/1e6:.1f} MB)")

    # Tokenizer kopyala
    dst_tok = ASSETS / "tokenizer"
    if dst_tok.exists():
        shutil.rmtree(str(dst_tok))
    shutil.copytree(str(TOK_DIR), str(dst_tok))
    print(f"  Tokenizer → {dst_tok}/")
    COPIED = True
else:
    print()
    print("  FlorisBoard assets/ klasörü otomatik bulunamadı.")
    print("  Manuel kopyalama için:")
    print()
    print(f"    Model    : {ONNX_INT8}")
    print(f"    Hedef    : .../florisboard/app/src/main/assets/ghost_text_model.onnx")
    print()
    print(f"    Tokenizer: {TOK_DIR}/")
    print(f"    Hedef    : .../florisboard/app/src/main/assets/tokenizer/")

# ─── ONNX RUNTIME DOĞRULAMA TESTİ ─────────────────────────────────
print()
print("[Doğrulama] ONNX Runtime INT8 inference testi...")

try:
    sess_opts = ort.SessionOptions()
    sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess_opts.intra_op_num_threads = 4   # CPU thread sayısı
    sess_opts.inter_op_num_threads = 1

    sess = ort.InferenceSession(
        str(ONNX_INT8),
        sess_options=sess_opts,
        providers=["CPUExecutionProvider"],
    )

    test_text = "iyi günler nasılsın"
    enc2 = tokenizer(
        test_text,
        return_tensors="np",
        max_length=SEQ_LEN,
        padding="max_length",
        truncation=True,
    )

    t0 = time.perf_counter()
    result = sess.run(
        None,
        {
            "input_ids":      enc2["input_ids"].astype(np.int64),
            "attention_mask": enc2["attention_mask"].astype(np.int64),
        },
    )
    ms = (time.perf_counter() - t0) * 1000

    # Son token tahmini
    seq_end = int(enc2["attention_mask"].sum()) - 1
    logits  = result[0][0, seq_end, :]
    probs   = np.exp(logits - logits.max())
    probs   = probs / probs.sum()
    top5    = np.argsort(probs)[-5:][::-1]

    print(f"  Prefix : '{test_text}'")
    print(f"  Top-5  :")
    for idx in top5:
        tok  = tokenizer.decode([idx], skip_special_tokens=True).strip()
        conf = probs[idx] * 100
        print(f"    '{tok}' ({conf:.1f}%)")
    print(f"  Latency: {ms:.0f}ms (CPU INT8)")

    best_conf = float(probs[top5[0]]) * 100
    if best_conf > 5:
        print(f"\n  Model çalışıyor! Ghost text gösterilecek.")
    else:
        print(f"\n  UYARI: Düşük confidence ({best_conf:.1f}%) — daha fazla eğitim verisi ekleyin.")

except Exception as e:
    print(f"  ONNX RT testi başarısız: {e}")
    import traceback
    traceback.print_exc()

# ─── ÖZET ──────────────────────────────────────────────────────────
print()
print("=" * 65)
print("  EXPORT TAMAMLANDI!")
print("=" * 65)
print(f"  ONNX INT8 : {ONNX_INT8}")
print(f"  Boyut     : {int8_mb:.1f} MB")
print(f"  Tokenizer : {TOK_DIR}/")
print(f"  Assets    : {'Kopyalandı ✓' if COPIED else 'Manuel kopyalama gerekli'}")
print()
print("  Sıradaki adım: FlorisBoard Kotlin entegrasyonu")
print("  Kotlin sınıfı: GhostTextEngine.kt")
