# -*- coding: utf-8 -*-
"""
=============================================================
03_export_tflite.py
Solenz AI Keyboard — SmolLM2 → ONNX INT8 → Android
=============================================================
Fine-tuned (veya base) SmolLM2-135M modelini:
  1. ONNX formatına export eder
  2. INT8 dynamic quantization uygular  (~45-55 MB)
  3. Android assets/ klasörüne kopyalar
  4. Kotlin GhostTextInterpreter ile uyumlu format

Çalıştırma: python 03_export_tflite.py
Çıktı     : outputs/tflite/ghost_text_model.tflite (~50 MB)
            outputs/tflite/tokenizer/
"""

import os, sys, shutil, time, gc, warnings
warnings.filterwarnings("ignore")
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

print("=" * 60)
print("  SmolLM2-135M ONNX INT8 Export")
print("=" * 60)

import torch
import numpy as np
from pathlib import Path

# ─── Model Yolu Seçimi ─────────────────────────────────────────────
# Öncelik: fine-tuned > base model
MERGED_PATH = Path("outputs/smollm2_merged")
BASE_PATH   = Path("outputs/smollm2_base")
HF_MODEL_ID = "HuggingFaceTB/SmolLM2-135M-Instruct"

if MERGED_PATH.exists() and any(MERGED_PATH.iterdir()):
    MODEL_PATH = MERGED_PATH
    print(f"  Model: Fine-tuned ({MERGED_PATH})")
elif BASE_PATH.exists() and any(BASE_PATH.iterdir()):
    MODEL_PATH = BASE_PATH
    print(f"  Model: Base ({BASE_PATH})")
else:
    MODEL_PATH = None
    print(f"  Model: HuggingFace ({HF_MODEL_ID}) - indirilecek")

OUTPUT_DIR = Path("outputs/tflite")
ONNX_FP32  = Path("outputs/smollm2_fp32.onnx")
ONNX_INT8  = Path("outputs/smollm2_int8.onnx")
TFLITE_OUT = OUTPUT_DIR / "ghost_text_model.tflite"
TOK_DIR    = OUTPUT_DIR / "tokenizer"
SEQ_LEN    = 64    # Klavye bağlamı için yeterli

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
TOK_DIR.mkdir(parents=True, exist_ok=True)

# ─── Import ────────────────────────────────────────────────────────
from transformers import AutoModelForCausalLM, AutoTokenizer
import onnxruntime as ort
from onnxruntime.quantization import quantize_dynamic, QuantType

# ─── 1. MODEL YUKLE ────────────────────────────────────────────────
print()
print("[1/5] Model yukleniyor...")
t0 = time.time()

if MODEL_PATH:
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_PATH), local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        str(MODEL_PATH),
        torch_dtype=torch.float32,
        local_files_only=True,
    )
else:
    # İndir
    LOCAL_DIR = str(BASE_PATH)
    BASE_PATH.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        HF_MODEL_ID,
        torch_dtype=torch.float32,
    )
    tokenizer.save_pretrained(LOCAL_DIR)
    model.save_pretrained(LOCAL_DIR)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = model.float()
model.eval()

elapsed = time.time() - t0
params_m = sum(p.numel() for p in model.parameters()) / 1e6
print(f"  OK: {params_m:.1f}M parametre, {elapsed:.1f}s")

# ─── 2. INFERENCE TESTİ ────────────────────────────────────────────
print()
print("[2/5] Inference testi (Türkçe tamamlama)...")

test_prompts = [
    "merhaba nasılsın",
    "yarın buluşalım",
    "çok teşekkür",
    "iyi geceler",
]

with torch.no_grad():
    for prompt in test_prompts:
        inp = tokenizer(prompt, return_tensors="pt",
                        max_length=SEQ_LEN, padding="max_length", truncation=True)
        t_inf = time.perf_counter()
        out = model(**inp)
        ms = (time.perf_counter() - t_inf) * 1000
        seq_len = int(inp["attention_mask"].sum())
        nid = int(torch.argmax(out.logits[0, seq_len-1, :]))
        ntok = tokenizer.decode([nid], skip_special_tokens=True).strip()
        # Top-3 confidence
        probs = torch.softmax(out.logits[0, seq_len-1, :], dim=-1)
        top3 = torch.topk(probs, 3)
        conf = float(top3.values[0]) * 100
        print(f"  '{prompt}' -> '{ntok}' (confidence: {conf:.1f}%, {ms:.0f}ms)")

# ─── 3. ONNX EXPORT ────────────────────────────────────────────────
print()
print("[3/5] ONNX FP32 export (legacy exporter, weights dahil)...")
print("  Bu adim ~2-5 dakika surebilir...")

class SmolLMWrapper(torch.nn.Module):
    """
    Sadece input_ids + attention_mask alan,
    son token'ın logits'ini döndüren basit wrapper.
    KV-Cache olmayan versiyonu (Android uyumlu).
    """
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        out = self.m(input_ids=input_ids, attention_mask=attention_mask)
        return out.logits   # [batch, seq, vocab]


wrapped = SmolLMWrapper(model)
wrapped.eval()

dummy_ids  = torch.zeros(1, SEQ_LEN, dtype=torch.long)
dummy_mask = torch.ones(1, SEQ_LEN, dtype=torch.long)

t0 = time.time()
with torch.no_grad():
    torch.onnx.export(
        wrapped,
        args=(dummy_ids, dummy_mask),
        f=str(ONNX_FP32),
        opset_version=18,
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids":      {0: "batch", 1: "seq_len"},
            "attention_mask": {0: "batch", 1: "seq_len"},
            "logits":         {0: "batch", 1: "seq_len"},
        },
        do_constant_folding=True,
        dynamo=False,    # Eski exporter — weights dosyaya gömülür
    )

elapsed = time.time() - t0
fp32_size = sum(
    f.stat().st_size for f in [ONNX_FP32] +
    list(ONNX_FP32.parent.glob("smollm2_fp32*"))
) / 1e6
print(f"  OK: {fp32_size:.0f} MB toplam, {elapsed:.0f}s")

# Bellek temizle
del model, wrapped
gc.collect()

# ─── 4. INT8 QUANTIZATION ─────────────────────────────────────────
print()
print("[4/5] INT8 dynamic quantization (~1-3 dakika)...")
t0 = time.time()

quantize_dynamic(
    model_input=str(ONNX_FP32),
    model_output=str(ONNX_INT8),
    weight_type=QuantType.QInt8,
)

elapsed = time.time() - t0

# Boyut hesapla (external data varsa dahil et)
int8_files = [ONNX_INT8] + list(Path("outputs").glob("smollm2_int8*"))
int8_size = sum(f.stat().st_size for f in int8_files if f.exists()) / 1e6
reduction = (1 - int8_size / max(fp32_size, 1)) * 100
print(f"  OK: {int8_size:.1f} MB (-%{abs(reduction):.0f} kuculme), {elapsed:.0f}s")

# ─── BOYUT KONTROLÜ ────────────────────────────────────────────────
if int8_size > 100:
    print()
    print(f"  [UYARI] Model {int8_size:.0f} MB — 100 MB'tan buyuk!")
    print("  Cozum: Vocab boyutunu kucult veya katman sayisini azalt.")
    print("  Mevcut haliyle calisir ama APK buyur (Split APK gerekebilir).")
else:
    print(f"  [OK] Model boyutu hedef araliginda ({int8_size:.0f} MB)")

# ─── 5. ANDROID ASSETS'E KOPYALA ──────────────────────────────────
print()
print("[5/5] Android assets'e kopyalanıyor...")

# Önce outputs/tflite/ altına kopyala
shutil.copy2(str(ONNX_INT8), str(TFLITE_OUT))
# External data varsa onu da kopyala
for ext_f in Path("outputs").glob("smollm2_int8.onnx.data"):
    shutil.copy2(str(ext_f), str(OUTPUT_DIR / ext_f.name))

# Tokenizer kaydet
tokenizer.save_pretrained(str(TOK_DIR))
tok_files = [f.name for f in TOK_DIR.iterdir() if f.is_file()]
print(f"  Tokenizer: {', '.join(tok_files)}")

# FlorisBoard assets/ bul
import glob as _glob
candidates = _glob.glob(
    r"C:/Users/senol/OneDrive/Desktop/*/florisboard-main/florisboard-main/app/src/main/assets"
)
ASSETS = Path(candidates[0]) if candidates else None

if ASSETS and ASSETS.exists():
    dst_model = ASSETS / "ghost_text_model.tflite"
    dst_tok   = ASSETS / "tokenizer"

    shutil.copy2(str(TFLITE_OUT), str(dst_model))
    size_mb = dst_model.stat().st_size / 1e6

    if dst_tok.exists():
        shutil.rmtree(str(dst_tok))
    shutil.copytree(str(TOK_DIR), str(dst_tok))

    print(f"  [OK] Model  -> {dst_model} ({size_mb:.1f} MB)")
    print(f"  [OK] Tokenizer -> {dst_tok}/")
    COPIED = True
else:
    print(f"  [UYARI] assets/ bulunamadi.")
    print(f"  Manuel: {TFLITE_OUT} -> .../app/src/main/assets/ghost_text_model.tflite")
    COPIED = False

# ─── ONNX RT DOGRULAMA ─────────────────────────────────────────────
print()
print("[DOGRULAMA] ONNX Runtime ile test...")
try:
    # External data varsa working dir ayarla
    os.chdir(str(OUTPUT_DIR))
    sess_opts = ort.SessionOptions()
    sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess = ort.InferenceSession(
        str(TFLITE_OUT),
        sess_options=sess_opts,
        providers=["CPUExecutionProvider"]
    )
    os.chdir(str(Path(__file__).parent))

    test_text = "iyi gunler nasilsin"
    inp2 = tokenizer(test_text, return_tensors="np",
                     max_length=SEQ_LEN, padding="max_length", truncation=True)
    t0 = time.perf_counter()
    res = sess.run(None, {
        "input_ids":      inp2["input_ids"].astype(np.int64),
        "attention_mask": inp2["attention_mask"].astype(np.int64),
    })
    ms = (time.perf_counter() - t0) * 1000

    seq2 = int(inp2["attention_mask"].sum())
    logits = res[0][0, seq2-1, :]       # son token logits
    probs  = np.exp(logits) / np.sum(np.exp(logits))   # softmax
    top5_ids = np.argsort(probs)[-5:][::-1]
    top5_tok = [tokenizer.decode([i], skip_special_tokens=True).strip() for i in top5_ids]
    top5_conf = [f"{probs[i]*100:.1f}%" for i in top5_ids]

    print(f"  Prefix: '{test_text}'")
    print(f"  Top-5 tahmin:")
    for tok, conf in zip(top5_tok, top5_conf):
        print(f"    '{tok}' ({conf})")
    print(f"  Latency: {ms:.0f}ms (CPU, INT8)")
    print()

    # Confidence score kontrolü
    best_conf = float(probs[top5_ids[0]]) * 100
    if best_conf > 10:
        print(f"  [OK] Confidence score > 10% — ghost text gosterilecek")
    else:
        print(f"  [DUSUK] Confidence {best_conf:.1f}% — ghost text gizlenecek")

except Exception as e:
    print(f"  [UYARI] ONNX RT testi basarisiz: {e}")

# ─── FP32 TEMİZLE ──────────────────────────────────────────────────
print()
print("FP32 gecici dosyalar temizleniyor...")
for fp32_f in [ONNX_FP32] + list(Path("outputs").glob("smollm2_fp32*")):
    if fp32_f.exists():
        fp32_f.unlink()
        print(f"  Silindi: {fp32_f.name}")

# ─── ÖZET ──────────────────────────────────────────────────────────
print()
print("=" * 60)
print("  EXPORT TAMAMLANDI!")
print("=" * 60)
print(f"  Model   : {TFLITE_OUT} ({int8_size:.1f} MB)")
print(f"  Format  : ONNX INT8 (Android ONNX Runtime ile calisir)")
print(f"  Assets  : {'Kopyalandi' if COPIED else 'Manuel kopyalama gerekli'}")
print()
print("  SIRADAKI ADIM: gradlew assembleDebug")
print(f"  APK: florisboard-main/app/build/outputs/apk/debug/app-debug.apk")
