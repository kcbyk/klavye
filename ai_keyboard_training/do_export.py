# -*- coding: utf-8 -*-
"""
Qwen2-0.5B (local) -> ONNX INT8 -> Android assets
v3: legacy torch.onnx + yeni onnxruntime quantize API
"""
import sys, time, shutil, os, warnings
from pathlib import Path

os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'
warnings.filterwarnings('ignore')

import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

print("=" * 60)
print("  Solenz AI Keyboard - ONNX Export v3")
print("=" * 60)

import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
import onnxruntime as ort
from onnxruntime.quantization import quantize_dynamic, QuantType

MODEL_DIR  = Path("outputs/qwen2_model")
OUTPUT_DIR = Path("outputs/tflite")
ONNX_FP32  = Path("outputs/model_fp32.onnx")
ONNX_INT8  = Path("outputs/model_int8.onnx")
TFLITE_OUT = OUTPUT_DIR / "ghost_text_model.tflite"
TOK_DIR    = OUTPUT_DIR / "tokenizer"
SEQ_LEN    = 64

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
TOK_DIR.mkdir(parents=True, exist_ok=True)

# ── 1. MODEL YUKLE ─────────────────────────────────────────────────
print(f"[1/5] Model yukleniyor: {MODEL_DIR}")
t0 = time.time()
tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR), local_files_only=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    str(MODEL_DIR),
    local_files_only=True,
    low_cpu_mem_usage=True,
)
model = model.float()   # float32 garantisi
model.eval()
elapsed = time.time() - t0
params_m = sum(p.numel() for p in model.parameters()) / 1e6
print(f"  OK: {params_m:.0f}M parametre, {elapsed:.0f}s")
print()

# ── 2. TEST ────────────────────────────────────────────────────────
print("[2/5] Inference testi...")
with torch.no_grad():
    for txt in ["merhaba nasilsin", "iyi gunler"]:
        inp = tokenizer(txt, return_tensors="pt",
                        max_length=SEQ_LEN, padding="max_length", truncation=True)
        out = model(**inp)
        seq_r = int(inp["attention_mask"].sum())
        nid = int(torch.argmax(out.logits[0, seq_r-1, :]))
        print(f"  '{txt}' -> '{tokenizer.decode([nid], skip_special_tokens=True).strip()}'")
print()

# ── 3. ONNX EXPORT (legacy torch.onnx, opset 18) ──────────────────
print("[3/5] ONNX FP32 export (legacy exporter)...")
print("  Lutfen bekleyin (~5-15 dakika)...")

class QwenWrapper(torch.nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        out = self.m(input_ids=input_ids, attention_mask=attention_mask)
        return out.logits

wrapped = QwenWrapper(model)
wrapped.eval()

dummy_ids  = torch.zeros(1, SEQ_LEN, dtype=torch.long)
dummy_mask = torch.ones(1, SEQ_LEN, dtype=torch.long)

t0 = time.time()
with torch.no_grad():
    # Eski API'yi zorla kullan (dynamo=False)
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
        dynamo=False,        # <<< KRITIK: eski exporter, weights dahil eder
    )

elapsed = time.time() - t0
fp32_size = ONNX_FP32.stat().st_size / 1e6
print(f"  OK: {fp32_size:.0f} MB, {elapsed:.0f}s")
print()

# Bellek temizle
del model, wrapped
import gc; gc.collect()

# ── 4. INT8 QUANTIZATION (yeni API) ───────────────────────────────
print("[4/5] INT8 dynamic quantization...")
print("  (5-15 dakika surebilir...)")
t0 = time.time()

# Yeni onnxruntime API'si: optimize_model parametresi yok
quantize_dynamic(
    model_input=str(ONNX_FP32),
    model_output=str(ONNX_INT8),
    weight_type=QuantType.QInt8,
)

elapsed = time.time() - t0
int8_size = ONNX_INT8.stat().st_size / 1e6
reduction = (1 - int8_size / fp32_size) * 100
print(f"  OK: {int8_size:.0f} MB (-%{reduction:.0f} kuculme), {elapsed:.0f}s")
print()

# ── 5. ASSETS'E KOPYALA ────────────────────────────────────────────
print("[5/5] Android assets'e kopyalanıyor...")

shutil.copy2(str(ONNX_INT8), str(TFLITE_OUT))
tokenizer.save_pretrained(str(TOK_DIR))

ASSETS = (Path("..") / "florisboard-main" / "florisboard-main" /
          "app" / "src" / "main" / "assets").resolve()

COPIED = False
if ASSETS.exists():
    dst_model = ASSETS / "ghost_text_model.tflite"
    dst_tok   = ASSETS / "tokenizer"
    shutil.copy2(str(TFLITE_OUT), str(dst_model))
    if dst_tok.exists(): shutil.rmtree(str(dst_tok))
    shutil.copytree(str(TOK_DIR), str(dst_tok))
    print(f"  OK Model    -> {dst_model} ({int8_size:.0f} MB)")
    print(f"  OK Tokenizer-> {dst_tok}/")
    COPIED = True
else:
    print(f"  HATA: assets/ yok: {ASSETS}")

# ── DOGRULAMA ─────────────────────────────────────────────────────
print()
print("[DOGRULAMA] ONNX Runtime test...")
sess = ort.InferenceSession(str(ONNX_INT8), providers=["CPUExecutionProvider"])
inp2 = tokenizer("iyi gunler nasilsiniz", return_tensors="np",
                 max_length=SEQ_LEN, padding="max_length", truncation=True)
t0 = time.perf_counter()
res = sess.run(None, {
    "input_ids":      inp2["input_ids"].astype(np.int64),
    "attention_mask": inp2["attention_mask"].astype(np.int64),
})
ms = (time.perf_counter() - t0) * 1000
seq2 = int(inp2["attention_mask"].sum())
nid2 = int(np.argmax(res[0][0, seq2-1, :]))
print(f"  'iyi gunler nasilsiniz' -> '{tokenizer.decode([nid2], skip_special_tokens=True).strip()}'")
print(f"  Latency: {ms:.0f}ms (CPU, INT8)")

print()
print("=" * 60)
print("  EXPORT TAMAMLANDI!")
print("=" * 60)
print(f"  Model boyutu : {int8_size:.0f} MB")
print(f"  Assets       : {'Kopyalandi' if COPIED else 'Manuel kopyala gerekli'}")
print()
if COPIED:
    print("  SIRADAKI ADIM: gradlew assembleDebug")
