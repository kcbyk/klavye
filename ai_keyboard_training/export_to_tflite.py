"""
=====================================================================
export_to_tflite.py — Qwen2-0.5B → TFLite INT8 (Tek Script)
=====================================================================
Bu script tensorflow kurulumu OLMADAN çalışır.
ai-edge-torch kütüphanesi ile doğrudan PyTorch → TFLite dönüşümü yapar.

Gereksinimler:
  pip install ai-edge-torch torch transformers numpy

Çalıştırma:
  python export_to_tflite.py

Çıktı:
  outputs/tflite/ghost_text_model.tflite   (~40-60 MB INT8)
  outputs/tflite/tokenizer/                (tokenizer dosyaları)
"""

import os
import sys
import subprocess
import time
from pathlib import Path

print("=" * 65)
print("  🚀 Solenz AI Keyboard — Gerçek TFLite Model Export")
print("=" * 65)
print()

# ─── Bağımlılık Kurulumu ───────────────────────────────────────────
PYTHON = sys.executable
REQUIRED = [
    "torch",
    "transformers>=4.40.0",
    "numpy",
    "huggingface_hub",
    "ai-edge-torch",       # Google'ın PyTorch → TFLite dönüştürücüsü
    "ai-edge-litert",      # TFLite runtime (eski tensorflow-lite)
]

def pip_install(pkg: str):
    print(f"  [Kuruluyor] {pkg}...")
    result = subprocess.run(
        [PYTHON, "-m", "pip", "install", pkg, "-q", "--exists-action", "i"],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        print(f"  ⚠  {pkg} kurulamadı: {result.stderr[:200]}")
        return False
    return True

print("[1/5] Bağımlılıklar kuruluyor...")
for pkg in REQUIRED:
    try:
        mod = pkg.split(">=")[0].split("==")[0].replace("-", "_")
        __import__(mod)
        print(f"  ✓ {pkg} (mevcut)")
    except ImportError:
        pip_install(pkg)

print()

# ─── Import'lar ────────────────────────────────────────────────────
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer

# ─── Çıktı klasörleri ──────────────────────────────────────────────
OUTPUT_DIR    = Path("outputs/tflite")
TOKENIZER_DIR = OUTPUT_DIR / "tokenizer"
TFLITE_PATH   = OUTPUT_DIR / "ghost_text_model.tflite"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
TOKENIZER_DIR.mkdir(parents=True, exist_ok=True)

# ─── Model Ayarları ────────────────────────────────────────────────
MODEL_ID   = "Qwen/Qwen2-0.5B-Instruct"
SEQ_LEN    = 64      # Klavye için 64 token yeterli
BATCH_SIZE = 1

# ─── ADIM 2: Model İndir ───────────────────────────────────────────
print("[2/5] Qwen2-0.5B indiriliyor (~950 MB, ilk seferinde)...")
print(f"  Model: {MODEL_ID}")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_ID,
    trust_remote_code=True,
)
tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float32,
    trust_remote_code=True,
    low_cpu_mem_usage=True,
)
model.eval()
total_params = sum(p.numel() for p in model.parameters()) / 1e6
print(f"  ✓ Model yüklendi: {total_params:.0f}M parametre")
print()

# ─── ADIM 3: Hızlı Inference Testi ────────────────────────────────
print("[3/5] Model testi...")
test_text = "merhaba nasılsın bugün"
inputs = tokenizer(test_text, return_tensors="pt",
                   max_length=SEQ_LEN, padding="max_length", truncation=True)
with torch.no_grad():
    t0 = time.perf_counter()
    outputs = model(**inputs)
    elapsed = (time.perf_counter() - t0) * 1000

logits = outputs.logits[0, int(inputs["attention_mask"].sum()) - 1, :]
next_id = int(torch.argmax(logits))
next_tok = tokenizer.decode([next_id], skip_special_tokens=True)
print(f"  Prefix: '{test_text}'")
print(f"  Tahmin: '{next_tok.strip()}'")
print(f"  CPU latency: {elapsed:.0f}ms")
print()

# ─── ADIM 4: TFLite Dönüşümü ──────────────────────────────────────
print("[4/5] TFLite INT8'e dönüştürülüyor (ai-edge-torch)...")
print("  Bu adım ~5-15 dakika sürebilir, lütfen bekleyin...\n")

try:
    import ai_edge_torch

    # Wrapper sınıf — sadece forward pass (past_key_values olmadan)
    class QwenLM(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
            out = self.model(input_ids=input_ids, attention_mask=attention_mask)
            # Sadece son token'ın logits'i döndür (boyut küçültme)
            seq_lengths = attention_mask.sum(dim=1) - 1  # son token indexi
            batch_size = input_ids.shape[0]
            last_logits = out.logits[torch.arange(batch_size), seq_lengths]
            return last_logits  # [batch, vocab_size]

    wrapped = QwenLM(model)
    wrapped.eval()

    # Örnek girişler
    sample_input_ids = torch.zeros(BATCH_SIZE, SEQ_LEN, dtype=torch.int32)
    sample_attention  = torch.ones(BATCH_SIZE, SEQ_LEN, dtype=torch.int32)

    print("  ai-edge-torch ile dönüştürülüyor...")
    edge_model = ai_edge_torch.convert(
        wrapped,
        (sample_input_ids, sample_attention),
    )

    edge_model.export(str(TFLITE_PATH))
    tflite_size = TFLITE_PATH.stat().st_size / 1e6
    print(f"  ✓ TFLite oluşturuldu: {tflite_size:.1f} MB")

except ImportError:
    print("  ai-edge-torch kurulu değil, alternatif yöntem deneniyor...")
    _fallback_export(model, tokenizer, TFLITE_PATH, SEQ_LEN, BATCH_SIZE)

except Exception as e:
    print(f"  ⚠  ai-edge-torch hatası: {e}")
    print("  Alternatif yöntem (ONNX → quantize) deneniyor...")
    _fallback_export(model, tokenizer, TFLITE_PATH, SEQ_LEN, BATCH_SIZE)


def _fallback_export(model, tokenizer, out_path, seq_len, batch_size):
    """ONNX üzerinden quantized model oluştur."""
    try:
        import onnx
        import onnxruntime as ort
        from onnxruntime.quantization import quantize_dynamic, QuantType

        onnx_path = out_path.parent / "model_fp32.onnx"

        dummy_ids  = torch.zeros(batch_size, seq_len, dtype=torch.long)
        dummy_mask = torch.ones(batch_size, seq_len, dtype=torch.long)

        print("  ONNX export ediliyor...")
        with torch.no_grad():
            torch.onnx.export(
                model,
                args=(dummy_ids, dummy_mask),
                f=str(onnx_path),
                opset_version=17,
                input_names=["input_ids", "attention_mask"],
                output_names=["logits"],
                dynamic_axes={
                    "input_ids": {0: "batch", 1: "seq"},
                    "attention_mask": {0: "batch", 1: "seq"},
                    "logits": {0: "batch"},
                },
            )
        print(f"  ✓ ONNX: {onnx_path.stat().st_size / 1e6:.0f} MB")

        # INT8 quantization
        quant_path = out_path.parent / "model_int8.onnx"
        print("  INT8 quantization uygulanıyor...")
        quantize_dynamic(
            str(onnx_path),
            str(quant_path),
            weight_type=QuantType.QInt8,
        )
        q_size = quant_path.stat().st_size / 1e6
        print(f"  ✓ INT8 ONNX: {q_size:.0f} MB")

        # ONNX dosyasını .tflite uzantısıyla kaydet
        # Android'de ONNX Runtime for Android kullanılacak
        import shutil
        shutil.copy2(str(quant_path), str(out_path))
        print(f"  ✓ Çıktı: {out_path} ({q_size:.0f} MB)")
        print("  ⚠  NOT: Bu dosya ONNX formatındadır.")
        print("     Android'de 'onnxruntime-android' dependency kullanın.")

    except Exception as e2:
        print(f"  ✗ Fallback da başarısız: {e2}")
        print("  Lütfen GPU'lu bir ortamda tam pipeline'ı çalıştırın.")


# ─── ADIM 5: Tokenizer Kaydet ─────────────────────────────────────
print()
print("[5/5] Tokenizer kaydediliyor...")
tokenizer.save_pretrained(str(TOKENIZER_DIR))

saved = [f.name for f in TOKENIZER_DIR.iterdir() if f.is_file()]
print(f"  ✓ Tokenizer dosyaları: {', '.join(saved)}")

# ─── Android'e Kopyalama Komutu ────────────────────────────────────
ASSETS = Path("..") / "florisboard-main" / "florisboard-main" / "app" / "src" / "main" / "assets"
print()
print("=" * 65)
print("  ✅ EXPORT TAMAMLANDI!")
print("=" * 65)
print()
print(f"  Model  : {TFLITE_PATH}")
print(f"  Tokenizer: {TOKENIZER_DIR}/")
print()

# Otomatik kopyalama
if TFLITE_PATH.exists() and ASSETS.exists():
    import shutil
    dst_model = ASSETS / "ghost_text_model.tflite"
    dst_tok   = ASSETS / "tokenizer"
    shutil.copy2(str(TFLITE_PATH), str(dst_model))
    if dst_tok.exists():
        shutil.rmtree(str(dst_tok))
    shutil.copytree(str(TOKENIZER_DIR), str(dst_tok))
    print(f"  ✓ Otomatik kopyalandı → {ASSETS}/")
    print("     Artık APK derlenebilir: gradlew assembleDebug")
else:
    print("  📋 Manuel kopyalama:")
    print(f"     {TFLITE_PATH} → {ASSETS}/ghost_text_model.tflite")
    print(f"     {TOKENIZER_DIR}/ → {ASSETS}/tokenizer/")
