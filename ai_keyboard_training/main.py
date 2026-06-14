"""
=====================================================================
main.py — Ana Pipeline Orkestratörü
Solenz AI Keyboard Training Pipeline
=====================================================================
Tüm adımları tek bir CLI aracından yönetir.

Kullanım:
    # Tüm pipeline (veri → eğitim → export)
    python main.py --mode full

    # Sadece veri hazırlama
    python main.py --mode data

    # Sadece fine-tuning (veri hazırlama yapılmış varsayılır)
    python main.py --mode train

    # Sadece TFLite export (eğitim tamamlanmış varsayılır)
    python main.py --mode export

    # Sadece model doğrulama
    python main.py --mode validate --tflite-path outputs/tflite/ghost_text_model.tflite

Gelişmiş kullanım:
    # Özel model ile
    python main.py --mode full --base-model Qwen/Qwen2-0.5B

    # Özel epoch sayısı ile
    python main.py --mode train --epochs 5 --lr 1e-4

    # Farklı quantization modu
    python main.py --mode export --quant-mode float16
"""

import sys
import time
from enum import Enum
from pathlib import Path
from typing import Optional

import typer
from loguru import logger
from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule

# Proje kök dizinini Python path'ine ekle
sys.path.insert(0, str(Path(__file__).parent))

from src import (
    PipelineConfig,
    ModelConfig,
    LoRAConfig,
    TrainingConfig,
    DataConfig,
    ExportConfig,
    DataPipeline,
    KeyboardModelTrainer,
    QuantizationPipeline,
    TFLiteValidator,
)

# ─────────────────────────────────────────────────────────────────────
# Loglama konfigürasyonu
# ─────────────────────────────────────────────────────────────────────
logger.remove()  # Varsayılan handler'ı kaldır
logger.add(
    sys.stdout,
    format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{message}</cyan>",
    level="INFO",
    colorize=True,
)
logger.add(
    "outputs/training.log",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
    level="DEBUG",
    rotation="10 MB",
)

console = Console()
app = typer.Typer(
    name="solenz-keyboard-trainer",
    help="🤖 Solenz AI Keyboard — Model Training Pipeline",
    add_completion=False,
)


# ─────────────────────────────────────────────────────────────────────
# Pipeline Mode Enum
# ─────────────────────────────────────────────────────────────────────
class PipelineMode(str, Enum):
    FULL = "full"       # Tüm pipeline
    DATA = "data"       # Sadece veri hazırlama
    TRAIN = "train"     # Sadece eğitim
    EXPORT = "export"   # Sadece TFLite export
    VALIDATE = "validate" # Sadece doğrulama


# ─────────────────────────────────────────────────────────────────────
# Yardımcı Fonksiyonlar
# ─────────────────────────────────────────────────────────────────────
def print_banner():
    """ASCII banner yazdır."""
    console.print(Panel(
        """[bold cyan]
  ███████╗ ██████╗ ██╗     ███████╗███╗   ██╗███████╗
  ██╔════╝██╔═══██╗██║     ██╔════╝████╗  ██║╚══███╔╝
  ███████╗██║   ██║██║     █████╗  ██╔██╗ ██║  ███╔╝
  ╚════██║██║   ██║██║     ██╔══╝  ██║╚██╗██║ ███╔╝
  ███████║╚██████╔╝███████╗███████╗██║ ╚████║███████╗
  ╚══════╝ ╚═════╝ ╚══════╝╚══════╝╚═╝  ╚═══╝╚══════╝
  
  [white]AI Keyboard Training Pipeline[/white]
  [dim]Ghost Text Model — Qwen2-0.5B + LoRA + INT8 TFLite[/dim][/bold cyan]""",
        border_style="cyan",
    ))


def build_config(
    base_model: Optional[str],
    epochs: int,
    lr: float,
    batch_size: int,
    lora_rank: int,
    quant_mode: str,
    seq_length: int,
) -> PipelineConfig:
    """CLI parametrelerinden PipelineConfig oluştur."""
    model_cfg = ModelConfig(
        base_model_id=base_model or "Qwen/Qwen2-0.5B-Instruct",
        max_seq_length=seq_length,
    )
    lora_cfg = LoRAConfig(
        r=lora_rank,
        lora_alpha=lora_rank * 2,
    )
    training_cfg = TrainingConfig(
        num_train_epochs=epochs,
        learning_rate=lr,
        per_device_train_batch_size=batch_size,
    )
    export_cfg = ExportConfig(
        quantization_mode=quant_mode,
        export_seq_length=seq_length,
    )
    return PipelineConfig(
        model=model_cfg,
        lora=lora_cfg,
        training=training_cfg,
        data=DataConfig(),
        export=export_cfg,
    )


# ─────────────────────────────────────────────────────────────────────
# Ana CLI Komutu
# ─────────────────────────────────────────────────────────────────────
@app.command()
def main(
    mode: PipelineMode = typer.Option(
        PipelineMode.FULL,
        "--mode", "-m",
        help="Çalıştırılacak pipeline modu",
    ),
    base_model: Optional[str] = typer.Option(
        None,
        "--base-model",
        help="Hugging Face model ID (varsayılan: Qwen/Qwen2-0.5B-Instruct)",
    ),
    epochs: int = typer.Option(3, "--epochs", "-e", help="Eğitim epoch sayısı"),
    lr: float = typer.Option(2e-4, "--lr", help="Öğrenme hızı"),
    batch_size: int = typer.Option(4, "--batch-size", "-b", help="Batch size"),
    lora_rank: int = typer.Option(16, "--lora-rank", help="LoRA rank değeri"),
    quant_mode: str = typer.Option("int8", "--quant-mode", help="Quantization modu: int8|float16|dynamic"),
    seq_length: int = typer.Option(128, "--seq-length", help="Maksimum sequence uzunluğu"),
    tflite_path: Optional[str] = typer.Option(
        None,
        "--tflite-path",
        help="Sadece validate modu için TFLite dosya yolu",
    ),
    merged_model_dir: Optional[str] = typer.Option(
        None,
        "--merged-model-dir",
        help="Sadece export modu için birleştirilmiş model dizini",
    ),
):
    """🤖 Solenz AI Keyboard Model Training Pipeline."""
    print_banner()

    start_time = time.time()
    config = build_config(base_model, epochs, lr, batch_size, lora_rank, quant_mode, seq_length)

    try:
        # ─── DATA MODE ───────────────────────────────────────────
        if mode in [PipelineMode.DATA, PipelineMode.FULL]:
            console.print(Rule("[bold cyan]ADIM 1: Veri Hazırlama[/bold cyan]", style="cyan"))
            data_pipeline = DataPipeline(config)
            dataset = data_pipeline.run()

            if mode == PipelineMode.DATA:
                console.print("[green]✅ Veri hazırlama tamamlandı.[/green]")
                return

        # ─── TRAIN MODE ──────────────────────────────────────────
        if mode in [PipelineMode.TRAIN, PipelineMode.FULL]:
            console.print(Rule("[bold cyan]ADIM 2: LoRA Fine-Tuning[/bold cyan]", style="cyan"))

            if mode == PipelineMode.TRAIN:
                # Eğer sadece train modundaysak veriyi tekrar hazırla
                data_pipeline = DataPipeline(config)
                dataset = data_pipeline.run()

            trainer = KeyboardModelTrainer(config)
            merged_model_dir = trainer.train(dataset)

            if mode == PipelineMode.TRAIN:
                console.print(f"[green]✅ Eğitim tamamlandı. Model: {merged_model_dir}[/green]")
                return

        # ─── EXPORT MODE ─────────────────────────────────────────
        if mode in [PipelineMode.EXPORT, PipelineMode.FULL]:
            console.print(Rule("[bold cyan]ADIM 3: TFLite Export[/bold cyan]", style="cyan"))

            if mode == PipelineMode.EXPORT:
                if not merged_model_dir:
                    # Varsayılan merged model dizinini bul
                    merged_model_dir = str(Path("outputs") / "merged_model")
                    if not Path(merged_model_dir).exists():
                        console.print("[red]❌ Hata: Birleştirilmiş model bulunamadı.[/red]")
                        console.print("[yellow]Önce 'python main.py --mode train' çalıştırın.[/yellow]")
                        raise typer.Exit(1)

            export_pipeline = QuantizationPipeline(config)
            tflite_path = export_pipeline.run(merged_model_dir)

            if mode == PipelineMode.EXPORT:
                console.print(f"[green]✅ Export tamamlandı. TFLite: {tflite_path}[/green]")
                return

        # ─── VALIDATE MODE ───────────────────────────────────────
        if mode in [PipelineMode.VALIDATE, PipelineMode.FULL]:
            console.print(Rule("[bold cyan]ADIM 4: Model Doğrulama[/bold cyan]", style="cyan"))

            if mode == PipelineMode.VALIDATE:
                if not tflite_path:
                    tflite_path = str(Path("outputs/tflite/ghost_text_model.tflite"))
                if not Path(tflite_path).exists():
                    console.print(f"[red]❌ TFLite dosyası bulunamadı: {tflite_path}[/red]")
                    raise typer.Exit(1)

            tokenizer_dir = str(Path(config.export.tokenizer_output_dir))
            if not Path(tokenizer_dir).exists():
                tokenizer_dir = merged_model_dir  # Fallback

            validator = TFLiteValidator(tflite_path, tokenizer_dir)
            validator.run_full_validation()

        # ─── SÜRE RAPORU ─────────────────────────────────────────
        elapsed = time.time() - start_time
        console.print(Panel(
            f"[bold green]🏁 Pipeline Tamamlandı![/bold green]\n"
            f"Toplam süre: [yellow]{elapsed / 60:.1f} dakika[/yellow]",
            border_style="green",
        ))

    except KeyboardInterrupt:
        console.print("\n[yellow]⚠️  Kullanıcı tarafından iptal edildi.[/yellow]")
        raise typer.Exit(0)
    except Exception as e:
        logger.exception(f"Pipeline hatası: {e}")
        console.print(f"\n[red]❌ Hata: {e}[/red]")
        raise typer.Exit(1)


# ─────────────────────────────────────────────────────────────────────
# Entry Point
# ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app()
