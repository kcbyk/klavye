"""
=====================================================================
sample_data_generator.py — Türkçe Örnek Veri Üretici
Solenz AI Keyboard Training Pipeline
=====================================================================
Gerçek kullanıcı verisi yokken eğitim için sentetik
Türkçe mesajlaşma verisi üretir.

Bu script ayrı çalıştırılabilir:
    python data/sample_data_generator.py --count 5000

Gerçek projede bu scriptin çıktısını aşağıdaki kaynaklarla
zenginleştirin:
  - WhatsApp/Telegram mesaj exportları
  - Türkçe Twitter/sosyal medya arşivleri
  - Açık kaynak Türkçe NLP veri setleri:
      * TS Corpus: https://tscorpus.com/
      * OSCAR (Türkçe alt kümesi)
      * CC-100 Türkçe
"""

import json
import random
import re
from pathlib import Path

import typer
from rich.console import Console
from rich.progress import track

console = Console()
app = typer.Typer()

# ─────────────────────────────────────────────────────────────────────
# Veri Şablonları
# ─────────────────────────────────────────────────────────────────────

# Temel kelime havuzları
SUBJECTS = [
    "ben", "sen", "o", "biz", "siz", "onlar",
    "ali", "ayşe", "mehmet", "fatma", "ahmet",
    "arkadaş", "hocam", "abla", "abi", "kardeş",
]

GREETINGS = [
    "merhaba nasılsın",
    "selam ne haber",
    "iyi günler",
    "günaydın nasıl geçiyor",
    "tünaydın iyi misin",
    "iyi akşamlar",
    "naber canım",
    "hey nasılsın be",
    "selamlar her şey yolunda mı",
]

FAREWELLS = [
    "görüşürüz yarın",
    "tamam bay bay",
    "iyi geceler",
    "iyi günler tekrar",
    "görüşmek üzere",
    "kendine iyi bak",
    "yarın konuşuruz",
    "hoşça kal",
]

RESPONSES = [
    "iyiyim teşekkürler sen nasılsın",
    "çok iyiyim sağ ol",
    "idare ediyorum ya",
    "eh işte fena değil",
    "süper teşekkürler",
    "biraz yorgunum ama iyiyim",
    "harika geçiyor teşekkürler",
    "normal işte değişen bir şey yok",
]

WORK_MESSAGES = [
    "toplantı saat kaçta başlıyor",
    "raporları gönderebilir misin bana",
    "dosyayı paylaştım kontrol et",
    "proje ne zaman bitiyor",
    "sunum hazır mı yarın için",
    "müşteri toplantısı ertelendi mi",
    "ofise kaç dakikada gelirsin",
    "mail attım kontrol et lütfen",
    "bütçe onaylandı mı henüz",
    "ekiple toplantı yapalım bu hafta",
    "deadline ne zamandı hatırlatır mısın",
    "kod review yapabilir misin bugün",
    "sunucu çöktü ne yapacağız",
    "teklif maili attın mı müşteriye",
    "sözleşme imzalandı mı biliyor musun",
    "fazla mesai yapacak mıyız bu gece",
    "yeni proje için fikrin var mı",
    "tasarım onaylandı mı henüz",
    "test sonuçları nasıl geldi",
    "hangi framework kullanalım sence",
]

SOCIAL_MESSAGES = [
    "akşam yemeğe gidiyoruz gel",
    "sinemada ne oynuyor bugün",
    "hafta sonu boş musun bir şeyler yapalım",
    "çok yoruldum bugün keşke izin olsaydı",
    "harika bir gün geçirdim bugün",
    "aile ziyareti var bu hafta sonu",
    "doğum günün kutlu olsun canım",
    "çok güzel bir sürpriz yaptin",
    "bir dahaki sefere sen seç nereye gidelim",
    "nereye gidelim bu akşam söyle",
    "konsere gidiyoruz sen de gel",
    "piknik yapalım mı pazar günü",
    "spor salonuna gidiyor musun artık",
    "yeni restoran açılmış deneyelim mi",
    "tatil planın var mı bu yaz için",
    "köye gidiyoruz bir haftalığına",
    "kahve içelim mi öğle arası",
    "film izliyor musun bu aralar ne var",
    "kitap önerir misin bana güzel bir şey",
    "konser bileti aldım gelir misin",
]

SHOPPING_MESSAGES = [
    "marketten ekmek al dönerken",
    "fatura ödendi mi bu ay",
    "araba serviste ne zaman çıkıyor",
    "eczaneden ilaç aldım sana",
    "paket geldi imzaladım tamam",
    "bu ay çok para harcadık dikkat et",
    "indirim var yararlan bugün son gün",
    "fiyat çok pahalı başka bak",
    "sipariş verdim yarın gelecek",
    "geri iade ettim beğenmedim",
    "kredi kartı limitim doldu",
    "kira bu ay gecikti affedersin",
    "market listesine ekle şekeri",
    "online alışveriş yaptım tavsiye ederim",
    "ikinci el bak bulursun daha ucuz",
]

EMOTIONAL_MESSAGES = [
    "çok mutlu oldum haberini duyunca",
    "üzgünüm başına gelenlere iyi olmasını dilerim",
    "geçmiş olsun hızlı iyi ol",
    "bravo başardın tebrik ederim seni",
    "endişelenme hallederiz bir yolunu",
    "sabırlı ol her şey düzelecek merak etme",
    "sağ ol gerçekten teşekkür ederim",
    "ne kadar düşünceli bir hareket bu",
    "çok sinirlisin neden bu kadar",
    "sakin ol gergin olmanın faydası yok",
    "güle güle git iyi yolculuklar",
    "özledim seni ne zaman görüşeceğiz",
    "gülünç değil mi bu durum",
    "harika bir haber bu kutluyorum seni",
    "merak etme yanındayım her zaman",
]

QUESTION_MESSAGES = [
    "geliyor musun bize bu akşam",
    "hazır olur musun saat beşte",
    "anladın mı söylediklerimi",
    "emin misin bu konuda gerçekten",
    "düşündün mü bir kez daha",
    "ne zaman uygun olursun acaba",
    "kaç kişi gelecek söyledin mi",
    "nerede buluşalım sence en iyi",
    "hangi gün daha uygun senin için",
    "bunu nasıl yaptın anlat bakalım",
    "neden bu kararı aldın açıklar mısın",
    "kim söyledi bunu sana",
    "ne zaman öğrendin bu haberi",
    "nasıl hissediyorsun şu an",
    "yardım ister misin bu konuda",
]

# Bağlaçlar ve doldurucular
CONNECTORS = [
    "ama", "ve", "veya", "ya da", "fakat", "çünkü",
    "tabii ki", "kesinlikle", "belki", "sanırım",
    "aslında", "zaten", "hep", "hiç", "artık",
]


def generate_varied_sentence() -> str:
    """Rastgele çeşitli bir Türkçe cümle üret."""
    categories = [
        GREETINGS, FAREWELLS, RESPONSES, WORK_MESSAGES,
        SOCIAL_MESSAGES, SHOPPING_MESSAGES, EMOTIONAL_MESSAGES,
        QUESTION_MESSAGES,
    ]

    # Temel cümleyi seç
    category = random.choice(categories)
    sentence = random.choice(category)

    # Bazen iki cümleyi birleştir (daha zengin örnekler için)
    if random.random() < 0.3:
        connector = random.choice(CONNECTORS)
        second_category = random.choice(categories)
        second = random.choice(second_category)
        sentence = f"{sentence} {connector} {second}"

    return sentence


def generate_dataset(count: int, output_path: Path) -> list[dict]:
    """Belirtilen sayıda veri noktası üret."""
    messages = []
    base_messages = (
        GREETINGS + FAREWELLS + RESPONSES + WORK_MESSAGES +
        SOCIAL_MESSAGES + SHOPPING_MESSAGES + EMOTIONAL_MESSAGES +
        QUESTION_MESSAGES
    )

    # Önce gerçek örnekleri ekle
    for msg in base_messages:
        messages.append({"text": msg, "source": "template"})

    # Sonra üretilmiş örnekleri ekle
    target_remaining = count - len(messages)
    for _ in track(range(max(0, target_remaining)), description="Veri üretiliyor..."):
        messages.append({
            "text": generate_varied_sentence(),
            "source": "generated",
        })

    # Karıştır
    random.shuffle(messages)
    messages = messages[:count]  # Tam count'u sağla

    # Kaydet
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(messages, f, ensure_ascii=False, indent=2)

    console.print(f"✅ [green]{len(messages):,}[/green] mesaj üretildi: [cyan]{output_path}[/cyan]")
    return messages


@app.command()
def generate(
    count: int = typer.Option(1000, "--count", "-n", help="Üretilecek mesaj sayısı"),
    output: str = typer.Option(
        "data/raw/messages_raw.json",
        "--output", "-o",
        help="Çıktı dosyası yolu",
    ),
    seed: int = typer.Option(42, "--seed", help="Rastgele tohum"),
):
    """Türkçe örnek mesaj veri seti üret."""
    random.seed(seed)
    output_path = Path(output)
    generate_dataset(count, output_path)


if __name__ == "__main__":
    app()
