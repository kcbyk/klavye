# -*- coding: utf-8 -*-
"""
=============================================================
01_generate_dataset.py
Solenz AI Keyboard — Türkçe Veri Seti Üretici
=============================================================
Günlük Türkçe mesajlaşma kalıplarından oluşan fine-tuning
veri seti oluşturur.

Format: HuggingFace datasets uyumlu JSON Lines
Çıktı : data/turkish_chat_train.jsonl
         data/turkish_chat_eval.jsonl

Çalıştırma: python 01_generate_dataset.py
"""

import json
import random
import os
from pathlib import Path

random.seed(42)
os.makedirs("data", exist_ok=True)

# ─── TÜRKÇE VERİ KAYNAKLARI ────────────────────────────────────────

GUNLUK_MESAJLAR = [
    # Selamlaşma
    "merhaba nasılsın", "selam ne var ne yok", "hey iyi misin",
    "günaydın nasıl geçti", "iyi günler kolay gelsin", "merhaba günaydın",
    "selam canım nasılsın", "nbr ne haber", "naber iyimisin",
    # Günlük konuşma
    "bugün çok yoruldum", "yarın buluşalım mı", "akşam ne yapıyorsun",
    "saat kaçta geliyorsun", "neredesin şu an", "az sonra geliyorum",
    "tamam görüşürüz", "olur yarın konuşuruz", "iyi geceler",
    "hadi kolay gelsin", "görüşmek üzere", "kendine iyi bak",
    # Duygular
    "çok mutluyum bugün", "biraz stresli hissediyorum", "harika haber",
    "üzgünüm özür dilerim", "seni özledim", "çok teşekkür ederim",
    "harika bir gün geçirdim", "yorgun ama mutluyum",
    # Planlama
    "yarın sinemaya gidelim mi", "haftasonu ne yapıyorsun",
    "akşam yemek yiyelim mi", "buluşma saatini değiştirelim mi",
    "toplantı saat 3te", "öğle arası çıkıyorum",
    # İş / Okul
    "proje teslimi yarın", "sunum nasıl gitti", "sınav çok zordu",
    "bugün toplantı var mı", "raporu gönderdim mi kontrolledin",
    "hafta içi müsait misin", "online mı yoksa yüz yüze mi",
    # Yemek / Eğlence
    "akşam pizza söyleyelim mi", "hangi restoranda buluşalım",
    "film başlıyor hazır mısın", "konser biletleri aldım",
    "bu hafta sonu piknik var", "çay içmeye gelelim mi",
    # Alışveriş
    "markete gidiyorum bir şey lazım mı", "indirim var bugün",
    "o ürünü aldın mı sonunda", "fiyat çok pahalı değil mi",
    # Sağlık
    "doktora gitmem lazım", "ilaçlarını aldın mı",
    "bugün biraz hasta hissettim", "geçmiş olsun umarım iyileşirsin",
    # Teknoloji
    "telefonu güncelle bak", "internet çekmiyor burada",
    "uygulamayı indirdin mi", "şifre yanlış tekrar dene",
    # Aile / Arkadaş
    "annem seni soruyor", "kardeşim de geliyor", "aile toplantısı var",
    "arkadaşlar buluşuyor katılıyor musun", "doğum günün kutlu olsun",
]

TAMAMLAMA_KALIPLARI = [
    # Fiil tamamlamaları
    ("merhaba nasıl", "sın bugün"),
    ("selam ne", " var ne yok"),
    ("yarın", " buluşalım mı"),
    ("bugün akşam", " çıkıyor musun"),
    ("tamam o zaman", " görüşürüz"),
    ("çok teşekkür", " ederim yardımın için"),
    ("özür dile", "rim geç kaldım için"),
    ("harika bir", " gün geçirdim"),
    ("toplantı saat", " kaçta başlıyor"),
    ("seni çok", " özledim"),
    ("iyi geceler", " tatlı rüyalar"),
    ("günaydın nasıl", "sın bugün"),
    ("neredesin şu", " an gelebilir misin"),
    ("az sonra", " geliyorum bekle beni"),
    ("proje teslimi", " yarın gece yarısı"),
    ("sınav nasıl", " gitti umarım iyidir"),
    ("film saat", " kaçta başlıyor"),
    ("akşam yemeği", "ni birlikte yiyelim mi"),
    ("haftasonu", " boş musun buluşalım"),
    ("ilaçlarını", " aldın mı bugün"),
    ("internet", " çekmiyor burada sinyal yok"),
    ("doğum günün", " kutlu olsun canım"),
    ("çok yorgun", "um ama bitirdim"),
    ("stresli", " hissediyorum biraz"),
    ("market", "e gidiyorum ne lazım"),
]

SOHBET_TUMLECLER = [
    # Kısa doğal cümleler (5-15 kelime)
    "Bugün hava çok güzel yürüyüşe çıkalım mı?",
    "Toplantı ertelendi saat 5e aldık.",
    "Çok yoruldum ama iş bitti nihayet.",
    "Yarın saat 10da buluşalım tamam mı?",
    "Özür dilerim geç kaldım yolda trafik vardı.",
    "Harika haber tebrikler çok mutlu oldum.",
    "Seni aradım ama açmadın ne oldu?",
    "Raporu gönderdim kontrol edebilir misin?",
    "Akşam ne yemek yesek pizza mı?",
    "Film biletlerini aldım saat 8de başlıyor.",
    "Kardeşim de geliyor seni görmek istiyor.",
    "İndirim var bugün alışverişe çıkalım mı?",
    "Tamam yarın devam ederiz iyi geceler.",
    "Çok teşekkür ederim yardımın için.",
    "Geçmiş olsun çabuk iyileşirsin umarım.",
    "Telefonu şarj etmeyi unutma bitecek.",
    "Sınav çok zordu ama bence iyi geçti.",
    "Haftasonu piknik var gelebilir misin?",
    "Annem seni soruyor yakında uğra.",
    "Online mı yoksa ofiste mi buluşuyoruz?",
    "Bana bir kahve alabilir misin lütfen?",
    "Proje son aşamada bitiyor çok şükür.",
    "Arkadaşlar buluşuyor sen de katıl.",
    "Doktor randevusu aldım cuma günü.",
    "Uygulama güncellendi şimdi daha iyi.",
    "Şifreni unuttuysan sıfırla tekrar gir.",
    "Markete gidiyorum bir şey lazım mı?",
    "Konser çok iyiydi keşke sen de gelseydin.",
    "Aile toplantısı var cumartesi unutma.",
    "Stresli bir gün geçirdim dinlenmem lazım.",
    "Seni özledim uzun zamandır görüşemedik.",
    "Harika iş çıkardın gerçekten çok iyiydi.",
    "Biraz hasta hissediyorum bugün evdeyim.",
    "Ödev var mı yarın teslim mi?",
    "Restoran rezervasyonu yaptım akşama.",
    "Çay içmeye gelir misin bir ara?",
    "İndirimde aldım çok ucuzdu inanılmaz.",
    "Trafikte kaldım 20 dakika geç kalacağım.",
    "Bugün müsait değilim yarın konuşalım.",
    "Çok güzel olmuş ellerine sağlık.",
    "Sabah koşusu yaptım çok iyi hissettim.",
    "Hava soğudu mont giy üşüme.",
    "Masraf çok oldu ama değdi.",
    "Güzel bir gün geçirdik tekrar edelim.",
    "Sana önemli bir şey söyleyecektim.",
    "Yarın erken kalkman gerekiyor dikkat et.",
    "Müzik kulağıma küpe oldu o şarkı.",
    "Düşün bakalım karar verelim.",
    "Nasılsın uzun zamandır görüşemedik?",
    "Çok iyiyim teşekkür ederim.",
]

NEXT_TOKEN_PAIRS = []
for prefix, suffix in TAMAMLAMA_KALIPLARI:
    words = suffix.strip().split()
    if words:
        next_word = words[0]
        NEXT_TOKEN_PAIRS.append((prefix.strip(), next_word, suffix.strip()))


def make_chat_sample(text: str) -> dict:
    """Metin tamamlama formatında örnek oluştur."""
    words = text.split()
    if len(words) < 3:
        return None
    cut = random.randint(1, max(1, len(words) - 1))
    prefix = " ".join(words[:cut])
    completion = " ".join(words[cut:])
    return {
        "prompt": prefix,
        "completion": completion,
        "full_text": text,
    }


def make_instruct_sample(text: str) -> dict:
    """Instruction-following formatında örnek oluştur."""
    return {
        "instruction": "Türkçe mesajı tamamla:",
        "input": text[:len(text)//2 + random.randint(0,5)],
        "output": text,
    }


def make_lm_sample(text: str) -> dict:
    """Dil modeli formatında örnek (basit next-token)."""
    return {"text": text}


# ─── VERİ SETİ ÜRETİMİ ────────────────────────────────────────────

print("Türkçe veri seti üretiliyor...")

all_samples = []

# 1. Günlük mesaj tamamlamaları
for msg in GUNLUK_MESAJLAR:
    for _ in range(3):
        s = make_lm_sample(msg)
        all_samples.append(s)
        s2 = make_chat_sample(msg)
        if s2:
            all_samples.append(s2)

# 2. Tamamlama kalıpları
for prefix, suffix in TAMAMLAMA_KALIPLARI:
    full = prefix + suffix
    all_samples.append(make_lm_sample(full))
    all_samples.append({"prompt": prefix, "completion": suffix, "full_text": full})

# 3. Uzun sohbet cümleleri
for sent in SOHBET_TUMLECLER:
    for _ in range(4):
        s = make_lm_sample(sent)
        all_samples.append(s)
        s2 = make_chat_sample(sent)
        if s2:
            all_samples.append(s2)
        s3 = make_instruct_sample(sent)
        all_samples.append(s3)

# 4. Zincirleme sohbet (context-aware)
for i in range(100):
    turn1 = random.choice(SOHBET_TUMLECLER)
    turn2 = random.choice(SOHBET_TUMLECLER)
    combined = f"{turn1} {turn2}"
    all_samples.append(make_lm_sample(combined))

random.shuffle(all_samples)

# Train / Eval bölme (%90 / %10)
split = int(len(all_samples) * 0.9)
train_data = all_samples[:split]
eval_data = all_samples[split:]

def write_jsonl(data: list, path: str):
    with open(path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

write_jsonl(train_data, "data/turkish_chat_train.jsonl")
write_jsonl(eval_data,  "data/turkish_chat_eval.jsonl")

print(f"  Train: {len(train_data)} ornek")
print(f"  Eval : {len(eval_data)} ornek")
print(f"  Toplam: {len(all_samples)} ornek")
print(f"  Dosyalar: data/turkish_chat_train.jsonl, data/turkish_chat_eval.jsonl")
print("Veri seti hazir!")
