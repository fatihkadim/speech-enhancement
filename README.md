# SesTemiz — Derin Öğrenme ile Konuşma Gürültü Giderme

Gürültülü bir konuşma kaydından temiz konuşmayı kestiren U-Net tabanlı bir model.
Yapay Sinir Ağları dersi dönem projesi.

**Ekip:** Fatih Kadim · Mehmet Göktuğ Elbir · Mehmet Burak Albayrak

> Projenin ayrıntılı teknik şartnamesi (veri sözleşmesi, modül arayüzleri, deney matrisi, kurallar)
> [`docs/PROJECT_SPEC.md`](docs/PROJECT_SPEC.md) dosyasındadır.

## Nasıl Çalışıyor?

Gürültülü sinyal `x = s + n` verildiğinde temiz konuşma `s` kestirilir:

1. Gürültülü ses STFT ile spektrograma çevrilir (16 kHz, `n_fft=512`, `hop=128`).
2. U-Net, `log1p(|X|)` genlik spektrogramından bir **maske** (veya doğrudan temiz genlik) tahmin eder.
3. Tahmin edilen genlik, gürültülü sinyalin fazıyla birleştirilip iSTFT ile tekrar sese dönüştürülür.

Eğitim verisi, temiz konuşma (LibriSpeech / VoiceBank-DEMAND) ile gürültü kayıtlarının
(ESC-50 / UrbanSound8K) −5…15 dB SNR aralığında karıştırılmasıyla üretilir.
Bölmeler konuşmacıya göre yapılır; bazı gürültü sınıfları yalnızca test için ayrılır.

## Araştırma Soruları

1. Hangi kayıp fonksiyonu (L1, MSE, SI-SNR) hem metriklerde hem kulakla daha iyi?
2. U-Net'te derinlik ve skip connection'ların etkisi nedir?
3. Model eğitimde görmediği gürültü türlerine genelleşiyor mu?
4. Düşük SNR'da performans nasıl değişiyor?

## Kurulum

Proje [uv](https://docs.astral.sh/uv/) ile yönetilir (Python 3.10+). Windows ve Linux'ta torch, CUDA 12.8 sürümüyle kurulur.

```bash
uv sync --extra pesq
uv run pytest
```

PESQ metriği opsiyoneldir ve kurulum sırasında derlenir (C/C++ derleyicisi gerekir). Derlenemezse `uv sync`
ile PESQ'siz kurulabilir; değerlendirme bu durumda PESQ'i atlayıp uyarı verir. Not: `--extra pesq` olmadan
çalıştırılan `uv sync` kurulu PESQ'i kaldırır.

Colab gibi pip kullanılan ortamlarda: `pip install -r requirements.txt pesq`

## Kullanım

Veri setleri `data/raw/` altına açılmalıdır:
[LibriSpeech dev-clean](https://www.openslr.org/resources/12/dev-clean.tar.gz) → `data/raw/LibriSpeech/`,
[ESC-50](https://github.com/karolpiczak/ESC-50/archive/refs/heads/master.zip) → `data/raw/ESC-50-master/`.

```bash
uv run python -m src.data.make_dataset --config configs/baseline.yaml     # veri üretimi + manifest + örnekler
uv run python -m src.evaluate --config configs/baseline.yaml --split test # E0: işlenmemiş girdinin metrikleri
uv run python -m src.train --config configs/baseline.yaml                 # eğitim
uv run python -m src.evaluate --checkpoint experiments/baseline/seed42/best.pt --split test
uv run python -m src.evaluate --checkpoint experiments/baseline/seed42/best.pt --split test_unseen_noise
uv run python -m src.infer --checkpoint experiments/baseline/seed42/best.pt --input kayit.wav --output temiz.wav

# M5 deney matrisi (21 eğitim, ~9 saat) ve sonuç tabloları
uv run python -m src.run_experiments --base configs/baseline.yaml --matrix configs/experiments.yaml
uv run python -m src.collect_results --matrix configs/experiments.yaml --markdown docs/results.md
```

Tüm hiperparametreler `configs/` altındaki YAML dosyalarından yönetilir.

## Klasör Yapısı

```
configs/        Deney config'leri (YAML)
data/           Ham ve işlenmiş veri (git'e eklenmez)
docs/           Proje şartnamesi
src/            Kaynak kod (veri, model, kayıplar, eğitim, değerlendirme)
tests/          Birim testleri
notebooks/      Demo notebook
experiments/    Eğitim çıktıları, checkpoint'ler, loglar (git'e eklenmez)
samples/        Örnek ses ve spektrogram görselleri
```

## Yol Haritası

- [x] **M0** — Ortam, klasör iskeleti, test altyapısı
- [x] **M1** — Ses yardımcıları (STFT/iSTFT, SNR karışımı)
- [x] **M2** — Veri hattı ve sızıntı kontrolü
- [x] **M3** — U-Net ve kayıp fonksiyonları
- [x] **M4** — Baseline eğitim ve değerlendirme
- [~] **M5** — Deneyler (kayıp, skip, derinlik, çıkış türü, SNR, görülmemiş gürültü, gerçek kayıt)
- [ ] **M6** — Demo notebook ve sonuç tabloları

## Sonuçlar

LibriSpeech dev-clean (2072 eğitim / 340 test kaydı) + ESC-50, `test` bölmesi, kayıtların tamamı.
Her deney 3 tohumla eğitildi; değerler tohumlar arası ortalama ± std. Δ: aynı test kaydı ve tohum eşleştirilerek
baseline'a fark, köşeli parantez içinde %95 güven aralığı. Tüm tablolar (görülmemiş gürültü, SNR'a göre):
[`docs/results.md`](docs/results.md).

| Deney | Değişen ayar | SI-SNR (dB) | PESQ | STOI | Δ SI-SNR | Δ PESQ |
|-------|--------------|-------------|------|------|----------|--------|
| Gürültülü girdi (E0) | – | 4,50 | 1,367 | 0,833 | | |
| **Baseline** | U-Net d=4, skip, maske, L1 | 12,78 ± 0,27 | 1,970 ± 0,042 | 0,891 ± 0,004 | | |
| E1 | kayıp: MSE | 12,55 ± 0,59 | 1,945 ± 0,033 | 0,895 ± 0,005 | −0,23 [−0,38, −0,08] | −0,025 [−0,046, −0,004] |
| E1 | kayıp: SI-SNR | **13,49 ± 0,20** | 1,942 ± 0,007 | 0,892 ± 0,002 | **+0,71** [+0,50, +0,91] | −0,028 [−0,049, −0,010] |
| E2 | skip yok | 11,60 ± 0,26 | 1,802 ± 0,031 | 0,874 ± 0,004 | −1,18 [−1,34, −1,01] | −0,168 [−0,185, −0,150] |
| E3 | derinlik 3 | 12,50 ± 0,00 | 1,917 ± 0,015 | 0,886 ± 0,001 | −0,28 [−0,44, −0,12] | −0,054 [−0,068, −0,040] |
| E3 | derinlik 5 | 13,14 ± 0,14 | **2,043 ± 0,035** | **0,897 ± 0,002** | +0,36 [+0,24, +0,48] | **+0,072** [+0,059, +0,085] |
| E4 | çıkış: doğrudan genlik | 12,50 ± 0,26 | 1,947 ± 0,037 | 0,891 ± 0,003 | −0,28 [−0,39, −0,17] | −0,024 [−0,033, −0,015] |

Notlar: 21 eğitimin 17'si 50 epoch sınırına ulaştı (doğrulama kaybı hâlâ hafifçe iyileşiyordu); tüm deneyler
aynı sınırla eğitildiği için karşılaştırma adildir. Faz gürültülü sinyalden alınır (bkz. şartname). İlk pilot deney
(450 örnek, tek tohum) `experiments/pilot_450/` altında saklanmıştır.

## Kaynaklar

- Ronneberger ve ark., *U-Net: Convolutional Networks for Biomedical Image Segmentation*, 2015
- Panayotov ve ark., *LibriSpeech*, 2015
- Valentini-Botinhao ve ark., VoiceBank-DEMAND veri seti
- Piczak, *ESC-50*, 2015; Salamon ve ark., *UrbanSound8K*, 2014
- Le Roux ve ark., *SDR – Half-baked or Well Done?*, 2019
- Rix ve ark., PESQ, 2001; Taal ve ark., STOI, 2011
