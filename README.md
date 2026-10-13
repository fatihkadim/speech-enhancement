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
uv run python -m src.evaluate --checkpoint experiments/baseline/best.pt --split test
uv run python -m src.evaluate --checkpoint experiments/baseline/best.pt --split test_unseen_noise
uv run python -m src.infer --checkpoint experiments/baseline/best.pt --input kayit.wav --output temiz.wav
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
- [ ] **M5** — Deneyler (kayıp, skip, derinlik, çıkış türü, SNR, görülmemiş gürültü, gerçek kayıt)
- [ ] **M6** — Demo notebook ve sonuç tabloları

## Sonuçlar

Değerler `summary.json` çıktılarından alınmıştır (ortalama ± standart sapma, `test` bölmesi, kayıtların tamamı).

> **Ön sonuç:** küçük alt küme (300 eğitim / 50 test örneği), tek tohum. Deneyler (M5) tamamlandıkça güncellenecektir.

| Model / Kayıp | SI-SNR (dB) | PESQ | STOI |
|---------------|-------------|------|------|
| Gürültülü girdi (E0) | 5,07 ± 6,30 | 1,35 ± 0,34 | 0,827 ± 0,126 |
| U-Net + L1 (baseline) | 8,25 ± 5,92 | 1,45 ± 0,37 | 0,820 ± 0,129 |
| U-Net + MSE | – | – | – |
| U-Net + SI-SNR | – | – | – |

## Kaynaklar

- Ronneberger ve ark., *U-Net: Convolutional Networks for Biomedical Image Segmentation*, 2015
- Panayotov ve ark., *LibriSpeech*, 2015
- Valentini-Botinhao ve ark., VoiceBank-DEMAND veri seti
- Piczak, *ESC-50*, 2015; Salamon ve ark., *UrbanSound8K*, 2014
- Le Roux ve ark., *SDR – Half-baked or Well Done?*, 2019
- Rix ve ark., PESQ, 2001; Taal ve ark., STOI, 2011
