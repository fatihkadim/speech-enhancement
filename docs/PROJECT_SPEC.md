# Derin Öğrenme Tabanlı Konuşma Gürültü Giderme (Speech Enhancement)

Yapay Sinir Ağları dersi dönem projesi. 3 kişilik ekip: Fatih Kadim, Mehmet Göktuğ Elbir, Mehmet Burak Albayrak.

> **Bu belge öncelikle bir kodlama ajanı için yazılmıştır.** Ajan, koda başlamadan önce belgenin tamamını okumalı, bölümleri sırasıyla uygulamalı ve belirsizlik olduğunda varsayım yapmak yerine insana tek, net bir soru sormalıdır.

---

## 0. Ajan İçin Hızlı Özet

- **Ne yapıyoruz:** Gürültülü konuşma kaydından temiz konuşmayı kestiren bir U-Net eğitiyoruz. Veriyi temiz konuşma + gürültü karıştırarak kendimiz üretiyoruz.
- **Ne bekleniyor:** Çalışan bir veri hattı, eğitim ve değerlendirme kodu, tekrar üretilebilir deneyler ve dürüst raporlanmış sonuçlar.
- **Hangi sırayla:** Bölüm 6'daki kilometre taşları (M0 → M6). Her kilometre taşının "Bitti sayılır" kriteri sağlanmadan sonrakine geçme.
- **En önemli kurallar:** Sonuç uydurma. Değerlendirme protokolünü deney ortasında değiştirme. Büyük indirme, bağımlılık veya mimari değişikliğinden önce sor. (Bölüm 9)

## 1. Proje Tanımı

Gürültülü konuşma sinyali `x = s + n` verildiğinde temiz konuşma `s`'yi kestiren bir model eğitilir.

**Araştırma soruları**

1. Hangi kayıp (L1, MSE, SI-SNR) hem metriklerde hem kulakla daha iyi?
2. U-Net'te derinlik ve skip connection'ın etkisi nedir?
3. Model eğitimde görmediği gürültü türüne genelleşiyor mu?
4. Düşük SNR'da performans nasıl değişiyor?

**Teslim edilecekler:** kod, deney sonuç tabloları, spektrogram görselleri, önce/sonra ses örnekleri, canlı demo (notebook), rapor için gerekli tüm sayılar.

## 2. Ortam ve Kısıtlar

| Konu | Değer |
|------|-------|
| Ana makine | Windows 11, NVIDIA RTX 3050 4 GB VRAM |
| Ağır koşular | Google Colab (ücretsiz katman) |
| Dil / çerçeve | Python 3.10+, PyTorch, torchaudio |
| Bellek önlemleri | Küçük model, kısa kesitler, küçük batch, AMP (`torch.autocast`) |
| PESQ | Windows'ta derleme sorunu çıkarabilir. PESQ hesabı Colab'da yapılabilmeli; `evaluate.py` PESQ kurulu değilse atlayıp uyarı vermeli, çökmemeli. |

**Platform kuralları**

- Tüm yollar `pathlib.Path` ile kurulur; mutlak yol yazılmaz. Windows ve Linux'ta çalışmalı.
- `DataLoader` için `num_workers` config'den gelir (Windows'ta varsayılan 0).
- Veri, checkpoint ve büyük ses dosyaları git'e eklenmez (`.gitignore`).

## 3. Veri Sözleşmesi

Bu bölümdeki değerler **varsayılandır** ve `configs/` içinden değiştirilebilir. Kod içine sabit gömülmez.

### 3.1 Ses

- Örnekleme hızı: **16 kHz**, mono, `float32`, aralık yaklaşık `[-1, 1]`.
- Eğitim kesiti: **3.0 sn** (48 000 örnek). Eğitimde rastgele kesit, validation/test'te sabit (ilk) kesit.
- Dalga formu tensörü: `(B, T)`.

### 3.2 STFT

- `n_fft=512`, `hop_length=128`, `win_length=512`, Hann penceresi, `center=True`, `return_complex=True`.
- 257 frekans bandı çıkar. **Nyquist bandı atılır, 256 bant kullanılır.** Geri dönüşümde o banda sıfır eklenir.
- Zaman ekseni, U-Net derinliğine göre `2**depth`'in katına **sıfır dolgu** ile tamamlanır; çıkışta orijinal uzunluğa **kırpılır**.
- Spektrogram tensörü: `(B, 1, F, T_frames)`.

### 3.3 Model Girdisi ve Çıktısı

- Girdi: `log1p(|X|)` (gürültülü sesin genliği).
- Çıktı modu **`mask`** (varsayılan): sigmoid ile `[0, 1]` maske `M`. Tahmin genlik: `est_mag = M * |X|`.
- Çıktı modu **`direct`**: model `log1p(temiz genlik)` tahmin eder. `est_mag = expm1(clamp(out, min=0))`.
- Geri dönüşüm: `est_wave = istft(est_mag * exp(j * angle(X)), length=T)`. Yani **faz gürültülü sinyalden alınır** (baseline kabulü). `length=T` ile tahmin ve referansın uzunluğu eşitlenir.

### 3.4 Veri Üretimi

- Temiz konuşma: LibriSpeech küçük alt kümesi **veya** VoiceBank-DEMAND.
- Gürültü: ESC-50 **veya** UrbanSound8K.
- Karışım SNR listesi (dB): `[-5, 0, 5, 10, 15]`. Her örnek için SNR bu listeden rastgele seçilir.
- Gürültü kayıttan kısaysa tekrarlanır, uzunsa rastgele kesit alınır.
- Karışım sonrası `max(|x|) > 1` ise **hem gürültülü hem temiz** sinyal aynı katsayıyla ölçeklenir (SNR korunur).
- Çıktı: ses dosyaları + `manifest.csv`.

**`manifest.csv` sütunları**

| Sütun | Anlam |
|-------|-------|
| `id` | Benzersiz örnek kimliği |
| `clean_path` | Temiz konuşma dosyası |
| `noisy_path` | Karışım dosyası |
| `noise_path` | Kullanılan gürültü kaynağı |
| `noise_class` | Gürültü sınıfı |
| `snr_db` | Karışım SNR'ı |
| `speaker_id` | Konuşmacı |
| `split` | `train` / `val` / `test` / `test_unseen_noise` |

### 3.5 Bölme Kuralları (veri sızıntısını önlemek için zorunlu)

1. Train/val/test ayrımı **konuşmacıya göre** yapılır. Aynı konuşmacı iki farklı bölmede bulunamaz.
2. Gürültü sınıflarının bir kısmı (config'de `unseen_noise_classes`) **yalnızca `test_unseen_noise`** bölmesinde kullanılır. Eğitimde ve validation'da görünmez.
3. Rastgele tohum `42` (config'den). Aynı tohum, aynı manifest üretir.
4. `make_dataset.py` sonunda **sızıntı kontrolü** yapıp ihlal varsa hata verir (konuşmacı ve gürültü sınıfı kesişimi).

## 4. Modül Arayüzleri

Aşağıdaki imzalar bağlayıcıdır. Gerekirse genişletilebilir ama isimler ve davranışlar korunmalıdır.

### `src/utils/audio.py`

```python
def load_audio(path: Path, sr: int = 16000) -> torch.Tensor:
    """Mono, `sr` Hz, float32 dalga formu (T,). Gerekirse yeniden örnekler."""

def stft(wave: torch.Tensor, n_fft: int, hop: int) -> torch.Tensor:
    """(B, T) -> karmaşık (B, F, T_frames). Hann penceresi, center=True."""

def istft(spec: torch.Tensor, n_fft: int, hop: int, length: int) -> torch.Tensor:
    """(B, F, T_frames) karmaşık -> (B, length)."""

def mix_at_snr(clean: torch.Tensor, noise: torch.Tensor, snr_db: float) -> torch.Tensor:
    """clean + scale * noise; ölçek, hedef SNR'ı sağlayacak şekilde hesaplanır."""
```

### `src/data/dataset.py`

`NoisySpeechDataset(manifest_path, split, segment_seconds, sr, ...)`. `__getitem__` bir sözlük döndürür:

```python
{"noisy": Tensor(T), "clean": Tensor(T), "snr_db": float, "id": str}
```

### `src/models/unet.py`

```python
class UNet(nn.Module):
    def __init__(self, in_ch=1, base_ch=16, depth=4, skip=True, out_mode="mask"): ...
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 1, F, T). mask modunda [0,1] maske, direct modunda log1p genlik."""
```

- `skip=False` iken skip connection'lar tamamen kapanır (karşılaştırma deneyi için). Kanal sayıları tutarlı kalmalı.
- Çıktı, girdinin `(F, T)` boyutuyla aynı olmalı (dolgu/kırpma model dışında veya içinde tutarlı yönetilir).

### `src/losses.py`

```python
def l1_loss(est_mag, clean_mag): ...       # log1p genlik uzayında
def mse_loss(est_mag, clean_mag): ...      # log1p genlik uzayında
def si_snr_loss(est_wave, clean_wave, eps=1e-8) -> torch.Tensor:
    """Negatif SI-SNR (dB), batch ortalaması."""
```

SI-SNR tanımı (önce ikisinin de ortalaması çıkarılır):

```
s_target = (<ŝ, s> / ||s||²) · s
e_noise  = ŝ − s_target
SI-SNR   = 10 · log10( ||s_target||² / ||e_noise||² )
```

### `src/train.py`

- `--config <yaml>` alır. Config'in kopyasını çıktı klasörüne kaydeder.
- AMP, gradyan kırpma, öğrenme hızı zamanlayıcı ve erken durdurma config'den yönetilir.
- En iyi validation checkpoint'i (`best.pt`) ve son checkpoint (`last.pt`) kaydedilir.
- Eğitim/validation kaybı CSV'ye yazılır (`log.csv`).

### `src/evaluate.py`

- `--checkpoint` ve `--split` alır (`test` veya `test_unseen_noise`).
- Örnek bazında `metrics_per_sample.csv`, özet olarak `summary.json` (ortalama ± std) üretir.
- Metrikler: **SI-SNR (dB)**, **PESQ** (`wb`, 16 kHz), **STOI**. Gürültülü girdinin metrikleri de aynı şekilde hesaplanıp raporlanır (iyileşme farkı için).
- Değerlendirme **sadece test bölmelerinde** yapılır.

### `src/infer.py`

Tek bir `.wav` alıp temizlenmiş `.wav` yazar. Demo notebook'u bunu kullanır.

## 5. Config Şeması (örnek)

```yaml
seed: 42
data:
  manifest: data/processed/manifest.csv
  sr: 16000
  segment_seconds: 3.0
  snr_list: [-5, 0, 5, 10, 15]
  unseen_noise_classes: []        # make_dataset sırasında doldurulur / belirlenir
stft:
  n_fft: 512
  hop: 128
model:
  name: unet
  base_ch: 16
  depth: 4
  skip: true
  out_mode: mask                  # mask | direct
loss:
  name: l1                        # l1 | mse | si_snr
train:
  batch_size: 8
  epochs: 50
  lr: 1.0e-3
  amp: true
  grad_clip: 5.0
  num_workers: 0
  early_stop_patience: 8
output_dir: experiments/baseline
```

## 6. Geliştirme Sırası ve "Bitti Sayılır" Kriterleri

### M0. Ortam
- `requirements.txt`, `.gitignore`, klasör iskeleti, `pytest` kurulumu.
- **Bitti:** Temiz bir ortamda kurulum ve `pytest` çalışıyor.

### M1. Ses yardımcıları
- `load_audio`, `stft`, `istft`, `mix_at_snr`.
- **Bitti:** Testler geçiyor:
  - STFT → iSTFT gidiş-dönüşü, rastgele sinyalde maksimum mutlak hata `< 1e-4`.
  - `mix_at_snr` çıktısında ölçülen SNR, hedefe `±0.1 dB` içinde.

### M2. Veri hattı
- `make_dataset.py`, `NoisySpeechDataset`, sızıntı kontrolü.
- Başlangıçta **küçük alt küme** (ör. birkaç yüz kayıt) ile çalış; tam veri indirmeden önce insana sor.
- **Bitti:** Manifest üretiliyor, sızıntı kontrolü geçiyor, 5 örnek dinlenip spektrogramı çizilebiliyor (insan kontrolü için `samples/` altına yaz).

### M3. Model ve kayıplar
- `UNet`, üç kayıp, birim testleri.
- **Bitti:**
  - Rastgele girdiyle çıktı boyutu girdiyle aynı.
  - `skip=False` ve farklı `depth` değerlerinde hata vermeden çalışıyor.
  - `si_snr_loss` ölçekten bağımsız (`est` ve `est * 3` aynı değeri veriyor).
  - **Tek bir batch'e overfit testi:** kayıp belirgin biçimde sıfıra yaklaşıyor.

### M4. Baseline eğitim ve değerlendirme
- Küçük U-Net + `mask` + L1.
- **Bitti:** `train.py` baştan sona çalışıyor, `evaluate.py` `summary.json` üretiyor, E0 (işlem yapılmamış gürültülü ses) metrikleri hesaplanmış.

### M5. Deneyler
- Bölüm 7'deki E1-E7. Her deney kendi config'i ve çıktı klasörüyle çalışır.
- **Bitti:** Her deney için `summary.json` mevcut, birden fazla tohumla tekrar edilebilenler tekrar edilmiş (mümkünse 3 tohum).

### M6. Demo ve sonuç tabloları
- `notebooks/demo.ipynb`: gürültülü → temiz canlı dinletme, spektrogram önce/sonra.
- Tüm deneylerin sonuçlarını toplayan tablo üreten script.
- **Bitti:** Demo, temiz bir ortamda baştan sona çalışıyor.

## 7. Deney Matrisi

| # | Deney | Karşılaştırılan | Config anahtarı |
|---|-------|-----------------|-----------------|
| E0 | Referans çizgisi | İşlem yok | – |
| E1 | Kayıp fonksiyonu | L1 / MSE / SI-SNR | `loss.name` |
| E2 | Skip connection | Var / yok | `model.skip` |
| E3 | Derinlik | 3 / 4 / 5 | `model.depth` |
| E4 | Çıkış türü | Maske / doğrudan | `model.out_mode` |
| E5 | SNR dayanıklılığı | -5 … 15 dB (SNR bazında raporla) | `data.snr_list` |
| E6 | Görülmemiş gürültü | `test` / `test_unseen_noise` | `unseen_noise_classes` |
| E7 | Gerçek kayıt | Telefonla kaydedilen gerçek ortam sesi | `infer.py` |
| E8 _(opsiyonel)_ | Girdi türü | Spektrogram / ham dalga formu | ayrı model |
| E9 _(opsiyonel)_ | Sınıflandırma etkisi | Gürültülü / temizlenmiş / temiz | ayrı script |

Her deneyde **tek bir değişken** değişir; diğer her şey baseline ile aynı kalır.

## 8. Kodlama Kuralları

- Type hint ve kısa docstring kullan. Fonksiyonlar küçük ve test edilebilir olsun.
- `print` yerine `logging`. Uzun döngülerde `tqdm`.
- **Tekrar üretilebilirlik:** Tohumları (`random`, `numpy`, `torch`) config'ten ayarla. Her çıktı klasörüne kullanılan config'i ve git commit bilgisini (varsa) yaz.
- Sihirli sayıları koda gömme, config'e koy.
- Birim testleri `tests/` altında: STFT gidiş-dönüşü, SNR karışımı, SI-SNR ölçek değişmezliği, model çıktı boyutları, sızıntı kontrolü.
- Küçük, anlamlı commit'ler. Bir commit tek bir değişikliği içersin.
- Bağımlılık eklemeden önce nedenini söyle ve `requirements.txt`'e ekle.

## 9. Ajan Davranış Kuralları

**Yap**
- Her kilometre taşını bitirince ne yaptığını, nasıl doğruladığını ve varsa açık sorunları kısaca özetle.
- Belirsiz bir noktada **tek, net bir soru** sor (ör. "LibriSpeech mi VoiceBank-DEMAND mi?").
- Varsayım yaptıysan açıkça yaz.
- Hata veya beklenmeyen sonuçları olduğu gibi raporla.

**Yapma**
- **Sonuç, metrik veya tablo değeri uydurma.** Değerler yalnızca gerçekten çalıştırılan koddan gelir.
- Değerlendirme protokolünü (metrik, bölme, ön işleme) deneyler başladıktan sonra sessizce değiştirme.
- Test setine bakarak hiperparametre seçme. Seçim validation üzerinden yapılır.
- İzinsiz büyük veri seti indirme, ağır eğitim başlatma veya mimariyi değiştirme.
- Mevcut deney çıktılarını üzerine yazma; yeni deney yeni klasöre yazılır.
- Sorun çıktığında hatayı gizleyen `try/except` ile geçiştirme.

## 10. Bilinen Tuzaklar

| Tuzak | Önlem |
|-------|-------|
| Tahmin ve referans uzunlukları iSTFT sonrası farklı çıkar | `istft(..., length=T)` kullan |
| Karışım sonrası klipleme | Gürültülü ve temiz sinyali aynı katsayıyla ölçekle |
| Aynı konuşmacı train ve test'te | Konuşmacıya göre böl, sızıntı kontrolü ekle |
| Görülmeyen gürültü yanlışlıkla eğitime karışır | Sınıfa göre ayır, sızıntı kontrolüyle doğrula |
| Spektrogram boyutu U-Net derinliğine uymaz | `2**depth` katına dolgula, çıkışta kırp |
| SI-SNR kaybında NaN | `eps` ekle, sıfır enerjili segmentleri kontrol et |
| PESQ Windows'ta kurulmuyor | PESQ'i opsiyonel yap, Colab'da çalıştır |
| Robotik ses | Faz gürültülü sinyalden geldiği için beklenen bir sınırlılık; raporda belirt |
| 4 GB VRAM'de bellek hatası | Batch ve kesit uzunluğunu düşür, AMP aç, taban kanalı küçült |

## 11. Hedef Klasör Yapısı

```
.
├── README.md
├── requirements.txt
├── .gitignore
├── configs/
│   └── baseline.yaml
├── data/                    # git'e eklenmez
│   ├── raw/
│   └── processed/
├── src/
│   ├── data/
│   │   ├── make_dataset.py
│   │   └── dataset.py
│   ├── models/
│   │   └── unet.py
│   ├── losses.py
│   ├── train.py
│   ├── evaluate.py
│   ├── infer.py
│   └── utils/
│       ├── audio.py
│       └── plotting.py
├── tests/
├── notebooks/
│   └── demo.ipynb
├── experiments/             # çıktılar, loglar (git'e eklenmez)
└── samples/                 # örnek ses ve görseller
```

## 12. Sonuçlar

Bu bölüm deneyler bittikçe `summary.json` dosyalarından doldurulur. Değerler elle uydurulmaz.

| Model / Kayıp | SI-SNR (dB) | PESQ | STOI |
|---------------|-------------|------|------|
| Gürültülü girdi (E0) | – | – | – |
| U-Net + L1 | – | – | – |
| U-Net + MSE | – | – | – |
| U-Net + SI-SNR | – | – | – |

## 13. Kaynaklar

Yön gösterici içindir; kullanmadan önce güncel hallerini ve veri setlerinin lisanslarını kontrol et.

- Ronneberger ve ark., *U-Net: Convolutional Networks for Biomedical Image Segmentation*, 2015
- Panayotov ve ark., *LibriSpeech*, 2015
- Valentini-Botinhao ve ark., VoiceBank-DEMAND veri seti
- Piczak, *ESC-50*, 2015; Salamon ve ark., *UrbanSound8K*, 2014
- Le Roux ve ark., *SDR – Half-baked or Well Done?* (SI-SNR), 2019
- Rix ve ark., PESQ, 2001; Taal ve ark., STOI, 2011