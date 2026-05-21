# Astronomical Transient Detection Tool

Astronomik görüntü çiftlerini karşılaştırarak **süpernova, nova, AGN parlaması** gibi geçici olayları (transient) otomatik olarak tespit eden masaüstü uygulaması.

## Projenin Özeti

İki farklı zamanda çekilmiş FITS görüntüsü yüklendiğinde sistem otomatik olarak:

1. **Yıldız tespiti** — Her görüntüdeki kaynakları bulur
2. **Astrometrik hizalama** — İki görüntüyü piksel düzeyinde hizalar
3. **Eşleştirme** — Karşılıklı en yakın komşu yöntemiyle yıldızları eşler
4. **Hareket tespiti** — IQR tabanlı istatistikle gerçek hareket eden cisimleri ayırt eder
5. **Parlaklık değişimi** — Apertur fotometrisiyle anlamlı değişimleri ölçer
6. **ML sınıflandırması** — Pre-trained Random Forest modeli ile her adayı transient / non-transient olarak sınıflandırır

### Makine Öğrenmesi Pipeline'ı

| Bileşen | Detay |
|---------|-------|
| Eğitim verisi | Deep-TAO veri seti (Gómez et al. 2020) — 2 991 FITS dosyası |
| Sınıflar | SN · AGN · CV · BZ · OTHER → Transient (1) &nbsp;/&nbsp; NON → Non-transient (0) |
| Özellik vektörü | **54 boyut**: 18 erken + 18 geç + 8 delta + 10 subtraction (image-differencing) |
| Model | Random Forest — GridSearchCV ile optimize edilmiş |
| AUC | **0.838** |
| CV Accuracy | **79.0% ± 1.2%** (5-fold) |
| Karar eşiği | Youden J istatistiği (ROC eğrisi optimumu) |
| Baseline | Dummy sınıflandırıcı %55.1 — modelimiz %78+ |

---

## Gereksinimler

- Python 3.11+
- Bağımlılıklar: `requirements.txt`

## Kurulum

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

## Çalıştırma

### Seçenek 1 — Python ile

```bash
python image_comparison_gui.py
```

### Seçenek 2 — Standalone EXE (Windows)

```bash
python build_app.py
# dist\StellarAnalysis.exe oluşur
```

> **Not:** `transient_model.joblib` dosyasının EXE ile aynı klasörde bulunması gerekir.

---

## Modeli Eğitmek

```bash
# Varsayılan: ./deeptao_data/ klasöründen okur, transient_model.joblib yazar
python train_model.py

# Özel dizin belirtmek için:
python train_model.py --data-dir /yol/deeptao --model-out ./model.joblib
```

Eğitim çıktısı — GridSearchCV, Youden J eşik optimizasyonu ve karşılaştırma tablosu:

```
  Metrik                     Baseline   RF(thr=0.5)  RF(optimal)
  Accuracy                     55.1%       78.8%        78.1%
  Transient precision            N/A        0.824        0.842
  Transient recall               N/A        0.867        0.827
  False Positive Rate            N/A        0.370        0.310
  AUC (ROC)                      N/A        0.838        0.838
```

---

## Test

```bash
pytest -q           # tüm testler
pytest tests/ -v    # ayrıntılı çıktı
```

Test kapsamı (111 test):
- `test_detection.py` — yıldız tespiti, hizalama, eşleştirme, fotometri
- `test_ml_classifier.py` — özellik çıkarma, TransientClassifier, Isolation Forest
- `test_pipeline_integration.py` — train↔inference uyumu, tam pipeline zinciri
- `test_fits_loader.py` / `test_star_detection.py` / `test_change_detection.py` / `test_noise_robustness.py` — modül testleri

---

## Proje Yapısı

```
cse492/
├── image_comparison_gui.py   # Ana uygulama (Tkinter GUI)
├── detection.py              # Yıldız tespiti, hizalama, fotometri
├── ml_classifier.py          # Özellik çıkarma, TransientClassifier
├── fits_loader.py            # FITS dosya yükleme
├── photometry.py             # Fotometrik kalibrasyon (APASS)
├── alerce_query.py           # ALeRCE ZTF broker sorgulama
├── train_model.py            # Model eğitim scripti
├── find_and_download_sn.py   # ZTF veri indirme yardımcısı
├── validate_at2020xnd.py     # AT2020xnd tespit doğrulama scripti
├── test_examples.py          # Komut satırı örnek testleri
├── build_app.py              # PyInstaller EXE üretici
├── test_ground_truth.json    # Sentetik test ground truth koordinatları
├── transient_model.joblib    # Pre-trained model (54 özellik)
├── deeptao_data/             # Deep-TAO eğitim verisi (SN/AGN/CV/BZ/OTHER/NON)
├── sn_target/                # Gerçek ZTF test verisi (AT2020xnd vb.)
├── moving_object/            # Hareketli cisim test görüntüleri
├── report_figures/           # ML metrik grafikleri
├── tests/                    # Pytest test paketi (111 test)
└── requirements.txt
```

## Desteklenen Formatlar

| Format | Destek |
|--------|--------|
| FITS (.fits) | Tam destek — astropy ile ham float verisi, WCS, fotometri |
| PNG / JPEG / BMP / TIFF | Temel karşılaştırma (ML filtresi aktif, fotometri yok) |

## Kaynaklar

- Gómez et al. (2020) — *"Deep learning for transient recognition and characterization of the Deep-TAO survey"*, A&A
- Deep-TAO Dataset: [https://zenodo.org/record/3820679](https://zenodo.org/record/3820679)
