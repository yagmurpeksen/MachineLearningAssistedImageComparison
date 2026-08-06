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
- Bağımlılıklar: `requirements.txt` (geliştirme/test için `requirements-dev.txt`)

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

### Ana uygulama (GUI)

```bash
python image_comparison_gui.py
```

GUI özellikleri: çift görüntü analizi, yıldız kataloğu, kullanıcı etiketleme ile model eğitimi, ALeRCE broker sorgusu (ayarlar penceresi üzerinden parametreler).

### Standalone EXE (Windows)

```bash
python build_app.py
# dist\BrightnessComparison.exe oluşur
```

> **Not:** `transient_model.joblib` dosyasının EXE ile aynı klasörde bulunması gerekir.

### Performans benchmark'ı (isteğe bağlı)

```bash
python benchmark.py
python benchmark.py --no-alerce   # ALeRCE API testini atla
```

Çıktı: `report_figures/benchmark.png` (klasör yoksa oluşturulur). Benchmark, yerel FITS çiftlerinin varlığına bağlıdır; eksik dosyalar atlanır.

---

## Gerçek ZTF testi (full-field, ~3000×3000 px)

GUI ve ALeRCE ile gerçekçi deneme için **IRSA tam alan** ZTF `sciimg` (~35–38 MB/FITS) kullanın. Küçük Fink kesitleri (~63×63) veya sentetik `test_epoch*.fits` tam alan testinin yerine geçmez.

| Klasör | Ne için? |
|--------|----------|
| `sn_target/AT2020xnd_short/` | **Önerilen** — AT2020xnd (ZTF20acigmel), yakın epoch; `target.json` içinde tarih/koordinat |
| `sn_target/AT2020xnd/` | Uzun baseline (2019 / 2020) — daha zor senaryo |
| `sn_target/AT2020xnd_balanced/` | Benchmark varsayılan çifti |
| `sn_target/fink_ZTF*/` | Yalnızca küçük kesit metadata + `.npy` (GUI tam alan testi için uygun değil) |

Her `sn_target/<isim>/` klasörüne IRSA'dan indirdiğiniz tam alan FITS'leri şu adlarla koyun:

- `pre_sn.fits` — önceki epoch
- `post_sn.fits` — sonraki epoch

> **Not:** `*.fits` dosyaları `.gitignore` ile repoya dahil edilmez; indirme IRSA [ZTF archive](https://irsa.ipac.caltech.edu/) üzerinden manuel yapılır. Klasördeki `target.json` hedef koordinat ve epoch bilgisini tutar.

GUI: **Image 1** = `pre_sn.fits`, **Image 2** = `post_sn.fits`. AT2020xnd koordinatı: RA ≈ 8.05°, Dec ≈ 36.76°.

---

## Modeli Eğitmek

Deep-TAO veri setini [Zenodo](https://zenodo.org/record/3820679) üzerinden indirip `deeptao_data/` altına SN, AGN, CV, BZ, OTHER ve NON alt klasörleriyle yerleştirin (`deeptao_data/` git'e dahil değildir).

```bash
# Varsayılan: ./deeptao_data/ → transient_model.joblib
python train_model.py

# Özel dizin:
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

Kullanıcı etiketleri GUI'den de toplanır; `user_model.joblib` olarak kaydedilir (gitignore).

---

## Test

```bash
pip install -r requirements-dev.txt
pytest -q           # tüm testler
pytest tests/ -v    # ayrıntılı çıktı
```

CI: `.github/workflows/ci.yml` — Python 3.11, `pytest -q`.

| Modül | Kapsam |
|-------|--------|
| `test_detection.py` | Yıldız tespiti, hizalama, eşleştirme, fotometri |
| `test_ml_classifier.py` | Özellik çıkarma, TransientClassifier, Isolation Forest |
| `test_pipeline_integration.py` | Eğitim ↔ inference özellik uyumu, kayıtlı model |
| `test_fits_loader.py` | FITS yükleme ve `sqrt_stretch` |
| `test_change_detection.py` | Sentetik transient recall/precision |
| `test_noise_robustness.py` | Gürültü ve kozmik ışın dayanıklılığı |
| `test_star_detection.py` | Temel tespit duman testi |

---

## Proje Yapısı

```
cse492/
├── image_comparison_gui.py   # Ana uygulama (Tkinter GUI)
├── detection.py                # Yıldız tespiti, hizalama, eşleştirme, fotometri
├── ml_classifier.py            # Özellik çıkarma, TransientClassifier, Isolation Forest
├── fits_loader.py              # FITS yükleme ve sqrt-stretch
├── photometry.py               # APASS / katalog fotometrik kalibrasyon
├── alerce_query.py             # ALeRCE REST API (ZTF broker)
├── train_model.py              # Deep-TAO üzerinde model eğitimi
├── benchmark.py                # Pipeline + ALeRCE performans ölçümü
├── build_app.py                # PyInstaller EXE üretici
├── transient_model.joblib      # Pre-trained model (54 özellik)
├── gui/
│   ├── settings_window.py      # Analiz parametreleri
│   ├── catalog_window.py       # Yıldız kataloğu
│   ├── label_window.py         # Olay etiketleme ve kullanıcı modeli
│   └── alerce_window.py        # ALeRCE sorgu penceresi
├── tests/                      # Pytest paketi
├── sn_target/                  # Gerçek SN hedef metadata (target.json; FITS yerel)
├── deeptao_data/               # Eğitim verisi (yerel, gitignore)
├── report_figures/             # benchmark.py çıktıları
├── requirements.txt
└── requirements-dev.txt
```

**Repoda olmayan / kaldırılmış:** Eski indirme ve doğrulama scriptleri (`download_*.py`, `find_and_download_sn.py`, `validate_at2020xnd.py`, `test_examples.py`) artık bu sürümde yok. FITS indirme IRSA üzerinden manuel yapılır.

---

## Desteklenen Formatlar

| Format | Destek |
|--------|--------|
| FITS (.fits) | Tam destek — astropy ile ham float verisi, WCS, fotometri |
| PNG / JPEG / BMP / TIFF | Temel karşılaştırma (ML filtresi aktif, fotometri yok) |

## Kaynaklar

- Gómez et al. (2020) — *"Deep learning for transient recognition and characterization of the Deep-TAO survey"*, A&A
- Deep-TAO Dataset: [https://zenodo.org/record/3820679](https://zenodo.org/record/3820679)
- ALeRCE: [https://alerce.online](https://alerce.online)
