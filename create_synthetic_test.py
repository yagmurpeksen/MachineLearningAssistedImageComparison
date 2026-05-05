"""
Sentetik test verisi olusturur.

STRATEJI (3. deneme):
  before = orijinal goruntu ANCAK secilen yildiz 3x soluklaştirilmiş
  after  = orijinal goruntu (yildiz normal parlaklıkta)

Boylece:
  - Her iki goruntude de yildiz mevcut ve algilanabilir (secimle kaybolmuyor)
  - before'un max degeri DEGISMIYOR (normalizasyon bozulmuyor)
  - flux_ratio = b2/b1 ≈ 3  >  esik 2.5 → "Brightness Change" olarak gozukecek
"""

import numpy as np
from astropy.io import fits
from scipy.ndimage import label as scipy_label, center_of_mass

BASE  = r"C:\Users\pekse\Desktop\cse492\test_frames\SN_field_DSS.fits"
OUT1  = r"C:\Users\pekse\Desktop\cse492\test_frames\synthetic_before.fits"
OUT2  = r"C:\Users\pekse\Desktop\cse492\test_frames\synthetic_after.fits"

# -- Goruntu yukle --
with fits.open(BASE) as hdul:
    data   = hdul[0].data.astype(np.float32).copy()
    header = hdul[0].header

h, w = data.shape
bg   = float(np.median(data))
print(f"Goruntu boyutu : {w} x {h}")
print(f"Arka plan medyan: {bg:.0f}")
print(f"Goruntu max    : {data.max():.0f}")

# -- Merkeze yakin, orta parlaklikta, KUCUK bir yildiz sec --
# max_area=150 kuralini geçemeyecek kadar buyuk olmasın
# yani ~20-60 piksel civarinda alan hedef
cx_min, cx_max = w//2 - 120, w//2 + 120
cy_min, cy_max = h//2 - 120, h//2 + 120
region = data[cy_min:cy_max, cx_min:cx_max]

sigma  = np.std(region)
thresh = bg + 4 * sigma
mask   = region > thresh
labeled, n_obj = scipy_label(mask)

best_star = None
best_score = -1

for i in range(1, n_obj + 1):
    pix = np.where(labeled == i)
    area = len(pix[0])
    if area < 4 or area > 40:   # kucuk yildiz sec
        continue
    peak = float(region[pix].max())
    # Orta parlaklik: bg*1.3 < peak < bg*4
    if bg * 1.3 < peak < bg * 4.0:
        if peak > best_score:
            best_score = peak
            cy_r, cx_r = center_of_mass(region, labeled, i)
            best_star = (int(cx_r) + cx_min, int(cy_r) + cy_min, peak, area)

if best_star is None:
    # fallback: merkezdeki varsayilan
    best_star = (w//2 - 20, h//2 + 15, bg*2.5, 10)
    print("Uyari: Uygun kucuk yildiz bulunamadi, varsayilan kullanildi.")

SN_X, SN_Y, orig_peak, orig_area = best_star
DIM_FACTOR = 0.30   # before'da yildiz bu kadar soldu
# flux ratio = 1 / DIM_FACTOR = 3.33x  >  esik 2.5 ✓

print(f"\nSecilen yildiz : ({SN_X}, {SN_Y})")
print(f"Orjinal peak   : {orig_peak:.0f}  (bg = {bg:.0f})")
print(f"Alan           : {orig_area} piksel")
print(f"DIM_FACTOR     : {DIM_FACTOR}  → flux_ratio beklenen ≈ {1/DIM_FACTOR:.1f}x")

# -- Before: secilen yildiz soluklaştirilmis --
#   Her piksel: yeni_deger = bg + (eski - bg) * DIM_FACTOR
before = data.copy()
r_edit = 10   # etkilenecek yarıcap (piksel)
for dy in range(-r_edit, r_edit + 1):
    for dx in range(-r_edit, r_edit + 1):
        py, px = SN_Y + dy, SN_X + dx
        if 0 <= px < w and 0 <= py < h:
            excess = before[py, px] - bg
            before[py, px] = bg + excess * DIM_FACTOR

fits.writeto(OUT1, before, header=header, overwrite=True)
print(f"\nBefore kaydedildi: {OUT1}")
print(f"  (yildiz soluklaştirildi: peak {orig_peak:.0f} → {bg + (orig_peak-bg)*DIM_FACTOR:.0f})")

# -- After: tamamen orijinal goruntu --
after = data.copy()
fits.writeto(OUT2, after, header=header, overwrite=True)
print(f"After  kaydedildi: {OUT2}")
print(f"  (orijinal, degistirilmedi)")

print()
print("=" * 55)
print("GUI test talimatı:")
print("  Image 1 → synthetic_before.fits  (yildiz soluk)")
print("  Image 2 → synthetic_after.fits   (yildiz parlak)")
print(f"  Beklenen: ({SN_X}, {SN_Y}) konumunda BRIGHTNESS CHANGE up")
print(f"  flux_ratio ≈ {1/DIM_FACTOR:.1f}x  >  esik 2.5  → bulunmali!")
print("=" * 55)
