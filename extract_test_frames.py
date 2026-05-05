"""
Deep-TAO FITS dosyasından ilk ve son frame'i
iki ayrı test FITS dosyası olarak kaydeder.
"""

import os
import numpy as np
from astropy.io import fits as astropy_fits

DATA_DIR = r"C:\Users\pekse\Desktop\cse492\deeptao_data"
OUT_DIR  = r"C:\Users\pekse\Desktop\cse492\test_frames"
os.makedirs(OUT_DIR, exist_ok=True)

def extract_frames(cls="SN"):
    cls_dir = os.path.join(DATA_DIR, cls)
    files   = [f for f in os.listdir(cls_dir) if f.endswith(".fits") or "." not in f]

    for fname in files:
        path = os.path.join(cls_dir, fname)
        try:
            with astropy_fits.open(path) as hdul:
                # Frame'leri topla
                frames = []
                for hdu in hdul[1:]:
                    if hdu.data is not None and np.ndim(hdu.data) == 2:
                        frames.append(hdu.data.astype(np.float32))

                if len(frames) < 5:
                    continue  # Çok az frame, atla

                early = frames[0]
                late  = frames[-1]

                # Değişim var mı kontrol et
                diff = abs(float(np.mean(late)) - float(np.mean(early)))
                if diff < 1.0:
                    continue  # Değişim yok, başka dosya dene

                # Kaydet
                base = os.path.splitext(fname)[0]
                early_path = os.path.join(OUT_DIR, f"{base}_early.fits")
                late_path  = os.path.join(OUT_DIR, f"{base}_late.fits")

                astropy_fits.writeto(early_path, early, overwrite=True)
                astropy_fits.writeto(late_path,  late,  overwrite=True)

                n_frames = len(frames)
                mean_e   = float(np.mean(early))
                mean_l   = float(np.mean(late))
                print(f"[{cls}] {fname}")
                print(f"  {n_frames} frame  |  erken ortalama={mean_e:.1f}  gec ortalama={mean_l:.1f}  fark={diff:.1f}")
                print(f"  Kaydedildi:")
                print(f"    {early_path}")
                print(f"    {late_path}")
                print()
                return early_path, late_path  # İlk uygun dosyayı bulduk, dur

        except Exception as e:
            continue

    print(f"[{cls}] Uygun dosya bulunamadi.")
    return None, None


if __name__ == "__main__":
    print("Test frame'leri cikartiliyor...\n")
    for cls in ["SN", "CV", "AGN", "BZ", "OTHER"]:
        e, l = extract_frames(cls)
        if e:
            break
