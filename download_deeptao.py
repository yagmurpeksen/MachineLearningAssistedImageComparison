"""
Deep-TAO Dataset Downloader
Her sınıftan N adet FITS dosyası indirir.
Raw GitHub URL kullanır (LFS bypass).
"""

import requests
import os
import re
import time
import sys
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

# ── Ayarlar ──────────────────────────────────────────────────────
N_PER_CLASS = 500
SAVE_DIR    = r"C:\Users\pekse\Desktop\cse492\deeptao_data"
ORG         = "MachineLearningUniandes"
API_BASE    = f"https://api.github.com/repos/{ORG}"
RAW_BASE    = f"https://raw.githubusercontent.com/{ORG}"
BRANCH      = "main"

TRANSIENT_CLASSES = ["SN", "AGN", "CV", "BZ", "OTHER"]

# ── Yardımcı fonksiyonlar ─────────────────────────────────────────

def safe_name(name):
    """Windows'ta geçersiz karakterleri temizle."""
    return re.sub(r'[:\\/*?<>|"]', '_', name)

def list_files_api(repo, path):
    """GitHub API ile klasördeki dosyaları listele (max 1000)."""
    url = f"{API_BASE}/{repo}/contents/{path}"
    headers = {"Accept": "application/vnd.github.v3+json"}
    try:
        r = requests.get(url, headers=headers, timeout=20)
        if r.status_code == 403:
            print("  Rate limit! 60 saniye bekleniyor...")
            time.sleep(60)
            r = requests.get(url, headers=headers, timeout=20)
        if r.status_code != 200:
            print(f"  API hata {r.status_code}: {url}")
            return []
        items = r.json()
        if isinstance(items, dict):
            print(f"  API mesajı: {items.get('message','?')}")
            return []
        return [it for it in items if it["type"] == "file"]
    except Exception as e:
        print(f"  Bağlantı hatası: {e}")
        return []

def download_raw(repo, path, filename, dest):
    """Raw GitHub URL ile dosyayı indir."""
    url = f"{RAW_BASE}/{repo}/{BRANCH}/{path}/{filename}"
    try:
        r = requests.get(url, timeout=30)
        if r.status_code == 200 and len(r.content) > 1000:
            with open(dest, "wb") as f:
                f.write(r.content)
            return True
        return False
    except Exception:
        return False

def download_class(repo, data_path, cls_name, save_dir, n):
    os.makedirs(save_dir, exist_ok=True)
    already = len([f for f in os.listdir(save_dir)
                   if f.endswith(".fits") or "." not in f])
    if already >= n:
        print(f"  [{cls_name}] Zaten {already} dosya var, atlanıyor.")
        return already

    print(f"\n[{cls_name}] dosyalar listeleniyor ({data_path})...")
    all_files = list_files_api(repo, data_path)
    fits_files = [f for f in all_files
                  if f["name"].endswith(".fits") or "." not in f["name"]]
    fits_files = fits_files[:n]

    if not fits_files:
        print(f"  Dosya bulunamadi!")
        return 0

    need = n - already
    print(f"  {len(fits_files)} hedef, {already} zaten var, {need} indirilecek...")

    ok = already
    skipped = 0
    for i, item in enumerate(fits_files):
        orig_name = item["name"]
        fname     = safe_name(orig_name)
        if not fname.endswith(".fits"):
            fname += ".fits"
        dest = os.path.join(save_dir, fname)

        if os.path.exists(dest) and os.path.getsize(dest) > 1000:
            skipped += 1
            continue

        success = download_raw(repo, data_path, orig_name, dest)
        if success:
            ok += 1
        else:
            # API download_url dene (fallback)
            try:
                r = requests.get(item["download_url"], timeout=30)
                if r.status_code == 200 and len(r.content) > 1000:
                    with open(dest, "wb") as f:
                        f.write(r.content)
                    ok += 1
            except Exception:
                pass

        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(fits_files)} islendi  ({ok} basarili)...")
        time.sleep(0.05)

    print(f"  [{cls_name}] tamamlandi: {ok}/{len(fits_files)} dosya -> {save_dir}")
    return ok


def main():
    print("=" * 60)
    print(f"Deep-TAO Dataset Downloader  —  her siniftan {N_PER_CLASS} dosya")
    print("=" * 60)

    # ── Transient siniflar ──
    for cls in TRANSIENT_CLASSES:
        save_dir  = os.path.join(SAVE_DIR, cls)
        data_path = f"data/{cls}"
        download_class("TAO_transients", data_path, cls, save_dir, N_PER_CLASS)

    # ── Non-transient ──
    save_dir  = os.path.join(SAVE_DIR, "NON")
    data_path = "data/NON"
    download_class("TAO_non-transients", data_path, "NON", save_dir, N_PER_CLASS)

    # ── Ozet ──
    print("\n" + "=" * 60)
    print("Indirme tamamlandi!")
    total = 0
    for cls in TRANSIENT_CLASSES + ["NON"]:
        d = os.path.join(SAVE_DIR, cls)
        if os.path.isdir(d):
            cnt = len(os.listdir(d))
            total += cnt
            print(f"  {cls:8s}: {cnt} dosya")
    print(f"  {'TOPLAM':8s}: {total} dosya")
    print("=" * 60)


if __name__ == "__main__":
    main()
