"""
Sadece eksik kalan sınıfları indirir (AGN, CV, BZ, OTHER, NON).
"""

import requests, os, re, time, sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

N_PER_CLASS = 500
SAVE_DIR    = r"C:\Users\pekse\Desktop\cse492\deeptao_data"
ORG         = "MachineLearningUniandes"
API_BASE    = f"https://api.github.com/repos/{ORG}"
RAW_BASE    = f"https://raw.githubusercontent.com/{ORG}"
BRANCH      = "main"

def safe_name(name):
    return re.sub(r'[:\\/*?<>|"]', '_', name)

def list_files_api(repo, path):
    url = f"{API_BASE}/{repo}/contents/{path}"
    headers = {"Accept": "application/vnd.github.v3+json"}
    try:
        r = requests.get(url, headers=headers, timeout=20)
        if r.status_code == 403:
            print("  Rate limit! 60 saniye bekleniyor...")
            time.sleep(60)
            r = requests.get(url, headers=headers, timeout=20)
        if r.status_code != 200:
            print(f"  API hata {r.status_code}")
            return []
        items = r.json()
        if isinstance(items, dict):
            print(f"  API: {items.get('message','?')}")
            return []
        return [it for it in items if it["type"] == "file"]
    except Exception as e:
        print(f"  Hata: {e}"); return []

def download_raw(repo, path, filename, dest):
    url = f"{RAW_BASE}/{repo}/{BRANCH}/{path}/{filename}"
    try:
        r = requests.get(url, timeout=30)
        if r.status_code == 200 and len(r.content) > 1000:
            with open(dest, "wb") as f: f.write(r.content)
            return True
    except Exception:
        pass
    return False

def download_class(repo, data_path, cls_name, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    already = len([f for f in os.listdir(save_dir) if os.path.getsize(os.path.join(save_dir,f)) > 1000])

    if already >= N_PER_CLASS:
        print(f"[{cls_name}] Zaten {already} dosya var, atlanıyor.")
        return already

    need = N_PER_CLASS - already
    print(f"\n[{cls_name}] {already} var, {need} daha lazım...")

    all_files = list_files_api(repo, data_path)
    fits_files = [f for f in all_files if f["name"].endswith(".fits") or "." not in f["name"]]

    if not fits_files:
        print(f"  Dosya bulunamadi!"); return already

    fits_files = fits_files[:N_PER_CLASS]
    print(f"  {len(fits_files)} dosya listelendi")

    ok = already
    for i, item in enumerate(fits_files):
        fname = safe_name(item["name"])
        if not fname.endswith(".fits"): fname += ".fits"
        dest = os.path.join(save_dir, fname)
        if os.path.exists(dest) and os.path.getsize(dest) > 1000:
            ok += 1; continue
        if download_raw(repo, data_path, item["name"], dest):
            ok += 1
        else:
            try:
                r = requests.get(item["download_url"], timeout=30)
                if r.status_code == 200 and len(r.content) > 1000:
                    with open(dest,"wb") as f: f.write(r.content)
                    ok += 1
            except: pass
        if (i+1) % 50 == 0:
            print(f"  {ok}/{N_PER_CLASS} tamamlandi...")
        time.sleep(0.05)

    print(f"  [{cls_name}] bitti: {ok} dosya")
    return ok

def main():
    print("="*55)
    print("Eksik siniflar indiriliyor...")
    print("="*55)

    # Transient sınıflar (SN hariç)
    for cls in ["AGN", "CV", "BZ", "OTHER"]:
        download_class("TAO_transients", f"data/{cls}", cls,
                      os.path.join(SAVE_DIR, cls))

    # Non-transient
    download_class("TAO_non-transients", "data/NON", "NON",
                  os.path.join(SAVE_DIR, "NON"))

    print("\n" + "="*55)
    total = 0
    for cls in ["SN","AGN","CV","BZ","OTHER","NON"]:
        d = os.path.join(SAVE_DIR, cls)
        if os.path.isdir(d):
            cnt = len(os.listdir(d))
            total += cnt
            print(f"  {cls:8s}: {cnt}")
    print(f"  {'TOPLAM':8s}: {total}")

if __name__ == "__main__":
    main()
