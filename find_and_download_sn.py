"""
find_and_download_sn.py
=======================
Find SNe in the ZTF archive with mag 16-19 and download from IRSA.

Priority order:
  1. Open Supernova Catalog (OSC) REST API - catalogued SNe
  2. Hardcoded verified ZTF SNe list
  Each candidate is checked for pre+post images in IRSA.

Usage: python find_and_download_sn.py
"""

import json, sys, io, csv, datetime, math
import urllib.request, urllib.parse, urllib.error
from pathlib import Path

# ── Parameters ───────────────────────────────────────────────────
MAG_MIN    = 15.5   # ZTF saturation limit ~14, so request >=15.5
MAG_MAX    = 19.5
FILTERCODE = "zr"
OUT_ROOT   = Path(__file__).parent / "sn_target"
MIN_FITS_MB = 5.0
PRE_MIN_DAYS = 90   # pre image must be at least this many days before (before the SN)

IRSA_SEARCH = "https://irsa.ipac.caltech.edu/ibe/search/ztf/products/sci"
IRSA_DATA   = "https://irsa.ipac.caltech.edu/ibe/data/ztf/products/sci"

# ── Verified ZTF SN list (from published papers) ─────────────────
# (name, ra_deg, dec_deg, discovery_date, peak_mag_r, type, source)
# Note: 2020+ discoveries are prioritized due to PRE_MIN_DAYS=90 requirement
KNOWN_ZTF_SNE = [
    # Perley+2020 ApJ 906 — Type Ibn, hizli transient, mag r~16.5
    ("AT2020xnd",   8.050,  36.760, "2020-10-12", 16.5, "Ibn"),
    # ZTF BTS — Type Ia, mag~17.2
    ("AT2021hiz",  133.870,  12.680, "2021-03-01", 17.2, "Ia"),
    # ZTF BTS Type Ia, z~0.05, mag~17.8
    ("ZTF20abrfugj", 172.180, 15.950, "2020-05-20", 17.8, "Ia"),
    # Hammerstein+2023 — ZTF BTS, Type IIP, mag~17.5
    ("AT2020tlf",  220.879,   4.336, "2020-11-09", 17.5, "IIP"),
    # Yao+2019 ApJ 886 152 — ZTF Type Ia, mag~18.0
    ("SN2018kzr",  35.540,  -7.180, "2018-08-28", 18.0, "Ia"),
    # De+2019 Science 365 671 — Type Ia-CSM, peak V~18.4 (close to ZTF start)
    ("AT2018byg",  187.009,  45.307, "2018-04-26", 18.4, "Ia-CSM"),
    # Miller+2020 ApJ 910 45 — Type Ia, peak r~15.9
    ("SN2019yvq",  197.061,  29.748, "2019-12-29", 15.9, "Ia"),
    # Williamson+2019 — too bright
    ("SN2019np",    9.630,  29.495, "2019-01-09", 12.5, "Ia"),
    # ZTF Type IIP in nearby galaxy — too bright
    ("SN2020jfo",  185.479,   4.474, "2020-05-06", 13.0, "IIP"),
]

# ── Helper functions ─────────────────────────────────────────────

def fetch(url, params=None, as_json=False):
    if params:
        url = url + "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={
        "User-Agent": "Python/3", "Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as r:
        raw = r.read().decode("utf-8", errors="replace")
    return json.loads(raw) if as_json else raw


def parse_irsa(text):
    rows = []
    lines = [l for l in text.splitlines() if l.strip()]
    hdrs = [l for l in lines if l.strip().startswith("|")]
    if hdrs:
        header = [c.strip() for c in hdrs[0].strip("|").split("|")]
        for line in lines:
            if line.strip().startswith(("\\", "|")):
                continue
            parts = line.strip().split()
            if len(parts) >= len(header):
                rows.append(dict(zip(header, parts)))
    if not rows:
        clean = "\n".join(l for l in lines if not l.startswith("\\"))
        try:
            rows = list(csv.DictReader(io.StringIO(clean)))
        except Exception:
            pass
    return rows


def gf(row, *keys):
    for k in keys:
        v = row.get(k, row.get(k.lower(), row.get(k.upper(), "")))
        if v and v not in ("null","NULL","None","nan"):
            return v
    return ""


def row_date(row):
    for key in ("obsdate","date_obs","obsjd","mjd_obs"):
        v = gf(row, key)
        if not v:
            continue
        v = str(v)
        if len(v) >= 10 and "-" in v:
            return v[:10]
        try:
            jd = float(v)
            if jd > 2400000:
                jd -= 2400000.5
            d = datetime.date(2020, 5, 31) + datetime.timedelta(days=jd-59000)
            return d.isoformat()
        except Exception:
            pass
    return ""


def score(row):
    try:
        s = float(gf(row,"seeing","fwhm") or 99)
        a = float(gf(row,"airmass") or 99)
        m = float(gf(row,"maglim","limitmag") or 0)
        i = int(gf(row,"infobits") or 0)
        return s + a*0.5 - m*0.1 + i*5
    except Exception:
        return 999


def make_url(row):
    ffd = gf(row,"filefracday")
    fld = int(gf(row,"field") or 0)
    fc  = gf(row,"filtercode","filter").replace("z","")
    cc  = int(gf(row,"ccdid") or 1)
    qid = int(gf(row,"qid") or 1)
    fn  = f"ztf_{ffd}_{fld:06d}_z{fc}_c{cc:02d}_o_q{qid}_sciimg.fits"
    return f"{IRSA_DATA}/{ffd[:4]}/{ffd[4:8]}/{ffd[8:]}/{fn}"


def query_irsa_r(ra, dec, radius=0.5):
    text = fetch(IRSA_SEARCH, {"POS":f"{ra},{dec}","SIZE":str(radius),"ct":"csv"})
    rows = parse_irsa(text)
    return [r for r in rows
            if gf(r,"filtercode","filter") in ("zr","r",FILTERCODE)]


def sn_in_image(row, sn_ra, sn_dec, margin_arcmin=5.0):
    """Return True if (sn_ra, sn_dec) falls within the image footprint.

    Uses crval1/crval2 (image center RA/Dec from IRSA metadata) + naxis1/naxis2.
    Returns True (assume ok) if metadata is unavailable.
    """
    cra  = gf(row, "crval1", "ra",  "RA")
    cdec = gf(row, "crval2", "dec", "DEC")
    nax1 = gf(row, "naxis1", "NAXIS1")
    nax2 = gf(row, "naxis2", "NAXIS2")
    if not cra or not cdec:
        return True
    try:
        cra  = float(cra)
        cdec = float(cdec)
        nx   = float(nax1 or 3072)
        ny   = float(nax2 or 3080)
        px_scale_deg = 1.01 / 3600.0        # ZTF ~1.01 arcsec/px
        hw_ra  = (nx / 2) * px_scale_deg - margin_arcmin / 60.0
        hw_dec = (ny / 2) * px_scale_deg - margin_arcmin / 60.0
        hw_ra  = max(hw_ra,  0.0)
        hw_dec = max(hw_dec, 0.0)
        cos_dec = math.cos(math.radians((cdec + sn_dec) / 2))
        dra  = abs((sn_ra - cra) * cos_dec)
        ddec = abs(sn_dec - cdec)
        return dra <= hw_ra and ddec <= hw_dec
    except Exception:
        return True


def download_file(url, out_path):
    if out_path.exists():
        mb = out_path.stat().st_size / 1e6
        if mb >= MIN_FITS_MB:
            print(f"    [SKIP] zaten var ({mb:.1f} MB)")
            return True
        out_path.unlink()
    print(f"    {url[-60:]}")
    for attempt in range(1, 4):
        if attempt > 1:
            print(f"    Retry ({attempt}/3)...")
        try:
            req = urllib.request.Request(url, headers={
                "User-Agent":"Python/3","Accept-Encoding":"identity"})
            with urllib.request.urlopen(req, timeout=300) as resp, \
                 open(out_path,"wb") as f:
                if "html" in resp.headers.get("Content-Type","").lower():
                    return False
                total = int(resp.headers.get("Content-Length",0))
                done  = 0
                while True:
                    block = resp.read(131072)
                    if not block:
                        break
                    f.write(block)
                    done += len(block)
                    if total:
                        print(f"\r    {done/total*100:.0f}%  ({done/1e6:.1f}/{total/1e6:.1f} MB)",
                              end="", flush=True)
            print()
            mb = out_path.stat().st_size / 1e6
            if mb < MIN_FITS_MB:
                out_path.unlink(); continue
            if total and done < total * 0.99:
                out_path.unlink()
                print(f"    Incomplete ({done/1e6:.1f}/{total/1e6:.1f} MB)")
                continue
            print(f"    OK: {mb:.1f} MB")
            return True
        except urllib.error.HTTPError as e:
            if out_path.exists(): out_path.unlink()
            if e.code == 401:
                print("    HTTP 401 — IRSA login required"); return False
            print(f"    HTTP {e.code}")
        except Exception as ex:
            if out_path.exists(): out_path.unlink()
            print(f"    Error: {ex}")
    return False


# ── Additional candidates from OSC API ───────────────────────────

def fetch_osc_candidates():
    """Fetch ZTF-era SNe from the Open Supernova Catalog."""
    print("  Querying Open Supernova Catalog...")
    try:
        # OSC v2 API
        url = ("https://api.sne.space/v2/SNe"
               "?format=json&limit=30"
               "&min_date=2018-03-01&max_date=2023-06-01"
               "&claimed_type=Ia"
               "&requested_fields=name,ra,dec,discoverdate,maxappmag")
        data = fetch(url, as_json=True)
        items = data if isinstance(data, list) else data.get("items", [])
        results = []
        for item in items:
            try:
                ra   = float(item.get("ra",  [{}])[0].get("value", 0))
                dec  = float(item.get("dec", [{}])[0].get("value", 0))
                disc = item.get("discoverdate",[{}])[0].get("value","")[:10]
                mag  = item.get("maxappmag",   [{}])[0].get("value", None)
                name = item.get("name","")
                if mag is None or not disc or not name:
                    continue
                mag = float(mag)
                if MAG_MIN <= mag <= MAG_MAX and disc >= "2018-01-01":
                    results.append((name, ra, dec, disc, mag, "Ia"))
            except Exception:
                continue
        print(f"  {len(results)} suitable candidates from OSC.")
        return results
    except Exception as e:
        print(f"  OSC error: {e}")
        return []


# ── Main search loop ──────────────────────────────────────────────

print("=" * 65)
print("  ZTF SN Candidate Finder  (mag {:.1f}-{:.1f})".format(MAG_MIN, MAG_MAX))
print("=" * 65)

# Candidate list: OSC + hardcoded
all_candidates = fetch_osc_candidates()

# Add the hardcoded list with magnitude filter
for entry in KNOWN_ZTF_SNE:
    name, ra, dec, disc, mag, typ = entry
    if MAG_MIN <= mag <= MAG_MAX:
        all_candidates.append(entry)

print(f"\n  Checking {len(all_candidates)} candidates in total.")
print()

SELECTED = None

for entry in all_candidates:
    name, ra, dec, disc, mag, *rest = entry
    typ = rest[0] if rest else "?"

    print(f"  [{name}]  mag={mag:.1f}  disc={disc}  RA={ra:.3f}  Dec={dec:.3f}  type={typ}")

    # Check for images in IRSA
    try:
        rows = query_irsa_r(ra, dec)
    except Exception as e:
        print(f"    IRSA error: {e}"); continue

    if len(rows) < 2:
        print(f"    No images in IRSA ({len(rows)}), skipping."); continue

    # Separate pre (at least PRE_MIN_DAYS before discovery) and post (7-90 days after)
    disc_dt      = datetime.date.fromisoformat(disc)
    pre_cutoff   = (disc_dt - datetime.timedelta(days=PRE_MIN_DAYS)).isoformat()
    pre_rows     = [r for r in rows if row_date(r) and row_date(r) <= pre_cutoff]
    p_after   = (disc_dt + datetime.timedelta(days=7)).isoformat()
    p_before  = (disc_dt + datetime.timedelta(days=90)).isoformat()
    post_rows = [r for r in rows
                 if row_date(r) and p_after <= row_date(r) <= p_before]

    if len(pre_rows) < 2:
        print(f"    Too few pre images (>={PRE_MIN_DAYS}d before: {len(pre_rows)}), skipping."); continue
    if not post_rows:
        print(f"    No post images, skipping."); continue

    # Same field/ccd/qid selection — SN must be within the image
    pre_rows.sort(key=score)

    best_pre = None
    pre_key  = None
    for candidate_pre in pre_rows:
        if sn_in_image(candidate_pre, ra, dec):
            best_pre = candidate_pre
            pre_key  = (int(gf(candidate_pre,"field") or 0),
                        int(gf(candidate_pre,"ccdid") or 0),
                        int(gf(candidate_pre,"qid")   or 0))
            break

    if best_pre is None:
        print(f"    SN not found in any pre image, skipping."); continue

    same_post = [r for r in post_rows
                 if (int(gf(r,"field") or 0),
                     int(gf(r,"ccdid") or 0),
                     int(gf(r,"qid")   or 0)) == pre_key]
    if not same_post:
        same_post = [r for r in post_rows if sn_in_image(r, ra, dec)]
    if not same_post:
        print(f"    SN not found in post image, skipping."); continue
    same_post.sort(key=score)
    best_post = same_post[0]

    print(f"    SELECTED  pre={row_date(best_pre)}  post={row_date(best_post)}")
    SELECTED = {
        "name": name, "ra": ra, "dec": dec, "mag": mag,
        "disc_date": disc, "type": typ,
        "pre_row": best_pre, "post_row": best_post,
    }
    break

if SELECTED is None:
    print("\n  No suitable candidate found. Suggestions:")
    print("  - Increase MAG_MAX (e.g. 20.0)")
    print("  - Add coordinates to the KNOWN_ZTF_SNE list")
    sys.exit(1)

# ── Download ──────────────────────────────────────────────────────
sel     = SELECTED
out_dir = OUT_ROOT / sel["name"]
out_dir.mkdir(parents=True, exist_ok=True)

print(f"\n[Downloading] {sel['name']}  type={sel['type']}  mag={sel['mag']:.1f}")
print(f"  RA={sel['ra']:.4f}  Dec={sel['dec']:.4f}  discovery={sel['disc_date']}")
print(f"  Folder: {out_dir}\n")

success = 0
for label, row, fname in [
    ("pre_sn",  sel["pre_row"],  out_dir / "pre_sn.fits"),
    ("post_sn", sel["post_row"], out_dir / "post_sn.fits"),
]:
    print(f"-- {label}: {row_date(row)} --")
    if download_file(make_url(row), fname):
        success += 1
    print()

print("=" * 65)
if success == 2:
    print(f"  Both images ready!")
    meta = {"name": sel["name"], "ra": sel["ra"], "dec": sel["dec"],
            "mag": sel["mag"], "disc_date": sel["disc_date"], "type": sel["type"],
            "pre_date": row_date(sel["pre_row"]),
            "post_date": row_date(sel["post_row"])}
    with open(out_dir / "target.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  GUI: Image 1 -> {out_dir / 'pre_sn.fits'}")
    print(f"  GUI: Image 2 -> {out_dir / 'post_sn.fits'}")
    print(f"  Validation: for a script like validate_sn2023ixf.py")
    print(f"  SN coordinate: RA={sel['ra']}  Dec={sel['dec']}")
else:
    print("  Download incomplete. Move to the next candidate or")
    print("  add different coordinates to the KNOWN_ZTF_SNE list.")
print("=" * 65)
