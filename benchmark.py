"""
benchmark.py
============
Sistem performans metriklerini ölçer ve rapor grafikleri üretir.

Kullanim:
    python benchmark.py
    python benchmark.py --no-alerce   # ALeRCE API testini atla
"""

import argparse
import os
import sys
import time
import threading
import statistics
import warnings
import traceback

import numpy as np
import psutil
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ── Kaynak izleyici ──────────────────────────────────────────────────────────

class ResourceMonitor:
    """Arka planda CPU ve RAM örnekler."""

    def __init__(self, interval=0.2):
        self.interval = interval
        self.cpu = []
        self.ram = []
        self._stop = threading.Event()
        self._proc = psutil.Process()

    def start(self):
        self._stop.clear()
        self._t = threading.Thread(target=self._run, daemon=True)
        self._t.start()

    def stop(self):
        self._stop.set()
        self._t.join()

    def _run(self):
        while not self._stop.is_set():
            self.cpu.append(psutil.cpu_percent(interval=None))
            self.ram.append(self._proc.memory_info().rss / 1024 / 1024)
            time.sleep(self.interval)

    def summary(self):
        return {
            "cpu_mean": statistics.mean(self.cpu) if self.cpu else 0,
            "cpu_max":  max(self.cpu) if self.cpu else 0,
            "ram_mean": statistics.mean(self.ram) if self.ram else 0,
            "ram_max":  max(self.ram) if self.ram else 0,
        }


# ── FITS pipeline benchmark ───────────────────────────────────────────────────

def bench_fits_pipeline(fits_pairs, runs=3):
    """Her FITS çifti için pipeline süresini ve kaynak kullanımını ölçer."""
    from fits_loader import load_fits_image
    from detection import (
        detect_stars_dao, star_affine_align, match_stars_mutual,
        detect_brightness_outliers, filter_stars_by_snr, normalize_pair_joint,
        filter_stars_from_border, match_detection_depth, DAO_AVAILABLE,
    )
    import cv2 as cv

    if not DAO_AVAILABLE:
        print("  [UYARI] DAOStarFinder yok, blob tespiti kullanılacak")

    results = []
    for label, p1, p2 in fits_pairs:
        if not (os.path.exists(p1) and os.path.exists(p2)):
            print(f"  [ATLA] {label}: dosya bulunamadı")
            continue

        times = []
        mon = ResourceMonitor()
        mon.start()

        for _ in range(runs):
            t0 = time.perf_counter()

            img1, _, _, _ = load_fits_image(p1)
            img2, _, _, _ = load_fits_image(p2)

            h, w = img1.shape[:2]
            img2r = cv.resize(img2, (w, h))
            img1n, img2n = normalize_pair_joint(img1, img2r)

            flux1 = img1.astype(np.float32)
            flux2 = img2r.astype(np.float32)

            s1 = detect_stars_dao(flux1, fwhm=3.0, sigma_thresh=5.0)
            s2 = detect_stars_dao(flux2, fwhm=3.0, sigma_thresh=5.0)

            s1 = filter_stars_from_border(s1, w, h, 50)
            s2 = filter_stars_from_border(s2, w, h, 50)
            s1, s2 = match_detection_depth(s1, s2, max_ratio=3.0)

            s1f, _, _ = filter_stars_by_snr(s1, img1n, 1.0, 4, 8)
            s2f, _, _ = filter_stars_by_snr(s2, img2n, 1.0, 4, 8)

            _, H, _ = star_affine_align(img1n, img2n, stars1=s1f, stars2=s2f,
                                        coarse_radius=25)

            matched, um1, um2 = match_stars_mutual(s1f, s2f, max_distance=8)

            detect_brightness_outliers(
                matched, s1f, s2f, flux1, flux2,
                r_inner=4, r_outer=8,
                flux_ratio_thr=1.5, min_flux=50.0,
            )

            times.append(time.perf_counter() - t0)

        mon.stop()
        mon_res = mon.summary()
        entry = {
            "label":  label,
            "mean_s": statistics.mean(times),
            "std_s":  statistics.stdev(times) if len(times) > 1 else 0,
            "min_s":  min(times),
            "max_s":  max(times),
            **mon_res,
        }
        results.append(entry)
        print(f"  {label}: {entry['mean_s']:.2f}s ± {entry['std_s']:.2f}s  "
              f"CPU={entry['cpu_mean']:.1f}%  RAM={entry['ram_max']:.0f}MB")

    return results


# ── ALeRCE API benchmark ─────────────────────────────────────────────────────

def bench_alerce(coords, runs=3):
    """ALeRCE cone search API yanıt sürelerini ölçer."""
    try:
        from alerce_query import query_by_position
    except ImportError:
        print("  [ATLA] alerce_query.py bulunamadı")
        return []

    results = []
    for label, ra, dec in coords:
        times = []
        errors = 0
        for _ in range(runs):
            t0 = time.perf_counter()
            try:
                query_by_position(ra, dec, radius_arcsec=10.0)
                times.append(time.perf_counter() - t0)
            except Exception:
                errors += 1
                times.append(None)

        valid = [t for t in times if t is not None]
        results.append({
            "label":    label,
            "mean_s":   statistics.mean(valid) if valid else None,
            "std_s":    statistics.stdev(valid) if len(valid) > 1 else 0,
            "min_s":    min(valid) if valid else None,
            "max_s":    max(valid) if valid else None,
            "error_rate": errors / runs,
        })
        if valid:
            print(f"  {label}: {statistics.mean(valid):.2f}s ± "
                  f"{(statistics.stdev(valid) if len(valid)>1 else 0):.2f}s  "
                  f"hata={errors}/{runs}")
        else:
            print(f"  {label}: tüm istekler başarısız")

    return results


# ── Grafik ───────────────────────────────────────────────────────────────────

def make_report(pipeline_res, alerce_res, out="report_figures/benchmark.png"):
    os.makedirs(os.path.dirname(out), exist_ok=True)

    n_rows = 2
    n_cols = 3
    fig = plt.figure(figsize=(16, 9), facecolor="#0d0d1a")
    gs  = gridspec.GridSpec(n_rows, n_cols, figure=fig,
                            hspace=0.45, wspace=0.35)

    dark  = "#1a1a2e"
    blue  = "#3498db"
    green = "#2ecc71"
    red   = "#e74c3c"
    gold  = "#f39c12"
    white = "#ecf0f1"
    gray  = "#7f8c8d"

    def ax_style(ax, title):
        ax.set_facecolor(dark)
        ax.set_title(title, color=white, fontsize=10, pad=8)
        ax.tick_params(colors=gray)
        ax.xaxis.label.set_color(gray)
        ax.yaxis.label.set_color(gray)
        for sp in ax.spines.values():
            sp.set_edgecolor("#333")

    labels_p = [r["label"] for r in pipeline_res]
    means_p  = [r["mean_s"] for r in pipeline_res]
    stds_p   = [r["std_s"]  for r in pipeline_res]

    # 1) Pipeline analiz süresi
    ax1 = fig.add_subplot(gs[0, 0])
    ax_style(ax1, "FITS Pipeline Analiz Süresi")
    bars = ax1.bar(labels_p, means_p, yerr=stds_p, color=blue,
                   error_kw=dict(ecolor=gold, capsize=4), width=0.5)
    for bar, v in zip(bars, means_p):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                 f"{v:.2f}s", ha="center", va="bottom", color=white, fontsize=8)
    ax1.set_ylabel("Süre (saniye)")
    ax1.tick_params(axis="x", rotation=15)

    # 2) CPU kullanımı
    ax2 = fig.add_subplot(gs[0, 1])
    ax_style(ax2, "CPU Kullanımı (Ortalama / Maks)")
    x = np.arange(len(labels_p))
    w = 0.35
    ax2.bar(x - w/2, [r["cpu_mean"] for r in pipeline_res], w,
            label="Ortalama", color=blue)
    ax2.bar(x + w/2, [r["cpu_max"]  for r in pipeline_res], w,
            label="Maks",      color=red, alpha=0.8)
    ax2.set_xticks(x); ax2.set_xticklabels(labels_p, rotation=15)
    ax2.set_ylabel("CPU %")
    ax2.legend(fontsize=8, facecolor="#16213e", labelcolor=white)

    # 3) RAM kullanımı
    ax3 = fig.add_subplot(gs[0, 2])
    ax_style(ax3, "RAM Kullanımı (Maks)")
    ax3.bar(labels_p, [r["ram_max"] for r in pipeline_res],
            color=green, width=0.5)
    ax3.set_ylabel("RAM (MB)")
    ax3.tick_params(axis="x", rotation=15)

    # 4) ALeRCE API yanıt süresi
    ax4 = fig.add_subplot(gs[1, 0])
    ax_style(ax4, "ALeRCE API Yanıt Süresi")
    if alerce_res:
        valid_a = [r for r in alerce_res if r["mean_s"] is not None]
        if valid_a:
            al_labels = [r["label"] for r in valid_a]
            al_means  = [r["mean_s"] for r in valid_a]
            al_stds   = [r["std_s"]  for r in valid_a]
            bars_a = ax4.bar(al_labels, al_means, yerr=al_stds,
                             color=gold, error_kw=dict(ecolor=white, capsize=4), width=0.5)
            for bar, v in zip(bars_a, al_means):
                ax4.text(bar.get_x() + bar.get_width()/2,
                         bar.get_height() + 0.02,
                         f"{v:.2f}s", ha="center", va="bottom",
                         color=white, fontsize=8)
            ax4.set_ylabel("Süre (saniye)")
        else:
            ax4.text(0.5, 0.5, "Veri yok", transform=ax4.transAxes,
                     ha="center", color=gray)
    else:
        ax4.text(0.5, 0.5, "Test atlandı\n(--no-alerce)", transform=ax4.transAxes,
                 ha="center", color=gray)

    # 5) ALeRCE hata oranı
    ax5 = fig.add_subplot(gs[1, 1])
    ax_style(ax5, "ALeRCE Hata Oranı")
    if alerce_res:
        err_rates = [r["error_rate"] * 100 for r in alerce_res]
        colors_e  = [red if e > 0 else green for e in err_rates]
        ax5.bar([r["label"] for r in alerce_res], err_rates,
                color=colors_e, width=0.5)
        ax5.set_ylabel("Hata %")
        ax5.set_ylim(0, 105)
        ax5.axhline(0, color=gray, lw=0.5)
    else:
        ax5.text(0.5, 0.5, "Test atlandı", transform=ax5.transAxes,
                 ha="center", color=gray)

    # 6) Özet tablo
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.set_facecolor(dark)
    ax6.axis("off")
    ax6.set_title("Özet", color=white, fontsize=10, pad=8)

    rows = []
    for r in pipeline_res:
        rows.append([r["label"], f"{r['mean_s']:.2f}s",
                     f"{r['cpu_mean']:.1f}%", f"{r['ram_max']:.0f}MB"])
    if alerce_res and any(r["mean_s"] for r in alerce_res):
        for r in alerce_res:
            if r["mean_s"]:
                rows.append([f"ALeRCE {r['label']}", f"{r['mean_s']:.2f}s",
                             "—", "—"])

    if rows:
        col_labels = ["Test", "Süre", "CPU", "RAM"]
        tbl = ax6.table(cellText=rows, colLabels=col_labels,
                        loc="center", cellLoc="center")
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(8)
        for (row, col), cell in tbl.get_celld().items():
            cell.set_facecolor("#16213e" if row == 0 else dark)
            cell.set_text_props(color=white)
            cell.set_edgecolor("#333")

    fig.suptitle("Performans Benchmark — Astronomical Transient Detection Tool",
                 color=white, fontsize=13, y=0.98)

    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="#0d0d1a")
    print(f"\nGrafik kaydedildi: {out}")


# ── Ana akış ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-alerce", action="store_true",
                        help="ALeRCE API testini atla")
    parser.add_argument("--runs", type=int, default=3,
                        help="Her test kaç kez tekrarlansin (varsayilan: 3)")
    args = parser.parse_args()

    print("=" * 60)
    print("  Astronomical Transient Detection — Benchmark")
    print("=" * 60)

    # Test çiftleri
    fits_pairs = [
        ("AT2020xnd\n(22-gün)",
         "sn_target/AT2020xnd_short/pre_sn.fits",
         "sn_target/AT2020xnd_short/post_sn.fits"),
        ("Hareketli\nCisim",
         "moving_object/epoch1.fits",
         "moving_object/epoch2.fits"),
        ("Sentetik\nTest",
         "test_epoch1.fits",
         "test_epoch2.fits"),
    ]

    print(f"\n[1] FITS Pipeline ({args.runs} tekrar):")
    pipeline_res = bench_fits_pipeline(fits_pairs, runs=args.runs)

    alerce_res = []
    if not args.no_alerce:
        print(f"\n[2] ALeRCE API ({args.runs} tekrar):")
        alerce_coords = [
            ("AT2020xnd",  8.050,  36.760),
            ("Küme alanı", 301.5,  35.75),
        ]
        alerce_res = bench_alerce(alerce_coords, runs=args.runs)
    else:
        print("\n[2] ALeRCE API: atlandı (--no-alerce)")

    print("\n[3] Grafik üretiliyor...")
    make_report(pipeline_res, alerce_res)

    print("\n" + "=" * 60)
    print("  ÖZET")
    print("=" * 60)
    for r in pipeline_res:
        print(f"  {r['label'].replace(chr(10),' '):<25} "
              f"{r['mean_s']:.2f}s ± {r['std_s']:.2f}s  "
              f"CPU={r['cpu_mean']:.1f}%  RAM={r['ram_max']:.0f}MB")
    for r in alerce_res:
        if r["mean_s"]:
            print(f"  ALeRCE {r['label']:<19} "
                  f"{r['mean_s']:.2f}s  hata={r['error_rate']*100:.0f}%")


if __name__ == "__main__":
    main()
