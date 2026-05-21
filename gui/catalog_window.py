import os
import csv
import threading
import numpy as np
import cv2 as cv
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from fits_loader import FITS_AVAILABLE
from detection import (
    detect_stars, filter_stars_by_snr,
    bg_subtracted_flux, calculate_star_snr,
)
from photometry import _try_photometry_zp, _lookup_sat_mags_from_catalog


def _catalog_set_status(app, text, color="#3498db"):
    app.root.after(0, lambda: app.status_label.config(text=text, fg=color))


def analyze_catalog(app):
    """Star catalog for a single image (in background — UI does not freeze)."""
    img = app.img1 if app.img1 is not None else app.img2
    if img is None:
        messagebox.showwarning("No image", "Load an image first.")
        return
    if app._catalog_busy:
        return

    raw = app.fits_raw1 if app.fits_raw1 is not None else app.fits_raw2
    path = app.img1_path if app.img1 is not None else app.img2_path
    hdr = (getattr(app, "fits_header1", None)
           if app.img1 is not None else getattr(app, "fits_header2", None))

    job = {
        "img": np.array(img),
        "raw": np.array(raw) if raw is not None else None,
        "path": path,
        "hdr": hdr.copy() if hdr is not None else None,
        "kw": dict(
            pct=app.p_thr_pct.get(),
            min_area=app.p_min_area.get(),
            max_area=app.p_max_area.get(),
            min_peak=app.p_min_peak.get(),
        ),
        "ri": app.p_r_inner.get(),
        "ro": app.p_r_outer.get(),
        "min_snr": app.p_min_snr.get(),
        "cmap": app.cmap.get(),
    }

    app._catalog_busy = True
    app.btn_catalog.config(state=tk.DISABLED)
    _catalog_set_status(app, "Star catalog: detecting stars (background)...", "#3498db")
    threading.Thread(target=_catalog_worker, args=(app, job), daemon=True).start()


def _catalog_worker(app, job):
    try:
        result = _build_catalog_data(app, job)
        app.root.after(0, lambda: _catalog_on_done(app, result))
    except Exception as e:
        app.root.after(0, lambda err=e: _catalog_on_error(app, err))


def _build_catalog_data(app, job):
    """Heavy work: detection + optional APASS (worker thread)."""
    img = job["img"]
    raw = job["raw"]
    hdr = job["hdr"]
    ri, ro = job["ri"], job["ro"]

    img_n = cv.normalize(img, None, 0, 255, cv.NORM_MINMAX)
    flux_img = raw.astype(np.float32) if raw is not None else img_n.astype(np.float32)

    stars_raw = detect_stars(img_n, **job["kw"])
    stars, _, _ = filter_stars_by_snr(stars_raw, img_n, job["min_snr"], ri, ro)

    ra_dec = {}
    has_wcs = False
    _wcs = None
    if hdr is not None and FITS_AVAILABLE:
        try:
            from astropy.wcs import WCS, FITSFixedWarning
            import warnings
            warnings.filterwarnings("ignore", category=FITSFixedWarning)
            _wcs = WCS(hdr)
            if any(k in hdr for k in ("CRVAL1", "CTYPE1", "CD1_1")):
                xs = np.array([s[0] for s in stars])
                ys = np.array([s[1] for s in stars])
                sky = _wcs.pixel_to_world(xs, ys)
                for k in range(len(stars)):
                    ra = float(sky.ra.deg[k])
                    dec = float(sky.dec.deg[k])
                    ra_h = int(ra / 15)
                    ra_m = int((ra / 15 - ra_h) * 60)
                    ra_s = ((ra / 15 - ra_h) * 60 - ra_m) * 60
                    ra_str = f"{ra_h:02d}:{ra_m:02d}:{ra_s:05.2f}"
                    sign = "+" if dec >= 0 else "-"
                    adec = abs(dec)
                    dec_d = int(adec)
                    dec_m = int((adec - dec_d) * 60)
                    dec_s = ((adec - dec_d) * 60 - dec_m) * 60
                    dec_str = f"{sign}{dec_d:02d}:{dec_m:02d}:{dec_s:04.1f}"
                    ra_dec[k] = (ra_str, dec_str)
                has_wcs = True
        except Exception:
            pass

    zp, n_cal = None, 0
    if hdr is not None and FITS_AVAILABLE and has_wcs:
        _catalog_set_status(app, "Star catalog: APASS query (~25s max)...", "#f39c12")
        zp, n_cal = _try_photometry_zp(hdr, stars, flux_img, ri, ro)

    catalog = []
    sat_indices = []
    for idx, s in enumerate(stars):
        cx, cy, area, peak = s
        star_r = max(ri, min(int(np.sqrt(area / np.pi) * 1.2), 40))
        star_ro = star_r + (ro - ri)
        flux = bg_subtracted_flux(flux_img, cx, cy, star_r, star_ro)
        snr = calculate_star_snr(img_n, cx, cy, star_r, star_ro)
        deeply_sat = peak >= 254 and area > 200
        vmag = None
        if zp is not None and flux > 0 and not deeply_sat:
            vmag = round(float(-2.5 * np.log10(max(flux, 1.0)) + zp), 2)
        elif deeply_sat:
            vmag = "sat."
        ra_s, dec_s = ra_dec.get(idx, (None, None))
        catalog.append({
            "id": idx + 1,
            "x": round(cx, 2),
            "y": round(cy, 2),
            "ra": ra_s,
            "dec": dec_s,
            "area": area,
            "peak": peak,
            "flux": round(flux, 1),
            "snr": round(snr, 2),
            "vmag": vmag,
        })
        if deeply_sat and has_wcs and _wcs is not None:
            try:
                _sky = _wcs.pixel_to_world(cx, cy)
                sat_indices.append((
                    len(catalog) - 1,
                    float(_sky.ra.deg),
                    float(_sky.dec.deg),
                ))
            except Exception:
                pass

    if sat_indices and FITS_AVAILABLE and _wcs is not None:
        _catalog_set_status(app, "Star catalog: bright-star lookup...", "#9b59b6")
        try:
            _cen = _wcs.pixel_to_world(
                int(hdr.get("NAXIS1", img_n.shape[1])) / 2,
                int(hdr.get("NAXIS2", img_n.shape[0])) / 2,
            )
            field_ra = float(_cen.ra.deg)
            field_dec = float(_cen.dec.deg)
        except Exception:
            field_ra = float(sat_indices[0][1])
            field_dec = float(sat_indices[0][2])
        sat_vmags = _lookup_sat_mags_from_catalog(
            [x[1] for x in sat_indices],
            [x[2] for x in sat_indices],
            field_ra,
            field_dec,
        )
        for (cat_idx, _, _), vmag_found in zip(sat_indices, sat_vmags):
            if vmag_found is not None:
                catalog[cat_idx]["vmag"] = vmag_found

    if zp is not None:
        valid = [r for r in catalog if isinstance(r["vmag"], float)]
        sat = [r for r in catalog if r["vmag"] == "sat."]
        invalid = [r for r in catalog if r["vmag"] is None]
        valid.sort(key=lambda r: r["vmag"])
        catalog = valid + sat + invalid
    else:
        catalog.sort(key=lambda r: r["flux"], reverse=True)
    for k, r in enumerate(catalog):
        r["id"] = k + 1

    fname = os.path.basename(job["path"]) if job["path"] else "image"
    return {
        "img_n": img_n,
        "catalog": catalog,
        "fname": fname,
        "zp": zp,
        "n_cal": n_cal,
        "has_wcs": has_wcs,
        "cmap": job["cmap"],
        "min_snr": job["min_snr"],
    }


def _catalog_on_done(app, result):
    app._catalog_busy = False
    app.btn_catalog.config(state=tk.NORMAL)
    _show_catalog_window(
        app,
        result["img_n"],
        result["catalog"],
        result["fname"],
        zp=result["zp"],
        n_cal=result["n_cal"],
        has_wcs=result["has_wcs"],
        cmap=result["cmap"],
        min_snr=result["min_snr"],
    )
    status_txt = f"Catalog: {len(result['catalog'])} stars"
    if result["zp"] is not None:
        status_txt += f"  |  ZP={result['zp']:.2f}  ({result['n_cal']} cal stars)"
    else:
        status_txt += "  |  No WCS — V mag unavailable"
    app.status_label.config(text=status_txt, fg="#2ecc71")


def _catalog_on_error(app, err):
    app._catalog_busy = False
    app.btn_catalog.config(state=tk.NORMAL)
    import traceback
    traceback.print_exc()
    messagebox.showerror("Catalog Error", str(err))


def _show_catalog_window(app, img_n, catalog, title, zp=None, n_cal=0,
                          has_wcs=False, cmap=None, min_snr=None):
    has_vmag = zp is not None

    win = tk.Toplevel(app.root)
    win.title(f"Star Catalog — {title}  ({len(catalog)} stars)")
    win.geometry("1300x700" if has_vmag else "1150x680")
    win.configure(bg="#1a1a2e")

    # ── Info bar ──────────────────────────────────────────────────
    info_bar = tk.Frame(win, bg="#16213e", pady=4)
    info_bar.pack(fill=tk.X, padx=8, pady=(6, 0))
    if has_vmag:
        info_txt = f"V mag (APASS)  |  ZP = {zp:.3f}  |  {n_cal} calibration stars"
        info_col = "#2ecc71"
    else:
        info_txt = "APASS calibration unavailable — V magnitude not computed  (flux shown in ADU)"
        info_col = "#e74c3c"
    tk.Label(info_bar, text=info_txt, font=("Courier", 9, "bold"),
             bg="#16213e", fg=info_col).pack(side=tk.LEFT, padx=12)

    # ── Left: image ───────────────────────────────────────────────
    fig_frame = tk.Frame(win, bg="#1a1a2e")
    fig_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=8, pady=8)

    fig, ax = plt.subplots(figsize=(5.5, 5.5), facecolor="#0f0f23")
    img_display = np.flipud(img_n)
    h_img = img_n.shape[0]
    _cmap = cmap if cmap is not None else app.cmap.get()
    _snr = min_snr if min_snr is not None else app.p_min_snr.get()
    ax.imshow(img_display, cmap=_cmap, vmin=0, vmax=255)
    ax.set_facecolor("#0f0f23"); ax.axis("off")
    ax.set_title(f"{len(catalog)} stars  (SNR ≥ {_snr:.1f})",
                 color="white", fontsize=10, fontweight="bold")

    patches = {}
    labels  = {}
    for r in catalog:
        saturated = r["peak"] >= 254
        if saturated:
            radius = min(20, max(8, np.sqrt(r["area"]) * 0.6))
            color  = "yellow"
            lw     = 1.0
        else:
            radius = min(15, max(5, np.sqrt(r["area"]) * 0.5))
            color  = "cyan"
            lw     = 1.0
        dy = h_img - 1 - r["y"]
        circ = plt.Circle((r["x"], dy), radius,
                           color=color, fill=False, lw=lw, alpha=0.85)
        ax.add_patch(circ)
        patches[r["id"]] = (circ, color, radius)
        vmag_v = r.get("vmag")
        if vmag_v == "sat.":
            txt = ax.text(r["x"] + radius + 2, dy - 3,
                          "sat.", color="orange", fontsize=5, alpha=0.95)
        elif has_vmag and isinstance(vmag_v, float):
            txt = ax.text(r["x"] + radius + 2, dy - 3,
                          f"{vmag_v:.2f}", color="cyan",
                          fontsize=5, alpha=0.95)
        else:
            txt = ax.text(r["x"] + radius + 2, dy - 5,
                          str(r["id"]), color=color, fontsize=5, alpha=0.9)
        labels[r["id"]] = txt

    canvas = FigureCanvasTkAgg(fig, fig_frame)
    canvas.draw()
    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    plt.close(fig)

    # ── Right: table + export ─────────────────────────────────────
    right = tk.Frame(win, bg="#1a1a2e", width=500 if has_vmag else 470)
    right.pack(side=tk.RIGHT, fill=tk.BOTH, padx=(0, 8), pady=8)
    right.pack_propagate(False)

    csv_fields = ["id", "ra", "dec", "area", "peak", "flux", "snr"] if has_wcs else \
                 ["id", "x", "y", "area", "peak", "flux", "snr"]
    if has_vmag:
        csv_fields.append("vmag")

    def export_csv():
        p = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV", "*.csv"), ("All", "*.*")])
        if not p:
            return
        with open(p, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=csv_fields, extrasaction="ignore")
            w.writeheader(); w.writerows(catalog)
        messagebox.showinfo("Exported", p)

    tk.Button(right, text="Export CSV", command=export_csv,
              bg="#e67e22", fg="white", font=("Arial", 10, "bold"),
              padx=12, pady=5, cursor="hand2", relief=tk.RAISED, bd=3
              ).pack(pady=(0, 6), anchor="w")

    # Treeview
    style = ttk.Style()
    style.theme_use("clam")
    style.configure("Cat.Treeview", background="#0d0d1a", foreground="#ecf0f1",
                    fieldbackground="#0d0d1a", font=("Courier", 9), rowheight=20)
    style.configure("Cat.Treeview.Heading", background="#16213e",
                    foreground="#00ff88", font=("Arial", 9, "bold"))
    style.map("Cat.Treeview", background=[("selected", "#2c3e50")])

    if has_wcs:
        cols  = ["ID", "RA",  "Dec", "Area", "Peak", "Flux", "SNR"]
        col_w = [35,   110,   110,   45,     50,     75,     50  ]
    else:
        cols  = ["ID", "X (px)", "Y (px)", "Area", "Peak", "Flux", "SNR"]
        col_w = [40,   70,       70,       50,     55,     80,     55  ]
    if has_vmag:
        cols  += ["V mag"]
        col_w += [70]

    tree = ttk.Treeview(right, columns=cols, show="headings", style="Cat.Treeview")
    for col, w in zip(cols, col_w):
        tree.heading(col, text=col)
        tree.column(col, width=w, anchor="center", stretch=False)

    for r in catalog:
        vmag_val = r.get("vmag")
        if vmag_val is None:
            vmag_str = "—"
        elif vmag_val == "sat.":
            vmag_str = "sat."
        else:
            vmag_str = f"{vmag_val:.2f}"
        if has_wcs:
            pos = (r.get("ra") or "—", r.get("dec") or "—")
        else:
            pos = (r["x"], r["y"])
        row = (r["id"],) + pos + (r["area"], r["peak"],
               f"{r['flux']:.0f}", f"{r['snr']:.1f}")
        if has_vmag:
            row += (vmag_str,)
        tree.insert("", "end", values=row)

    sb = ttk.Scrollbar(right, orient="vertical", command=tree.yview)
    tree.configure(yscrollcommand=sb.set)
    tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    sb.pack(side=tk.RIGHT, fill=tk.Y)

    # ── Click: selected star in red, hide others ──────────────────
    _selected = [None]

    def _show_all():
        for sid, (circ, col, _) in patches.items():
            circ.set_visible(True)
            circ.set_edgecolor(col)
            circ.set_linewidth(1.0)
        for txt in labels.values():
            txt.set_visible(True)

    def _on_star_select(event):
        sel = tree.selection()
        if not sel:
            _show_all()
            _selected[0] = None
            canvas.draw_idle()
            return
        vals = tree.item(sel[0], "values")
        try:
            star_id = int(vals[0])
        except (ValueError, IndexError):
            return
        for sid, (circ, col, _) in patches.items():
            circ.set_visible(False)
        for txt in labels.values():
            txt.set_visible(False)
        if star_id in patches:
            circ, orig_col, _ = patches[star_id]
            circ.set_visible(True)
            circ.set_edgecolor("red")
            circ.set_linewidth(2.5)
        if star_id in labels:
            labels[star_id].set_visible(True)
        _selected[0] = star_id
        canvas.draw_idle()

    def _on_tree_click(event):
        region = tree.identify_region(event.x, event.y)
        if region == "nothing":
            tree.selection_remove(tree.selection())
            _show_all()
            _selected[0] = None
            canvas.draw_idle()

    def _on_image_click(event):
        tree.selection_remove(tree.selection())
        _show_all()
        _selected[0] = None
        canvas.draw_idle()

    tree.bind("<<TreeviewSelect>>", _on_star_select)
    tree.bind("<Button-1>", _on_tree_click)
    canvas.get_tk_widget().bind("<Button-1>", _on_image_click)
