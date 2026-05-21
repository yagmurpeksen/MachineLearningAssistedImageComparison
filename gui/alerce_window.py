import threading
import warnings as _w
import tkinter as tk
from tkinter import ttk, messagebox


def open_alerce_window(app):
    """
    For each brightened/dimmed detection, convert pixel coords -> RA/Dec
    (using epoch-2 WCS) and query the ALeRCE ZTF alert broker.
    Results are shown in a scrollable table. Runs in a background thread.
    """
    hdr2 = getattr(app, 'fits_header2', None)
    if hdr2 is None:
        messagebox.showwarning("ALeRCE", "No WCS available — load FITS images first.")
        return

    events = [(r, "Brightened") for r in app.last_brightened] + \
             [(r, "Dimmed")     for r in app.last_dimmed]
    if not events:
        messagebox.showinfo("ALeRCE", "No brightness detections to look up.")
        return

    _wcs = None
    try:
        from astropy.wcs import WCS as _WCS
        with _w.catch_warnings():
            _w.simplefilter("ignore")
            try:
                _wcs = _WCS(hdr2, naxis=2)
            except Exception:
                _wcs = _WCS(hdr2)
        _wcs.all_pix2world(0.0, 0.0, 0)
    except Exception:
        _wcs = None

    if _wcs is None:
        messagebox.showerror("ALeRCE",
            "Could not build WCS from FITS header.\n\n"
            "Make sure both images are ZTF FITS files with valid WCS keywords "
            "(CRVAL1, CRVAL2, CRPIX1, CRPIX2).")
        return

    win = tk.Toplevel(app.root)
    win.title(f"ALeRCE ZTF Lookup  —  {len(events)} detection(s)")
    win.geometry("1000x500")
    win.configure(bg="#1a1a2e")

    hdr_bar = tk.Frame(win, bg="#16213e", pady=6)
    hdr_bar.pack(fill=tk.X)
    status_var = tk.StringVar(value="Querying ALeRCE…  (may take a few seconds per detection)")
    tk.Label(hdr_bar, textvariable=status_var, font=("Arial", 10, "bold"),
             bg="#16213e", fg="#f39c12").pack(side=tk.LEFT, padx=14)
    tk.Label(hdr_bar, text="Source: alerce.online (ZTF alert broker — no API key required)",
             font=("Arial", 9), bg="#16213e", fg="#7f8c8d").pack(side=tk.RIGHT, padx=14)

    cols  = ["#", "Type", "RA (deg)", "Dec (deg)", "Change %",
             "ZTF ID", "Class", "Prob", "N det", "First seen", "Sep (\")", "Score"]
    col_w = [30,  80,     90,       90,        80,
             130,   70,    60,    60,        100,         60,      60]

    style = ttk.Style()
    style.theme_use("clam")
    style.configure("ALeRCE.Treeview",
                    background="#0d0d1a", foreground="#ecf0f1",
                    fieldbackground="#0d0d1a", font=("Courier", 9), rowheight=22)
    style.configure("ALeRCE.Treeview.Heading",
                    background="#16213e", foreground="#1abc9c",
                    font=("Arial", 9, "bold"))
    style.map("ALeRCE.Treeview", background=[("selected", "#2c3e50")])

    frame = tk.Frame(win, bg="#1a1a2e")
    frame.pack(fill=tk.BOTH, expand=True, padx=8, pady=6)

    tree = ttk.Treeview(frame, columns=cols, show="headings", style="ALeRCE.Treeview")
    for col, w in zip(cols, col_w):
        tree.heading(col, text=col)
        tree.column(col, width=w, anchor="center", stretch=False)

    sb = ttk.Scrollbar(frame, orient="vertical", command=tree.yview)
    tree.configure(yscrollcommand=sb.set)
    tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    sb.pack(side=tk.RIGHT, fill=tk.Y)

    def _combined(alerce_s, ml_s):
        """Merge ALeRCE (0/0.5/1) and ML score (0-1) with equal weight."""
        if alerce_s is not None and ml_s is not None:
            return alerce_s * 0.5 + ml_s * 0.5
        if alerce_s is not None:
            return alerce_s
        if ml_s is not None:
            return ml_s
        return 0.0

    def _worker():
        try:
            from alerce_query import query_by_position, mjd_to_date
        except ImportError:
            win.after(0, lambda: status_var.set(
                "Error: alerce_query.py not found in project folder."))
            return

        found_total = 0
        for idx, (r, kind) in enumerate(events):
            px = r.get("x2", r.get("x1", 0))
            py = r.get("y2", r.get("y1", 0))
            try:
                with _w.catch_warnings():
                    _w.simplefilter("ignore")
                    sky = _wcs.all_pix2world(float(px), float(py), 0)
                ra_d, dec_d = float(sky[0]), float(sky[1])
            except Exception:
                win.after(0, lambda i=idx, k=kind: tree.insert(
                    "", "end",
                    values=(i + 1, k, "WCS error", "", "", "", "", "", "", "", "", "—"),
                    tags=("err",)))
                continue

            chg    = r.get("change_pct", 0)
            ml_scr = r.get("ml_score", None)
            win.after(0, lambda: status_var.set(
                f"Querying ALeRCE for detection {idx + 1}/{len(events)}…"))

            matches = query_by_position(ra_d, dec_d, radius_arcsec=10.0)

            if matches is None:
                row = (idx + 1, kind,
                       f"{ra_d:.4f}", f"{dec_d:.4f}", f"{chg:+.1f}%",
                       "Network error", "—", "—", "—", "—", "—", "—")
                win.after(0, lambda rv=row: tree.insert("", "end", values=rv, tags=("err",)))
            elif len(matches) == 0:
                score     = _combined(0.0, ml_scr)
                score_str = f"{score:.2f}"
                row = (idx + 1, kind,
                       f"{ra_d:.4f}", f"{dec_d:.4f}", f"{chg:+.1f}%",
                       "Not in ZTF", "—", "—", "—", "—", "—", score_str)
                win.after(0, lambda rv=row: tree.insert("", "end", values=rv, tags=("miss",)))
            else:
                found_total += 1
                for m in matches:
                    cls  = m.get("class_name") or "—"
                    prob = f"{m['probability']:.2f}" if m.get("probability") is not None else "—"
                    ndet = str(m.get("ndet") or "—")
                    first = mjd_to_date(m["firstmjd"]) if m.get("firstmjd") else "—"
                    sep  = str(m.get("sep_arcsec") or "—")
                    is_bogus = cls.lower() == "bogus" if cls not in ("—", None) else False
                    if cls in ("—", None):
                        alerce_s = 0.5
                    elif is_bogus:
                        alerce_s = 0.0
                    else:
                        alerce_s = 1.0
                    score     = _combined(alerce_s, ml_scr)
                    score_str = f"{score:.2f}"
                    row = (idx + 1, kind,
                           f"{ra_d:.4f}", f"{dec_d:.4f}", f"{chg:+.1f}%",
                           m["oid"], cls, prob, ndet, first, sep, score_str)
                    tag = "bogus" if is_bogus else ("hit" if cls not in ("—", None) else "found")
                    win.after(0, lambda rv=row, tg=tag: tree.insert("", "end", values=rv, tags=(tg,)))

        def _sort_by_score():
            rows = [(tree.set(iid, "Score"), iid) for iid in tree.get_children("")]
            rows.sort(key=lambda x: float(x[0]) if x[0] not in ("—", "") else -1.0,
                      reverse=True)
            for pos, (_, iid) in enumerate(rows):
                tree.move(iid, "", pos)

        win.after(0, _sort_by_score)
        win.after(0, lambda: status_var.set(
            f"Done — {found_total}/{len(events)} matched in ZTF  "
            f"(sorted by combined score: ALeRCE×0.5 + ML×0.5)"))

    tree.tag_configure("hit",   foreground="#2ecc71")
    tree.tag_configure("bogus", foreground="#e67e22")
    tree.tag_configure("found", foreground="#f0e68c")
    tree.tag_configure("miss",  foreground="#7f8c8d")
    tree.tag_configure("err",   foreground="#e74c3c")

    threading.Thread(target=_worker, daemon=True).start()
