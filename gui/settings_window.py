import tkinter as tk
from tkinter import messagebox


def open_settings(app):
    win = tk.Toplevel(app.root)
    win.title("Analysis Parameters")
    win.configure(bg="#1a1a2e"); win.resizable(False, False)
    sections = [
        ("── Star Detection ──────────────────────────────────────────", None, ""),
        ("Detection Percentile:", app.p_thr_pct, "97 = top 3% brightest pixels"),
        ("Min Star Area (px):", app.p_min_area, "Smaller blobs ignored (noise)"),
        ("Max Star Area (px):", app.p_max_area, "Saturated/bright stars can be 500-2000px²"),
        ("── Moving Object Detection ─────────────────────────────────", None, ""),
        ("Min Residual to Flag Moving (px):", app.p_min_res, "Minimum pixel displacement to flag a star as moving (default: 5.0)"),
        ("── Brightness Detection ────────────────────────────────────", None, ""),
        ("Brightness Flux Ratio Threshold:", app.p_flux_ratio_thr, "Recommended: 1.5–3.0  |  Values >10 may miss real transients."),
        ("── SNR Filter ──────────────────────────────────────────────", None, ""),
        ("Min SNR to Keep Star:", app.p_min_snr, "3.0 = star must be at least 3x above noise. Raising keeps only bright stars."),
        ("── DAO Detection (FITS) ────────────────────────────────────", None, ""),
        ("Use DAOStarFinder for FITS:", app.p_use_dao,
         "Use sigma-clipped threshold when FITS is loaded (separate per frame — balances depth)"),
        ("── ML Classifier ──────────────────────────────────────────────", None, ""),
        ("Use ML classifier filter:", app.p_use_ml,
         "Off: all candidates shown. On: Deep-TAO model filters out non-transient candidates."),
    ]
    row = 0
    for lbl, var, tip in sections:
        if var is None:
            tk.Label(win, text=lbl, font=("Arial", 9, "italic"), bg="#1a1a2e",
                     fg="#7f8c8d").grid(row=row, column=0, columnspan=3,
                                        sticky="w", padx=14, pady=(10, 2))
        elif isinstance(var, tk.BooleanVar):
            tk.Label(win, text=lbl, font=("Arial", 10), bg="#1a1a2e",
                     fg="#ecf0f1", anchor="w").grid(row=row, column=0,
                                                     sticky="w", padx=14, pady=3)
            tk.Checkbutton(win, variable=var, bg="#1a1a2e",
                           activebackground="#1a1a2e",
                           fg="#ecf0f1").grid(row=row, column=1, padx=8, sticky="w")
            tk.Label(win, text=tip, font=("Arial", 8), bg="#1a1a2e",
                     fg="#7f8c8d").grid(row=row, column=2, sticky="w", padx=6)
        else:
            tk.Label(win, text=lbl, font=("Arial", 10), bg="#1a1a2e",
                     fg="#ecf0f1", anchor="w").grid(row=row, column=0,
                                                     sticky="w", padx=14, pady=3)
            tk.Entry(win, textvariable=var, width=8,
                     font=("Arial", 10)).grid(row=row, column=1, padx=8)
            tk.Label(win, text=tip, font=("Arial", 8), bg="#1a1a2e",
                     fg="#7f8c8d").grid(row=row, column=2, sticky="w", padx=6)
        row += 1

    def _reset_defaults():
        app.p_thr_pct.set(97); app.p_min_area.set(3); app.p_max_area.set(500)
        app.p_min_peak.set(20); app.p_coarse_r.set(50); app.p_match_r.set(5)
        app.p_iqr_mult.set(3.0); app.p_min_res.set(1.5); app.p_max_mov_frac.set(0.02)
        app.p_r_inner.set(4); app.p_r_outer.set(8); app.p_min_flux.set(50)
        app.p_flux_ratio_thr.set(1.5); app.p_min_snr.set(1.0)
        app.p_dao_fwhm.set(3.0); app.p_dao_sigma.set(5.0)

    def _close_with_warn():
        thr = app.p_flux_ratio_thr.get()
        if thr > 10:
            messagebox.showwarning(
                "High Flux Ratio",
                f"Flux Ratio Threshold is set to {thr:.1f}.\n\n"
                "Values above 10 will miss most real transients.\n"
                "AT2020xnd (SN) was ~18x brighter — recommended: 1.5",
                parent=win)
        win.destroy()

    btn_frame = tk.Frame(win, bg="#1a1a2e")
    btn_frame.grid(row=row, column=0, columnspan=3, pady=14)
    tk.Button(btn_frame, text="Reset to Defaults", command=_reset_defaults,
              bg="#7f8c8d", fg="white", font=("Arial", 10), padx=12, pady=6,
              cursor="hand2").pack(side=tk.LEFT, padx=6)
    tk.Button(btn_frame, text="Close", command=_close_with_warn,
              bg="#2c3e50", fg="white", font=("Arial", 10), padx=16, pady=6,
              cursor="hand2").pack(side=tk.LEFT, padx=6)
