import random
import numpy as np
import tkinter as tk
from tkinter import messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from ml_classifier import (
    ML_AVAILABLE, classifier, extract_deeptao_features,
)


def open_label_window(app):
    """Label brightness changes one by one — card system."""
    events = list(app.last_brightened) + list(app.last_dimmed)
    if not events:
        messagebox.showinfo("Label Events",
                            "No events to label.\nRun Analyze first.")
        return

    n_total  = len(events)
    n_bright = len(app.last_brightened)
    current  = [0]
    labeled  = {}

    win = tk.Toplevel(app.root)
    win.title("Label Events for Training")
    win.geometry("700x560")
    win.configure(bg="#1a1a2e")
    win.resizable(False, False)

    # ── Header ────────────────────────────────────────────────────
    header = tk.Frame(win, bg="#16213e")
    header.pack(fill=tk.X)
    progress_var = tk.StringVar()
    tk.Label(header, textvariable=progress_var,
             font=("Arial", 12, "bold"), bg="#16213e", fg="#ecf0f1"
             ).pack(side=tk.LEFT, padx=16, pady=10)
    status_var = tk.StringVar(value="")
    tk.Label(header, textvariable=status_var,
             font=("Arial", 10), bg="#16213e", fg="#95a5a6"
             ).pack(side=tk.RIGHT, padx=16, pady=10)

    # ── Card area ─────────────────────────────────────────────────
    card = tk.Frame(win, bg="#0f0f23", relief=tk.SUNKEN, bd=2)
    card.pack(fill=tk.BOTH, expand=True, padx=16, pady=8)

    img_holder  = tk.Frame(card, bg="#0f0f23")
    img_holder.pack(pady=8)
    info_holder = tk.Frame(card, bg="#0f0f23")
    info_holder.pack(pady=4)

    win._last_canvas = None  # canvas referansını burada sakla (GC thread safety)

    # ── Bottom buttons ────────────────────────────────────────────
    btn_bar = tk.Frame(win, bg="#1a1a2e")
    btn_bar.pack(fill=tk.X, padx=16, pady=10)

    def _render(idx):
        win._last_canvas = None  # eski canvas'ı main thread'de serbest bırak
        for w in img_holder.winfo_children():  w.destroy()
        for w in info_holder.winfo_children(): w.destroy()

        r         = events[idx]
        is_bright = r in app.last_brightened
        star_num  = idx + 1 if is_bright else (idx - n_bright + 1)
        star_kind = "Brightened ↑" if is_bright else "Dimmed ↓"
        color     = "#ff6b35" if is_bright else "#74b9ff"
        sign      = "+" if is_bright else ""
        chg       = r.get("change_pct", 0.0)

        progress_var.set(f"Event {idx+1} / {n_total}")
        done  = sum(1 for v in labeled.values() if v == 0)
        trans = sum(1 for v in labeled.values() if v == 1)
        status_var.set(f"Transient: {trans}  |  Artifact: {done}  |  Remaining: {n_total-len(labeled)}")

        # ── Images ──
        try:
            cx, cy       = int(r["x1"]), int(r["y1"])
            h_img, w_img = app.img1_n.shape

            fig, (ax_map, ax_zoom) = plt.subplots(
                1, 2, figsize=(6.5, 3.0),
                facecolor="#0f0f23",
                gridspec_kw={"width_ratios": [1.4, 1]})

            ax_map.imshow(app.img1_n, cmap=app.cmap.get(), vmin=0, vmax=255)
            ax_map.plot(cx, cy, "o", color=color,
                        markersize=14, markerfacecolor="none",
                        markeredgewidth=2.5)
            ax_map.plot(cx, cy, "+", color=color,
                        markersize=16, markeredgewidth=2)
            ax_map.set_facecolor("#0f0f23"); ax_map.axis("off")
            ax_map.set_title(f"#{star_num} {star_kind}  —  position in full image",
                             color=color, fontsize=9, fontweight="bold", pad=4)

            half = 55
            x0 = max(0, cx-half); x1 = min(w_img, cx+half)
            y0 = max(0, cy-half); y1 = min(h_img, cy+half)
            ctx      = app.img1_n[y0:y1, x0:x1]
            lcx, lcy = cx-x0, cy-y0
            ax_zoom.imshow(ctx, cmap=app.cmap.get(), vmin=0, vmax=255)
            ax_zoom.add_patch(plt.Circle((lcx, lcy), 12,
                              color=color, fill=False, lw=2.2))
            ax_zoom.axvline(lcx, color=color, lw=0.8, alpha=0.5)
            ax_zoom.axhline(lcy, color=color, lw=0.8, alpha=0.5)
            ax_zoom.set_facecolor("#0f0f23"); ax_zoom.axis("off")
            ax_zoom.set_title("Close-up view", color="#aaa", fontsize=8, pad=4)

            fig.tight_layout(pad=0.3)
            cv_widget = FigureCanvasTkAgg(fig, img_holder)
            cv_widget.draw()
            cv_widget.get_tk_widget().pack()
            plt.close(fig)
            win._last_canvas = cv_widget  # referansı tut, background thread GC etmesin
        except Exception:
            tk.Label(img_holder, text="Image could not be loaded",
                     bg="#0f0f23", fg="gray").pack()

        # ── Info row ──
        tk.Label(info_holder,
                 text=f"Change: {sign}{chg:.0f}%",
                 font=("Arial", 11, "bold"),
                 bg="#0f0f23", fg=color).pack(side=tk.LEFT, padx=20)
        tk.Label(info_holder,
                 text=f"Position: ({r['x1']:.0f}, {r['y1']:.0f})",
                 font=("Arial", 10),
                 bg="#0f0f23", fg="#7f8c8d").pack(side=tk.LEFT, padx=20)
        if "pt_label" in r:
            pt_lbl = "Transient" if r["pt_label"] == 1 else "Non-transient"
            pt_pct = int(r.get("pt_prob", 0.0) * 100)
            ml_col = "#2ecc71" if r["pt_label"] == 1 else "#95a5a6"
            tk.Label(info_holder,
                     text=f"Model: {pt_lbl} {pt_pct}%",
                     font=("Arial", 10, "bold"),
                     bg="#0f0f23", fg=ml_col).pack(side=tk.LEFT, padx=20)
        else:
            ml = r.get("ml_score", 0.0)
            tk.Label(info_holder,
                     text=f"Anomaly: {ml:.2f}",
                     font=("Arial", 10),
                     bg="#0f0f23", fg="#95a5a6").pack(side=tk.LEFT, padx=20)

    def _label(lbl):
        idx = current[0]
        r   = events[idx]
        try:
            img_early = app.fits_raw1 if app.fits_raw1 is not None else app.img1_n.astype(np.float32)
            img_late  = app.img2_n.astype(np.float32)

            match_idx = r.get("idx", -1)
            if (match_idx >= 0 and match_idx < len(app.last_matched)
                    and app.last_stars1 and app.last_stars2):
                i, j = app.last_matched[match_idx][0], app.last_matched[match_idx][1]
                cx_e, cy_e = app.last_stars1[i][0], app.last_stars1[i][1]
                cx_l, cy_l = app.last_stars2[j][0], app.last_stars2[j][1]
            else:
                cx_e = cx_l = int(r.get("x2", r.get("x1", 0)))
                cy_e = cy_l = int(r.get("y2", r.get("y1", 0)))

            fv = extract_deeptao_features(img_early, cx_e, cy_e,
                                          img_late,  cx_l, cy_l)
            fv = np.nan_to_num(fv, nan=0.0, posinf=0.0, neginf=0.0)
            classifier.add_sample(fv, lbl)
        except Exception as e:
            messagebox.showerror("Feature Error", str(e)); return

        labeled[idx] = lbl

        vals = list(labeled.values())
        if 1 in vals:
            app.btn_train.config(state=tk.NORMAL)

        next_idx = idx + 1
        if next_idx >= n_total:
            win._last_canvas = None          # canvas'ı main thread'de serbest bırak
            for w in img_holder.winfo_children():  w.destroy()
            for w in info_holder.winfo_children(): w.destroy()
            t = sum(1 for v in labeled.values() if v == 1)
            a = sum(1 for v in labeled.values() if v == 0)
            progress_var.set(f"Done!  {n_total} events labeled")
            status_var.set(f"Transient: {t}  |  Artifact: {a}")
            tk.Label(img_holder,
                     text=f"✓ All events labeled\n\nTransient: {t}   Artifact: {a}\n\n"
                          f"{'You can click Train Model.' if (t>=2 and a>=2) else 'At least 2 transients + 2 artifacts required.'}",
                     font=("Arial", 13), bg="#0f0f23",
                     fg="#2ecc71" if (t>=2 and a>=2) else "#e74c3c",
                     justify="center").pack(expand=True, pady=40)
            btn_transient.config(state=tk.DISABLED)
            btn_artifact.config(state=tk.DISABLED)
            return

        current[0] = next_idx
        _render(next_idx)

    # ── Buttons ───────────────────────────────────────────────────
    btn_artifact  = tk.Button(btn_bar, text="Artifact  ✗",
                              command=lambda: _label(0),
                              font=("Arial", 13, "bold"),
                              bg="#e74c3c", fg="white",
                              padx=30, pady=12, cursor="hand2",
                              relief=tk.RAISED, bd=3)
    btn_artifact.pack(side=tk.LEFT, padx=8)

    btn_transient = tk.Button(btn_bar, text="Transient  ✓",
                              command=lambda: _label(1),
                              font=("Arial", 13, "bold"),
                              bg="#27ae60", fg="white",
                              padx=30, pady=12, cursor="hand2",
                              relief=tk.RAISED, bd=3)
    btn_transient.pack(side=tk.RIGHT, padx=8)

    tk.Button(btn_bar, text="Skip →",
              command=lambda: _label_skip(),
              font=("Arial", 11), bg="#2c3e50", fg="white",
              padx=16, pady=12, cursor="hand2").pack(side=tk.LEFT, padx=4)

    def _label_skip():
        next_idx = current[0] + 1
        if next_idx >= n_total:
            win.destroy(); return
        current[0] = next_idx
        _render(next_idx)

    _render(0)


def train_model(app):
    """Train the Random Forest on all labeled samples."""
    if not ML_AVAILABLE:
        messagebox.showerror("ML Unavailable", "scikit-learn is not installed.")
        return

    vals = [s[1] for s in classifier._samples]
    n_transient = vals.count(1)
    n_artifact  = vals.count(0)

    if n_transient < 1:
        messagebox.showwarning(
            "Not Enough Samples",
            "At least 1 Transient label is required.")
        return

    auto_artifact = 0
    if n_artifact == 0 and app.img1_n is not None:
        img_early = (app.fits_raw1 if app.fits_raw1 is not None
                     else app.img1_n.astype(np.float32))
        img_late  = app.img2_n.astype(np.float32)
        stationary = getattr(app, "_last_stationary", [])
        candidates = stationary[:]
        random.shuffle(candidates)
        for r in candidates[:n_transient * 2]:
            try:
                match_idx = r.get("idx", -1)
                if (match_idx >= 0 and match_idx < len(app.last_matched)
                        and app.last_stars1 and app.last_stars2):
                    i, j = app.last_matched[match_idx][0], app.last_matched[match_idx][1]
                    cx_e, cy_e = app.last_stars1[i][0], app.last_stars1[i][1]
                    cx_l, cy_l = app.last_stars2[j][0], app.last_stars2[j][1]
                else:
                    cx_e = cx_l = int(r.get("x2", r.get("x1", 0)))
                    cy_e = cy_l = int(r.get("y2", r.get("y1", 0)))
                fv = extract_deeptao_features(img_early, cx_e, cy_e,
                                              img_late,  cx_l, cy_l)
                fv = np.nan_to_num(fv, nan=0.0, posinf=0.0, neginf=0.0)
                classifier.add_sample(fv, 0)
                auto_artifact += 1
            except Exception:
                continue

    vals = [s[1] for s in classifier._samples]
    n_total     = len(vals)
    n_transient = vals.count(1)
    n_artifact  = vals.count(0)

    if n_artifact == 0:
        messagebox.showwarning(
            "Not Enough Samples",
            "No artifact samples found.\n"
            "Training cannot proceed without stationary stars either.")
        return

    from image_comparison_gui import USER_MODEL_PATH

    app.btn_train.config(state=tk.DISABLED)
    app.root.update()

    try:
        classifier.train()
    except Exception as e:
        app.btn_train.config(state=tk.NORMAL)
        messagebox.showerror("Training Error", str(e))
        return

    app.btn_train.config(state=tk.NORMAL)

    acc     = classifier.cross_val_accuracy()
    acc_str = f"{acc*100:.1f}%" if acc is not None else "N/A (too few samples for CV)"

    try:
        classifier.save(str(USER_MODEL_PATH))
        save_msg = f"\nSaved: {USER_MODEL_PATH.name}"
    except Exception as e:
        save_msg = f"\n(Save failed: {e})"

    auto_msg = (f"\n(+{auto_artifact} artifacts added automatically — stationary stars)"
                if auto_artifact > 0 else "")

    messagebox.showinfo(
        "Model Trained",
        f"Model trained!\n"
        f"Samples: {n_total}  ({n_transient} transient, {n_artifact} artifact)"
        f"{auto_msg}\n"
        f"Cross-val accuracy: {acc_str}"
        + save_msg)

    if app.last_brightened is not None and app.img1_n is not None:
        img_early = app.fits_raw1 if app.fits_raw1 is not None else app.img1_n.astype(np.float32)
        img_late  = app.img2_n.astype(np.float32)
        for r in app.last_brightened + app.last_dimmed:
            try:
                match_idx = r.get("idx", -1)
                if (match_idx >= 0 and match_idx < len(app.last_matched)
                        and app.last_stars1 and app.last_stars2):
                    i, j = app.last_matched[match_idx][0], app.last_matched[match_idx][1]
                    cx_e, cy_e = app.last_stars1[i][0], app.last_stars1[i][1]
                    cx_l, cy_l = app.last_stars2[j][0], app.last_stars2[j][1]
                else:
                    cx_e = cx_l = int(r.get("x2", r.get("x1", 0)))
                    cy_e = cy_l = int(r.get("y2", r.get("y1", 0)))
                fv = extract_deeptao_features(img_early, cx_e, cy_e,
                                              img_late,  cx_l, cy_l)
                fv = np.nan_to_num(fv, nan=0.0, posinf=0.0, neginf=0.0)
                lbl, prob = classifier.predict(fv)
                r["ml_label"] = lbl
                r["ml_prob"]  = prob
            except Exception:
                r["ml_label"] = 0
                r["ml_prob"]  = 0.0

        try:
            app._draw(app.img1_n, app.img2_n,
                      app.last_stars1, app.last_stars2,
                      getattr(app, "_last_moving", []),
                      getattr(app, "_last_stationary", []),
                      app.last_brightened, app.last_dimmed,
                      getattr(app, "_last_unm1", []),
                      getattr(app, "_last_unm2", []),
                      app.last_matched,
                      getattr(app, "_last_med_dx", 0.0),
                      getattr(app, "_last_med_dy", 0.0),
                      getattr(app, "_last_thr_r", 1.0),
                      getattr(app, "_last_residuals", np.array([])),
                      getattr(app, "_last_changes_arr", np.array([])),
                      app.mode.get())
        except Exception:
            pass
