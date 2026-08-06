"""
build_app.py
============
Builds a standalone Windows EXE.

Usage:
    python build_app.py

Output:
    dist\StellarAnalysis.exe

Note: transient_model.joblib must be in the same folder as the EXE.
      (Model is not bundled into the EXE — kept external for easy updates.)
"""
import os
import sys
import PyInstaller.__main__

ROOT = os.path.dirname(os.path.abspath(__file__))

args = [
    "image_comparison_gui.py",
    "--name=BrightnessComparison",
    "--onefile",
    "--windowed",
    "--clean",
    "--noconfirm",

    # ── Project modules ──────────────────────────────────────────
    f"--add-data={os.path.join(ROOT, 'detection.py')}{os.pathsep}.",
    f"--add-data={os.path.join(ROOT, 'ml_classifier.py')}{os.pathsep}.",
    f"--add-data={os.path.join(ROOT, 'fits_loader.py')}{os.pathsep}.",
    f"--add-data={os.path.join(ROOT, 'photometry.py')}{os.pathsep}.",
    f"--add-data={os.path.join(ROOT, 'alerce_query.py')}{os.pathsep}.",

    # ── Standard libraries ───────────────────────────────────────
    "--hidden-import=tkinter",
    "--hidden-import=tkinter.ttk",
    "--hidden-import=tkinter.filedialog",
    "--hidden-import=tkinter.messagebox",

    # ── Scientific / ML libraries ────────────────────────────────
    "--hidden-import=numpy",
    "--hidden-import=scipy",
    "--hidden-import=scipy.spatial",
    "--hidden-import=scipy.spatial._ckdtree",
    "--hidden-import=cv2",
    "--hidden-import=matplotlib",
    "--hidden-import=matplotlib.backends.backend_tkagg",
    "--hidden-import=sklearn",
    "--hidden-import=sklearn.ensemble",
    "--hidden-import=sklearn.ensemble._forest",
    "--hidden-import=sklearn.preprocessing",
    "--hidden-import=sklearn.utils._cython_blas",
    "--hidden-import=sklearn.neighbors._partition_nodes",
    "--hidden-import=joblib",
    "--hidden-import=astropy",
    "--hidden-import=astropy.io.fits",
    "--hidden-import=astropy.wcs",
    "--hidden-import=astroquery",
    "--hidden-import=astroquery.vizier",
    "--hidden-import=requests",
    "--hidden-import=alerce_query",
    "--hidden-import=PIL",
    "--hidden-import=PIL.Image",

    # ── Collect all (for heavy packages) ─────────────────────────
    "--collect-all=matplotlib",
    "--collect-all=sklearn",
    "--collect-all=astropy",
    "--collect-all=PIL",
]

print("=" * 55)
print("  Astronomical Transient Detection Tool — EXE Build")
print("=" * 55)
print()
print("Packaging... (this may take 2-5 minutes)")
print()

try:
    PyInstaller.__main__.run(args)
    exe_path = os.path.join(ROOT, "dist", "BrightnessComparison.exe")
    model_src = os.path.join(ROOT, "transient_model.joblib")
    model_dst = os.path.join(ROOT, "dist", "transient_model.joblib")

    print()
    print("=" * 55)
    print("  Build complete!")
    print("=" * 55)
    print(f"  EXE : {exe_path}")
    print()

    # Copy model file to dist/ folder
    if os.path.exists(model_src):
        import shutil
        shutil.copy2(model_src, model_dst)
        print(f"  Model copied: {model_dst}")
    else:
        print("  WARNING: transient_model.joblib not found!")
        print("           Run 'python train_model.py' before launching the EXE.")

    print()
    print("  Share the dist/ folder to distribute.")
    print("  (BrightnessComparison.exe + transient_model.joblib)")

except Exception as e:
    print(f"\nERROR: {e}")
    input("\nPress Enter to continue...")
    sys.exit(1)
