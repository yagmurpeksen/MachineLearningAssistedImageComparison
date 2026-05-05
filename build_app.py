"""
Build script for creating standalone executable
Run: python build_app.py
"""
import PyInstaller.__main__
import os

# PyInstaller parametreleri
args = [
    'image_comparison_gui.py',
    '--name=BrightnessComparison',
    '--onefile',  # Tek dosya olarak paketle
    '--windowed',  # Konsol penceresi gösterme
    '--clean',     # Önceki build'i temizle
    '--noconfirm', # Onay istemeden üzerine yaz
    '--hidden-import=tkinter',
    '--hidden-import=PIL',
    '--hidden-import=matplotlib',
    '--hidden-import=cv2',
    '--hidden-import=numpy',
    '--collect-all=matplotlib',
    '--collect-all=PIL',
    '--collect-all=tkinter',
]

print("Building Brightness Comparison Application...")
print("This may take a few minutes...")
print()

try:
    PyInstaller.__main__.run(args)
    print()
    print("=" * 50)
    print("Build completed successfully!")
    print("=" * 50)
    print(f"Executable location: {os.path.join('dist', 'BrightnessComparison.exe')}")
    print()
except Exception as e:
    print(f"Error during build: {e}")
    input("Press Enter to exit...")

