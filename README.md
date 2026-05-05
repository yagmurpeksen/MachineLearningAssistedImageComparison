# Brightness Comparison Tool

This application is a GUI tool for visualizing brightness differences between two images.

## Requirements

- Python: 3.11+ recommended

## Installation

### Recommended (virtual environment)

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

macOS/Linux:

```bash
source .venv/bin/activate
```

### Install required packages

```bash
pip install -r requirements.txt
```

## Usage

### Option 1: Run as Python Script

1. Run the application:
```bash
python image_comparison_gui.py
```

Notes:
- On some Linux systems, `tkinter` may be missing by default. On Ubuntu/Debian you can install it with:

```bash
sudo apt-get update && sudo apt-get install -y python3-tk
```

### Option 2: Run as Standalone Application (.exe)

1. Build the executable:
```bash
python build_app.py
```
   Or use the batch file:
```bash
build_app.bat
```

2. The executable will be created in the `dist` folder as `BrightnessComparison.exe`
3. You can run `BrightnessComparison.exe` directly without Python installed

## Tests

Run automated tests:

```bash
pytest -q
```

## How to Use

1. Select two images:
   - Click "Select Image 1" to choose the first image
   - Click "Select Image 2" to choose the second image
   - JPEG format is recommended

2. Compare images:
   - Click "Compare" button
   - Results will be displayed automatically

3. Save result (optional):
   - Click "Save Result" to save the difference map

## Features

- Load and compare two images
- Automatic size matching
- Normalization (corrects lighting differences)
- Brightness difference visualization
- Save results

## Supported Formats

- JPEG (.jpg, .jpeg) - Recommended
- PNG (.png)
- BMP (.bmp)
- TIFF (.tiff, .tif)

## Building the Executable

The application can be packaged as a standalone Windows executable (.exe) using PyInstaller:

- **Method 1**: Run `python build_app.py`
- **Method 2**: Run `build_app.bat`

The executable will be created in the `dist` folder and can be distributed without requiring Python installation.

## Data / model files

- Some helper scripts in this repo expect local FITS files under folders like `test_frames/` or `deeptao_data/`.
- The app can load FITS if `astropy` is installed.
- Pretrained model file (if present): `transient_model.joblib`

## Repo hygiene (Git)

This project includes a `.gitignore` that excludes:
- build artifacts (`build/`, `dist/`)
- caches (`__pycache__/`, `.pytest_cache/`)
- large data/output files (`*.fits`, `*.joblib`, `deeptao_data/`, `fits_out/`, `test_frames/`)
