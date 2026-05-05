import argparse
import os
from pathlib import Path

import numpy as np
from astropy.io import fits
from PIL import Image


def png_to_fits(input_path: Path, output_path: Path) -> None:
    with Image.open(input_path) as im:
        # Ensure a single 2D plane for FITS.
        if im.mode in ("I;16", "I"):
            arr = np.array(im)
        else:
            arr = np.array(im.convert("L"))

    hdu = fits.PrimaryHDU(arr)
    hdu.header["ORIGFILE"] = (input_path.name, "Source image filename")
    hdu.header["IMGFMT"] = ("PNG", "Source image format")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    hdu.writeto(output_path, overwrite=True)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Convert PNG images to FITS (2D grayscale)."
    )
    parser.add_argument("inputs", nargs="+", help="Input PNG paths")
    parser.add_argument(
        "-o",
        "--out-dir",
        default="fits_out",
        help="Output directory (default: fits_out)",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    for inp in args.inputs:
        in_path = Path(inp)
        out_path = out_dir / (in_path.stem + ".fits")
        png_to_fits(in_path, out_path)
        print(f"Wrote: {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

import os
from pathlib import Path

import numpy as np
from astropy.io import fits
from PIL import Image


def png_to_fits(src: str, dst: str) -> None:
    img = Image.open(src).convert("L")  # grayscale
    arr = np.asarray(img)

    hdu = fits.PrimaryHDU(arr)
    hdu.header["ORIGIN"] = "PNG->FITS"
    hdu.header["SRCFILE"] = Path(src).name[:68]
    fits.HDUList([hdu]).writeto(dst, overwrite=True)


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    out_dir = project_root / "tao"
    out_dir.mkdir(parents=True, exist_ok=True)

    inputs = [
        (
            Path(
                r"C:\Users\pekse\.cursor\projects\c-Users-pekse-Desktop-cse492\assets\c__Users_pekse_AppData_Roaming_Cursor_User_workspaceStorage_f7371053faeba72902041d812e4fca57_images_81EK25-1-5d1928b9-b4f4-48e5-ad26-c92e6c010b07.png"
            ),
            out_dir / "81EK25-1.fits",
        ),
        (
            Path(
                r"C:\Users\pekse\.cursor\projects\c-Users-pekse-Desktop-cse492\assets\c__Users_pekse_AppData_Roaming_Cursor_User_workspaceStorage_f7371053faeba72902041d812e4fca57_images_81EK25-5-7941eca0-e38f-466f-80e2-99958e8f2980.png"
            ),
            out_dir / "81EK25-5.fits",
        ),
    ]

    for src_path, dst_path in inputs:
        src = str(src_path)
        dst = str(dst_path)
        if not os.path.exists(src):
            raise FileNotFoundError(f"Missing source image: {src}")
        png_to_fits(src, dst)
        print(dst)


if __name__ == "__main__":
    main()

