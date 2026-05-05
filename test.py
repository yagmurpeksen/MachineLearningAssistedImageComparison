import numpy as np
from astropy.io import fits

src = r"tao\CSS070320_124616+112555.fits"

hdul = fits.open(src)

# ext=1 tablosunda her görüntünün zamanı var (MJD sütunu)
t = hdul[1].data
mjd = np.array(t["MJD"], dtype=float)

# en erken ve en geç görüntüyü seç (değişim yakalama şansı en yüksek)
i_early = int(np.nanargmin(mjd))
i_late  = int(np.nanargmax(mjd))

# Bu tabloda HDU_Ext, gerçek image extension numarasını verir
ext_early = int(t["HDU_Ext"][i_early])
ext_late  = int(t["HDU_Ext"][i_late])

img_early = hdul[ext_early].data
img_late  = hdul[ext_late].data

fits.writeto(r"tao\img_early.fits", img_early, overwrite=True)
fits.writeto(r"tao\img_late.fits",  img_late,  overwrite=True)

print("Wrote:", r"tao\img_early.fits", "from ext", ext_early, "MJD", mjd[i_early])
print("Wrote:", r"tao\img_late.fits",  "from ext", ext_late,  "MJD", mjd[i_late])
hdul.close()
