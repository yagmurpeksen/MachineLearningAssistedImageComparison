"""
photometry.py
=============
Photometry calibration: zero-point (ZP) calculation using the APASS catalog
and multi-catalog V-mag lookup for saturated stars.

Requirements (optional):
  - astropy.wcs   : WCS coordinate transformation
  - astroquery    : Vizier / SIMBAD catalog queries

Dependencies
------------
  - detection.py : bg_subtracted_flux
"""

import numpy as np
from detection import bg_subtracted_flux

# Vizier/SIMBAD network queries — kept short to avoid blocking the main thread
_CATALOG_QUERY_TIMEOUT = 25  # saniye


# ════════════════════════════════════════════════════════════════
#  SATURATED STAR MAGNITUDE LOOKUP
# ════════════════════════════════════════════════════════════════

def _lookup_sat_mags_from_catalog(sat_ra_list, sat_dec_list,
                                   field_ra, field_dec,
                                   field_radius_arcmin=20):
    """
    Multi-catalog V-mag lookup for saturated stars.

    Catalog priority:
      1. APASS  (V ≈ 7–17)
      2. Hipparcos  (V ≈ 2–12, bright stars)
      3. Tycho-2  (VTmag → V approximation)
      4. SIMBAD  (last resort)

    Parameters
    ----------
    sat_ra_list  : list[float]   RA values (degrees)
    sat_dec_list : list[float]   Dec values (degrees)
    field_ra     : float         Field center RA
    field_dec    : float         Field center Dec
    field_radius_arcmin : float  Query radius (arcmin)

    Returns
    -------
    list of (float | None) — V-mag for each star, or None
    """
    if not sat_ra_list:
        return []
    try:
        from astropy.coordinates import SkyCoord
        from astroquery.vizier import Vizier
        from astroquery.simbad import Simbad
        import astropy.units as u
        import warnings
        warnings.filterwarnings('ignore')

        MATCH_ARCSEC = 15

        center      = SkyCoord(ra=field_ra, dec=field_dec, unit='deg')
        star_coords = SkyCoord(ra=sat_ra_list, dec=sat_dec_list, unit='deg')
        found = {}

        def _cross_match(cat_sc, vmag_col):
            idx, sep, _ = star_coords.match_to_catalog_sky(cat_sc)
            for i, (ii, s) in enumerate(zip(idx, sep)):
                if s.arcsec < MATCH_ARCSEC and i not in found:
                    try:
                        v = float(vmag_col[ii])
                        if not np.isnan(v):
                            found[i] = round(v, 2)
                    except Exception:
                        pass

        viz = Vizier(row_limit=1000, timeout=_CATALOG_QUERY_TIMEOUT)

        # 1. APASS
        try:
            viz.columns = ['RAJ2000', 'DEJ2000', 'Vmag']
            raw = viz.query_region(center,
                                   radius=field_radius_arcmin * u.arcmin,
                                   catalog='APASS')
            if raw:
                cat = raw[0]
                try:
                    cat = cat[~cat['Vmag'].data.mask]
                except Exception:
                    pass
                if len(cat) > 0:
                    _cross_match(
                        SkyCoord(ra=np.array(cat['RAJ2000']),
                                 dec=np.array(cat['DEJ2000']), unit='deg'),
                        np.array(cat['Vmag']))
        except Exception:
            pass

        # 2. Hipparcos
        try:
            viz.columns = ['RAJ2000', 'DEJ2000', 'Vmag']
            raw = viz.query_region(center,
                                   radius=field_radius_arcmin * u.arcmin,
                                   catalog='I/239/hip_main')
            if raw:
                cat = raw[0]
                _cross_match(
                    SkyCoord(ra=np.array(cat['RAJ2000']),
                             dec=np.array(cat['DEJ2000']), unit='deg'),
                    np.array(cat['Vmag']))
        except Exception:
            pass

        # 3. Tycho-2
        try:
            viz.columns = ['RAmdeg', 'DEmdeg', 'VTmag']
            raw = viz.query_region(center,
                                   radius=field_radius_arcmin * u.arcmin,
                                   catalog='I/259/tyc2')
            if raw:
                cat = raw[0]
                vt = np.array(cat['VTmag'], dtype=float)
                _cross_match(
                    SkyCoord(ra=np.array(cat['RAmdeg']),
                             dec=np.array(cat['DEmdeg']), unit='deg'),
                    vt - 0.09)   # VTmag → V approximation
        except Exception:
            pass

        # 4. SIMBAD (last resort — slow; used only if APASS+Hipparcos are insufficient)
        if len(found) < len(sat_ra_list):
            try:
                sb = Simbad()
                sb.TIMEOUT = _CATALOG_QUERY_TIMEOUT
                sb.add_votable_fields('flux(V)')
                sb_result = sb.query_region(center,
                                            radius=field_radius_arcmin * u.arcmin)
                if sb_result is not None and 'FLUX_V' in sb_result.colnames:
                    sb_sc = SkyCoord(ra=sb_result['RA'], dec=sb_result['DEC'],
                                     unit=(u.hourangle, u.deg))
                    _cross_match(sb_sc,
                                 np.array(sb_result['FLUX_V'], dtype=float))
            except Exception:
                pass

        return [found.get(i) for i in range(len(sat_ra_list))]

    except Exception:
        return [None] * len(sat_ra_list)


# ════════════════════════════════════════════════════════════════
#  ZERO-POINT CALIBRATION  (APASS + WCS)
# ════════════════════════════════════════════════════════════════

def _try_photometry_zp(header, stars, flux_img, r_inner=6, r_outer=12, max_stars=400):
    """
    Computes photometric zero point using WCS from the FITS header + APASS catalog.

    Formula: V = -2.5 * log10(flux) + ZP

    Method:
      1. Convert star pixel positions to sky coordinates via WCS
      2. Retrieve V-mag of matching stars from APASS
      3. Median of (V_cat - V_inst) → ZP
      4. Robustify with sigma clipping

    Returns
    -------
    (zp: float, n_cal: int)   — on success
    (None, 0)                 — no WCS or insufficient matches
    """
    try:
        from astropy.wcs import WCS, FITSFixedWarning
        from astropy.coordinates import SkyCoord
        from astroquery.vizier import Vizier
        import astropy.units as u
        import warnings
        warnings.filterwarnings('ignore', category=FITSFixedWarning)
        warnings.filterwarnings('ignore')

        if header is None:
            return None, 0
        if not any(k in header for k in ('CRVAL1', 'CTYPE1', 'CD1_1')):
            return None, 0

        wcs = WCS(header)
        nx  = int(header.get('NAXIS1', flux_img.shape[1]))
        ny  = int(header.get('NAXIS2', flux_img.shape[0]))
        cen = wcs.pixel_to_world(nx / 2, ny / 2)

        viz = Vizier(columns=['RAJ2000', 'DEJ2000', 'Vmag'],
                     row_limit=400, timeout=_CATALOG_QUERY_TIMEOUT)
        cat_raw = viz.query_region(cen, radius=12 * u.arcmin, catalog='APASS')
        if not cat_raw or len(cat_raw) == 0:
            return None, 0
        cat = cat_raw[0]
        try:
            cat = cat[~cat['Vmag'].data.mask]
        except Exception:
            pass
        if len(cat) < 5:
            return None, 0

        # Full-field ZTF has tens of thousands of stars → brightest subset is sufficient for APASS ZP
        if len(stars) > max_stars:
            stars = sorted(stars, key=lambda s: s[3], reverse=True)[:max_stars]

        fluxes  = np.array([
            bg_subtracted_flux(flux_img, s[0], s[1], r_inner, r_outer)
            for s in stars
        ])
        inst_mag = -2.5 * np.log10(np.maximum(fluxes, 1.0))

        x_arr = np.array([s[0] for s in stars])
        y_arr = np.array([s[1] for s in stars])
        sky   = wcs.pixel_to_world(x_arr, y_arr)
        fit_sc = SkyCoord(ra=sky.ra.deg, dec=sky.dec.deg, unit='deg')
        cat_sc = SkyCoord(ra=np.array(cat['RAJ2000']),
                          dec=np.array(cat['DEJ2000']), unit='deg')
        idx, sep, _ = fit_sc.match_to_catalog_sky(cat_sc)
        ok = sep.arcsec < 5.0

        if ok.sum() < 5:
            return None, 0

        diff   = np.array(cat['Vmag'])[idx[ok]] - inst_mag[ok]
        med_zp = float(np.median(diff))
        robust = np.abs(diff - med_zp) < 0.5
        if robust.sum() < 3:
            return None, 0
        zp = float(np.median(diff[robust]))
        return zp, int(robust.sum())

    except Exception:
        return None, 0
