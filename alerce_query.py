"""
alerce_query.py
===============
Thin wrapper around the ALeRCE public REST API for ZTF object lookup.

No API key required.  Works offline-gracefully (returns None on any network error).
"""

import requests

_BASE = "https://api.alerce.online/ztf/v1"
_TIMEOUT = 12   # seconds per request


def query_by_position(ra_deg: float, dec_deg: float,
                      radius_arcsec: float = 10.0) -> list | None:
    """
    Cone-search ALeRCE for ZTF objects within *radius_arcsec* of (ra_deg, dec_deg).

    Returns
    -------
    list of dicts — one entry per matched ZTF object:
        oid         : str   — ZTF identifier (e.g. "ZTF20acwlpnm")
        ndet        : int   — number of detections in ZTF alert stream
        firstmjd    : float — first detection (Modified Julian Date)
        lastmjd     : float — last  detection (MJD)
        ra          : float — mean RA  (deg)
        dec         : float — mean Dec (deg)
        sep_arcsec  : float — angular separation from query position
        class_name  : str or None — top classifier class (e.g. "SNIa", "AGN")
        classifier  : str or None — classifier that produced class_name
        probability : float or None — classification probability [0-1]
    None if the request timed out or the network is unavailable.
    """
    try:
        resp = requests.get(
            f"{_BASE}/objects/",
            params={"ra": ra_deg, "dec": dec_deg, "radius": radius_arcsec},
            timeout=_TIMEOUT,
        )
        resp.raise_for_status()
        items = resp.json().get("items", [])
    except requests.exceptions.Timeout:
        return None
    except Exception:
        return None

    import math

    def _sep(ra1, dec1, ra2, dec2):
        """Great-circle separation in arcsec (small-angle)."""
        cos_dec = math.cos(math.radians((dec1 + dec2) / 2.0))
        d_ra  = (ra1 - ra2) * cos_dec
        d_dec = dec1 - dec2
        return math.sqrt(d_ra ** 2 + d_dec ** 2) * 3600.0

    results = []
    for obj in items:
        oid = obj.get("oid", "")
        mra  = obj.get("meanra")
        mdec = obj.get("meandec")
        sep  = _sep(ra_deg, dec_deg, mra or ra_deg, mdec or dec_deg) if mra else None

        entry = {
            "oid":        oid,
            "ndet":       obj.get("ndethist"),
            "firstmjd":   obj.get("firstmjd"),
            "lastmjd":    obj.get("lastmjd"),
            "ra":         mra,
            "dec":        mdec,
            "sep_arcsec": round(sep, 1) if sep is not None else None,
            "class_name": None,
            "classifier": None,
            "probability": None,
        }

        # Fetch top classification probability
        try:
            pr = requests.get(
                f"{_BASE}/objects/{oid}/probabilities",
                timeout=_TIMEOUT,
            )
            if pr.ok and pr.text.strip():
                probs = pr.json()
                if probs:
                    best = min(probs, key=lambda x: x.get("ranking", 9999))
                    entry["class_name"]  = best.get("class_name")
                    entry["probability"] = best.get("probability")
                    entry["classifier"]  = (best.get("classifier_name") or "").split("(")[0].strip()
        except Exception:
            pass

        results.append(entry)

    return results


def mjd_to_date(mjd: float) -> str:
    """Convert MJD to a human-readable UTC date string (YYYY-MM-DD)."""
    try:
        from astropy.time import Time
        return Time(mjd, format="mjd").iso[:10]
    except Exception:
        # Approximate fallback: MJD 51544.5 = 2000-01-01
        days = mjd - 51544.5
        import datetime
        d = datetime.date(2000, 1, 1) + datetime.timedelta(days=days)
        return str(d)
