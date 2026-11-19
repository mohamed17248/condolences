"""Pure helpers shared by app.py and the test-suite.

No Streamlit imports — everything here is deterministic and cheap to unit test.
"""

from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd

PLATFORM_MAP = {
    "N": "SNPP", "NPP": "SNPP", "SNPP": "SNPP", "S-NPP": "SNPP", "SUOMI-NPP": "SNPP",
    "1": "NOAA-20", "1.0": "NOAA-20", "N20": "NOAA-20", "NOAA-20": "NOAA-20", "NOAA20": "NOAA-20",
    "2": "NOAA-21", "2.0": "NOAA-21", "N21": "NOAA-21", "NOAA-21": "NOAA-21", "NOAA21": "NOAA-21",
    "T": "Terra", "TERRA": "Terra",
    "A": "Aqua", "AQUA": "Aqua",
}
MODIS_PLATFORMS = {"Terra", "Aqua"}
VIIRS_PLATFORMS = {"SNPP", "NOAA-20", "NOAA-21"}


def normalize_platform(sat: pd.Series) -> pd.Series:
    # Category-level mapping: avoids materializing millions of Python strings
    # on huge country files (Russia ~25M rows).
    if isinstance(sat.dtype, pd.CategoricalDtype):
        sat_cats = list(sat.cat.categories)
        plat_per_cat = [
            PLATFORM_MAP.get(str(c).strip().upper(), "UNKNOWN") for c in sat_cats
        ]
        plat_cats = sorted(set(plat_per_cat) | {"UNKNOWN"})
        unknown = plat_cats.index("UNKNOWN")
        lookup = np.array([plat_cats.index(p) for p in plat_per_cat] or [unknown], dtype=np.int16)
        codes = sat.cat.codes.to_numpy()
        # Missing satellite values (code -1) are labelled UNKNOWN, not dropped
        new_codes = np.where(codes >= 0, lookup[np.clip(codes, 0, None)], unknown).astype(np.int16)
        return pd.Series(
            pd.Categorical.from_codes(new_codes, categories=plat_cats),
            index=sat.index,
        )
    s = sat.astype(str).str.strip().str.upper()
    return s.map(PLATFORM_MAP).fillna("UNKNOWN")


def theil_sen(x, y):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    slopes = [(y[j] - y[i]) / (x[j] - x[i]) for i, j in combinations(range(len(x)), 2)]
    slope = float(np.median(slopes))
    return slope, float(np.median(y - slope * x))


def rel_trend(series: pd.Series, years_full):
    if len(years_full) < 5:
        return None
    y = series.reindex(years_full).fillna(0).to_numpy(float)
    med = float(np.median(y))
    if med <= 0:
        return None
    slope, icpt = theil_sen(years_full, y)
    return slope / med * 100, slope, icpt


def status_for(current: float, hist: pd.Series):
    n = len(hist)
    med = float(hist.median())
    pct = ((hist < current).sum() + 0.5 * (hist == current).sum()) / n * 100
    rank = int((hist > current).sum()) + 1
    if current == 0 and med == 0:
        label, level = "typical (no detections in most years)", "normal"
    elif pct < 25:
        label, level = "below the normal range", "low"
    elif pct <= 75:
        label, level = "within the normal range", "normal"
    elif pct <= 90:
        label, level = "above normal", "elevated"
    else:
        label, level = "unusually high", "high"
    return dict(label=label, level=level, pct=pct, rank=rank, n=n + 1, median=med)


def ordinal(n: int) -> str:
    suffix = "th" if 10 <= n % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


YEAR_LEVELS = [
    (1.5, "much_more", "A much heavier fire year than usual", "Running far above usual so far"),
    (1.2, "more", "More fire than usual", "Running above usual so far"),
    (0.8, "typical", "A typical fire year", "Tracking a typical year so far"),
    (0.6, "less", "Quieter than usual", "Running below usual so far"),
    (0.0, "much_less", "Much quieter than usual", "Running far below usual so far"),
]


def level_for(ratio):
    for lo, key, head, ytd_head in YEAR_LEVELS:
        if ratio >= lo:
            return key, head, ytd_head
    return YEAR_LEVELS[-1][1:]


def size_text(ratio, unit="the usual year"):
    if ratio is None:
        return "with no baseline year to compare"
    if ratio >= 2:
        return f"{ratio:.1f}× {unit}"
    if ratio >= 1.05:
        return f"{(ratio - 1) * 100:.0f}% above {unit}"
    if ratio <= 0.95:
        return f"{(1 - ratio) * 100:.0f}% below {unit}"
    return f"right at {unit}"


def rank_text(rank, n, noun):
    if n >= 3 and rank == 1:
        return f"It is the highest of {n} {noun}."
    if n >= 3 and rank == n:
        return f"It is the lowest of {n} {noun}."
    return f"It ranks {ordinal(rank)} highest of {n} {noun}."


def short_text(ratio):
    if ratio is None:
        return ""
    if abs(ratio - 1) < 0.005:
        return "at the usual level"
    return f"{ratio:.1f}× usual" if ratio >= 2 else f"{(ratio - 1) * 100:+.0f}% vs usual"
