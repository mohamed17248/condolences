"""Fire Activity Explorer
A harmonized MODIS + VIIRS burning-activity calendar for any area of interest.

Run:    streamlit run app.py
Data:   FIRMS archive downloads (https://firms.modaps.eosdis.nasa.gov/download/)
        saved in a "data" folder next to this script, named after the region:
            data/egypt.csv     or     data/egypt.parquet
        Required columns: latitude, longitude, acq_date
        Used when present: satellite, instrument, confidence, type, acq_time,
                           brightness (MODIS), bright_ti4 (VIIRS)
"""

from __future__ import annotations

import calendar
import hashlib
import inspect
import json
import re
import unicodedata
from html import escape
from itertools import combinations
from pathlib import Path

import folium
import geopandas as gpd
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import shapely
import streamlit as st
from folium.plugins import Draw
from matplotlib import colormaps, colors
from shapely import wkt as shapely_wkt
from shapely.geometry import Polygon, box, mapping, shape
from streamlit_folium import st_folium

st.set_page_config(
    page_title="Fire Activity Explorer",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
NE_PATH = BASE_DIR / "ne_countries.geojson"
if not NE_PATH.exists():
    NE_PATH = Path("ne_countries.geojson")

MODIS_COLOR = "#4f8fc0"
VIIRS_COLOR = "#ef6c33"
OTHER_COLOR = "#8a8f98"
AOI_COLOR = "#3ec1d3"
FONT_STACK = '"Public Sans", "Segoe UI", system-ui, -apple-system, Roboto, sans-serif'
FONT_PLOT = "Public Sans, Segoe UI, system-ui, -apple-system, Roboto, sans-serif"
SENSOR_COLORS = {"MODIS": MODIS_COLOR, "VIIRS": VIIRS_COLOR, "UNKNOWN": OTHER_COLOR}
LEVEL_COLORS = {"much_more": "#ef6c33", "more": "#d9a441", "typical": "#6f94b3",
                "less": "#8a78c0", "much_less": "#6d5aa8", "partial": "#59635f"}

ALIASES = {
    "ivory coast": "côte d'ivoire",
    "east timor": "timor-leste",
    "türkiye": "turkey",
    "turkiye": "turkey",
    "vatican city": "vatican",
    "united states": "united states of america",
    "sao tome": "são tomé and principe",
    "bosnia and herzegovina": "bosnia and herz.",
    "central african republic": "central african rep.",
    "democratic republic of the congo": "dem. rep. congo",
    "republic of the congo": "congo",
    "dominican republic": "dominican rep.",
    "solomon islands": "solomon is.",
    "south sudan": "s. sudan",
    "equatorial guinea": "eq. guinea",
    "dr congo": "dem. rep. congo",
    "drc": "dem. rep. congo",
    "congo republic": "congo",
    "czech republic": "czechia",
}

# (lat_min, lat_max, lon_min, lon_max) for microstates missing from the 110m dataset
MANUAL_REGIONS = {
    "andorra": (42.4, 42.75, 1.4, 1.8),
    "liechtenstein": (47.0, 47.3, 9.5, 9.65),
    "monaco": (43.72, 43.78, 7.4, 7.5),
    "san marino": (43.88, 43.99, 12.4, 12.5),
    "vatican": (41.89, 41.93, 12.44, 12.47),
    "malta": (35.8, 36.1, 14.2, 14.65),
    "singapore": (1.2, 1.5, 103.6, 104.1),
    "bahrain": (25.7, 26.4, 50.3, 50.95),
    "maldives": (-0.7, 7.4, 72.6, 73.9),
    "mauritius": (-20.6, -19.9, 57.3, 57.95),
    "seychelles": (-4.8, -4.5, 55.2, 55.7),
    "comoros": (-13.0, -11.3, 43.1, 45.3),
    "sao tome and principe": (0.0, 2.3, 6.2, 7.6),
    "dominica": (15.15, 15.7, -61.55, -61.15),
    "grenada": (11.9, 12.3, -61.85, -61.55),
    "saint lucia": (13.7, 14.1, -61.1, -60.85),
    "saint vincent and the grenadines": (12.5, 13.45, -61.55, -61.05),
    "antigua and barbuda": (16.95, 17.75, -61.95, -61.6),
    "saint kitts and nevis": (17.2, 17.45, -62.9, -62.55),
    "barbados": (13.0, 13.45, -59.75, -59.35),
    "trinidad and tobago": (10.0, 11.35, -61.95, -60.5),
    "tonga": (-23.0, -15.4, -176.0, -173.0),
    "samoa": (-14.05, -13.4, -172.85, -171.3),
    "vanuatu": (-20.3, -13.0, 166.4, 170.2),
    "kiribati": (-11.5, 5.0, 168.0, 180.0),
    "tuvalu": (-11.2, -5.6, 175.0, 180.0),
    "palau": (2.7, 8.2, 134.0, 134.8),
    "marshall islands": (4.5, 15.0, 160.7, 172.2),
    "nauru": (-0.55, -0.45, 166.85, 166.95),
    "cape verde": (14.8, 17.2, -25.4, -22.6),
}

OVERRIDES = {
    "france": (41.0, 51.5, -5.5, 9.8),
    "united states of america": (24.0, 50.0, -125.0, -66.5),
    "russia": (41.0, 70.0, 27.0, 180.0),
    "fiji": (-20.3, -15.5, 176.5, 180.0),
    "new zealand": (-47.5, -34.0, 166.0, 179.0),
}

WANTED = {
    "latitude", "longitude", "acq_date", "acq_time", "satellite", "instrument",
    "confidence", "brightness", "bright_ti4", "type",
}

PLATFORM_MAP = {
    "N": "SNPP", "NPP": "SNPP", "SNPP": "SNPP", "S-NPP": "SNPP", "SUOMI-NPP": "SNPP",
    "1": "NOAA-20", "1.0": "NOAA-20", "N20": "NOAA-20", "NOAA-20": "NOAA-20", "NOAA20": "NOAA-20",
    "2": "NOAA-21", "2.0": "NOAA-21", "N21": "NOAA-21", "NOAA-21": "NOAA-21", "NOAA21": "NOAA-21",
    "T": "Terra", "TERRA": "Terra",
    "A": "Aqua", "AQUA": "Aqua",
}
MODIS_PLATFORMS = {"Terra", "Aqua"}
VIIRS_PLATFORMS = {"SNPP", "NOAA-20", "NOAA-21"}

# ---- HARMONIZATION START ---------------------------------------------------


def normalize_platform(sat: pd.Series) -> pd.Series:
    s = sat.astype(str).str.strip().str.upper()
    return s.map(PLATFORM_MAP).fillna("UNKNOWN")


def infer_sensor(df: pd.DataFrame, platform: pd.Series) -> pd.Series:
    """Row-level MODIS / VIIRS label. Later rules override earlier ones."""
    sensor = pd.Series("UNKNOWN", index=df.index, dtype=object)
    if "brightness" in df:
        sensor[df["brightness"].notna()] = "MODIS"
    if "bright_ti4" in df:
        sensor[df["bright_ti4"].notna()] = "VIIRS"
    sensor[platform.isin(MODIS_PLATFORMS)] = "MODIS"
    sensor[platform.isin(VIIRS_PLATFORMS)] = "VIIRS"
    if "instrument" in df:
        ins = df["instrument"].astype(str).str.upper()
        sensor[ins.str.contains("MODIS", na=False)] = "MODIS"
        sensor[ins.str.contains("VIIRS", na=False)] = "VIIRS"
    return sensor


@st.cache_data(show_spinner="Reading hotspot file…")
def read_file(path_str: str, mtime: float):
    """Parse a FIRMS file once and keep only what the app needs."""
    path = Path(path_str)
    if path.suffix.lower() == ".parquet":
        df = pd.read_parquet(path)
        df.columns = [c.strip().lower() for c in df.columns]
        df = df[[c for c in df.columns if c in WANTED]].copy()
    else:
        df = pd.read_csv(
            path,
            usecols=lambda c: c.strip().lower() in WANTED,
            dtype={"satellite": str, "instrument": str, "confidence": str},
        )
        df.columns = [c.strip().lower() for c in df.columns]

    df["acq_date"] = pd.to_datetime(df["acq_date"]).dt.normalize()
    for c in ("satellite", "instrument", "confidence"):
        if c in df:
            df[c] = df[c].astype(str)

    subset = [c for c in ("latitude", "longitude", "acq_date", "acq_time", "satellite") if c in df]
    df = df.drop_duplicates(subset=subset)

    if "satellite" in df:
        platform = normalize_platform(df["satellite"])
    else:
        platform = pd.Series("UNKNOWN", index=df.index)
    sensor = infer_sensor(df, platform)

    # MODIS confidence is 0-100, VIIRS is l / n / h. Low = MODIS < 30 or VIIRS 'l'.
    lowconf = pd.Series(False, index=df.index)
    if "confidence" in df:
        c = df["confidence"].str.strip().str.lower()
        lowconf = (pd.to_numeric(c, errors="coerce") < 30) | c.isin(["l", "low"])

    # FIRMS type: 0 vegetation fire, 1 volcano, 2 other static land source, 3 offshore
    nonveg = pd.Series(False, index=df.index)
    if "type" in df:
        t = pd.to_numeric(df["type"], errors="coerce")
        nonveg = t.notna() & (t != 0)

    out = pd.DataFrame({
        "latitude": df["latitude"].astype("float32"),
        "longitude": df["longitude"].astype("float32"),
        "acq_date": df["acq_date"],
        "sensor": sensor.astype("category"),
        "platform": platform.astype("category"),
        "lowconf": lowconf.to_numpy(),
        "nonveg": nonveg.to_numpy(),
    }).reset_index(drop=True)
    flags = {
        "has_conf": "confidence" in df,
        "has_type": "type" in df,
        "has_platform": "satellite" in df,
        "n_raw": len(out),
    }
    return out, flags


@st.cache_data(show_spinner="Harmonizing sensors…")
def build_dataset(path_str, mtime, drop_low, veg_only, viirs_mode, seasonal, min_monthly):
    """Quality filter -> per-satellite normalization -> MODIS->VIIRS scaling -> splice.

    1. Drop low-confidence and non-vegetation detections (optional).
    2. Divide every detection by the number of satellites of that sensor already
       flying on that date (Terra alone before Aqua, SNPP alone before NOAA-20...).
    3. Over the months where MODIS and VIIRS both exist, compute VIIRS / MODIS
       monthly ratios; their median is the scaling factor (optionally per calendar
       month). A bootstrap over months gives a range, and leave-one-year-out
       prediction gives an out-of-sample error.
    4. Use scaled MODIS before the first full VIIRS month and VIIRS after it.
    """
    raw, flags = read_file(path_str, mtime)
    first_seen = raw.groupby("platform", observed=True)["acq_date"].min()

    df = raw
    if drop_low:
        df = df[~df["lowconf"]]
    if veg_only:
        df = df[~df["nonveg"]]
    df = df.drop(columns=["lowconf", "nonveg"]).copy()
    n_after_qc = len(df)

    df["ym"] = df["acq_date"].values.astype("datetime64[M]").astype("datetime64[ns]")
    raw_monthly = (
        df.groupby(["ym", "sensor", "platform"], observed=True).size().reset_index(name="n")
    )

    if viirs_mode == "snpp" and ((df["sensor"] == "VIIRS") & (df["platform"] == "SNPP")).any():
        df = df[(df["sensor"] != "VIIRS") | (df["platform"] == "SNPP")]

    w0 = np.ones(len(df))
    for sensor in ("MODIS", "VIIRS"):
        mask = (df["sensor"] == sensor).to_numpy()
        if not mask.any():
            continue
        plats = [str(p) for p in df.loc[mask, "platform"].unique()]
        firsts = np.sort(first_seen.reindex(plats).to_numpy(dtype="datetime64[ns]"))
        dates = df.loc[mask, "acq_date"].to_numpy(dtype="datetime64[ns]")
        n_active = np.searchsorted(firsts, dates, side="right")
        w0[mask] = 1.0 / np.maximum(n_active, 1)
    df["w0"] = w0

    modis = (df["sensor"] == "MODIS").to_numpy()
    viirs = (df["sensor"] == "VIIRS").to_numpy()
    has_both = modis.any() and viirs.any()

    info = {
        "has_both": has_both, "factor": 1.0, "factor_ci": None,
        "seasonal_map": {m: 1.0 for m in range(1, 13)},
        "split": None, "overlap": pd.DataFrame(), "corr": np.nan,
        "err_single": np.nan, "err_seasonal": np.nan, "err_loyo": np.nan, "n_valid": 0,
        "raw_monthly": raw_monthly, "flags": flags,
        "n_raw": flags["n_raw"], "n_after_qc": n_after_qc,
        "platforms": {s: sorted(map(str, df.loc[df["sensor"] == s, "platform"].unique()))
                      for s in ("MODIS", "VIIRS")},
    }

    f = np.ones(len(df))
    keep = np.ones(len(df), dtype=bool)

    if has_both:
        viirs_first = df.loc[viirs, "acq_date"].min()
        split = viirs_first + pd.offsets.MonthBegin(0)       # first full VIIRS month
        info["split"] = split

        monthly = df.groupby(["ym", "sensor"], observed=True)["w0"].sum().unstack(fill_value=0)
        for s in ("MODIS", "VIIRS"):
            if s not in monthly:
                monthly[s] = 0.0
        ov = monthly.loc[monthly.index >= split, ["MODIS", "VIIRS"]].copy()

        valid = (ov["MODIS"] >= min_monthly) & (ov["VIIRS"] >= min_monthly)
        if valid.sum() < 6:                                   # tiny region: relax
            valid = (ov["MODIS"] > 0) & (ov["VIIRS"] > 0)
        ov["ratio"] = ov["VIIRS"] / ov["MODIS"].replace(0, np.nan)
        ov["valid"] = valid

        factor, seasonal_map = 1.0, {m: 1.0 for m in range(1, 13)}
        if valid.sum() >= 1:
            ratios = ov.loc[valid, "ratio"]
            factor = float(ratios.median())
            by_m = ratios.groupby(ratios.index.month).median()
            cnt = ratios.groupby(ratios.index.month).size()
            seasonal_map = {m: float(by_m[m]) if cnt.get(m, 0) >= 2 else factor for m in range(1, 13)}

            m_arr = ov.loc[valid, "MODIS"].to_numpy(float)
            v_arr = ov.loc[valid, "VIIRS"].to_numpy(float)
            yrs = ov.index[valid].year.to_numpy()
            mons = ov.index[valid].month.to_numpy()
            info["err_single"] = float(np.median(np.abs(m_arr * factor - v_arr) / v_arr))
            fm = np.array([seasonal_map[k] for k in mons])
            info["err_seasonal"] = float(np.median(np.abs(m_arr * fm - v_arr) / v_arr))
            if len(m_arr) >= 3:
                info["corr"] = float(np.corrcoef(m_arr, v_arr)[0, 1])
                rng = np.random.default_rng(0)
                r_arr = v_arr / m_arr
                boots = [np.median(rng.choice(r_arr, len(r_arr))) for _ in range(500)]
                info["factor_ci"] = (float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5)))

            # leave-one-year-out: fit on the other years, predict the held-out year
            errs = []
            for y in np.unique(yrs):
                te = yrs == y
                tr = ~te
                if tr.sum() >= 3:
                    fy = float(np.median(v_arr[tr] / m_arr[tr]))
                    errs.append(float(np.median(np.abs(m_arr[te] * fy - v_arr[te]) / v_arr[te])))
            if errs:
                info["err_loyo"] = float(np.median(errs))

        info.update(factor=factor, seasonal_map=seasonal_map, overlap=ov.reset_index(),
                    n_valid=int(valid.sum()))

        if seasonal:
            lookup = np.array([seasonal_map[m] for m in range(1, 13)])
            months = df["acq_date"].dt.month.to_numpy()
            f[modis] = lookup[months[modis] - 1]
        else:
            f[modis] = factor

        ym = df["ym"]
        keep = (
            (modis & (ym < split).to_numpy())
            | (viirs & (ym >= split).to_numpy())
            | (~modis & ~viirs)
        )

    df["weight"] = df["w0"].to_numpy() * f
    df = df[keep].drop(columns=["w0"])
    df["year"] = df["acq_date"].dt.year
    df["month"] = df["acq_date"].dt.month
    info["n_final"] = len(df)
    return df.reset_index(drop=True), info


def theil_sen(x, y):
    """Robust linear trend: median of pairwise slopes."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    slopes = [(y[j] - y[i]) / (x[j] - x[i]) for i, j in combinations(range(len(x)), 2)]
    slope = float(np.median(slopes))
    return slope, float(np.median(y - slope * x))


# ---- HARMONIZATION END -----------------------------------------------------

# ---- HELPERS START ---------------------------------------------------------


def rel_trend(series: pd.Series, years_full):
    """Theil-Sen trend over complete years -> (% of median per year, slope, intercept)."""
    if len(years_full) < 5:
        return None
    y = series.reindex(years_full).fillna(0).to_numpy(float)
    med = float(np.median(y))
    if med <= 0:
        return None
    slope, icpt = theil_sen(years_full, y)
    return slope / med * 100, slope, icpt


def circle_polygon(lon, lat, radius_m, n=72):
    ang = np.linspace(0, 2 * np.pi, n, endpoint=False)
    dlat = (radius_m / 111320.0) * np.sin(ang)
    dlon = (radius_m / (111320.0 * max(np.cos(np.radians(lat)), 0.05))) * np.cos(ang)
    return Polygon(list(zip(lon + dlon, lat + dlat)))


def drawing_to_polygon(feat):
    """Leaflet.draw feature (polygon, rectangle or circle) -> shapely polygon."""
    g = (feat or {}).get("geometry") or {}
    props = (feat or {}).get("properties") or {}
    if g.get("type") == "Polygon":
        poly = shape(g)
    elif g.get("type") == "Point" and props.get("radius"):
        lon, lat = g["coordinates"][:2]
        poly = circle_polygon(lon, lat, float(props["radius"]))
    else:
        return None
    if not poly.is_valid:
        poly = poly.buffer(0)
    return None if poly.is_empty else poly


def _contains_xy(poly, x, y):
    try:
        return shapely.contains_xy(poly, x, y)
    except AttributeError:                                   # shapely < 2
        from shapely import vectorized
        return vectorized.contains(poly, x, y)


def apply_aoi(df: pd.DataFrame, poly) -> pd.DataFrame:
    minx, miny, maxx, maxy = poly.bounds
    near = df[
        (df["longitude"] >= minx) & (df["longitude"] <= maxx)
        & (df["latitude"] >= miny) & (df["latitude"] <= maxy)
    ]
    if near.empty:
        return near
    inside = _contains_xy(poly, near["longitude"].to_numpy(float), near["latitude"].to_numpy(float))
    return near[inside]


def status_for(current: float, hist: pd.Series):
    """Where does the current period sit among the same days in earlier years?"""
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


def fmt(n) -> str:
    return f"{n:,.0f}"


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
        return "against a usual of none"
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


def year_verdicts(annual_total, modis_share, pivot_cov, full_years, med_by_month, ytd=None):
    """One plain-language verdict per year, rated against the usual (median) complete year."""
    years = sorted(int(y) for y in annual_total.index)
    full = [y for y in full_years if y in annual_total.index]
    rated = len(full) >= 4
    baseline = float(annual_total.loc[full].median()) if rated else None
    has_clim = bool(med_by_month.notna().any()) and float(med_by_month.max()) > 0
    typ_peak = int(med_by_month.idxmax()) if has_clim else None
    clim_total = float(med_by_month.sum()) if has_clim else 0.0

    out, run_dir, run_len, prev = [], 0, 0, None
    for y in years:
        total = float(annual_total.loc[y])
        rec = dict(year=y, kind="partial", total=total, ratio=None, level="partial",
                   headline="Partial year", detail="", short="not rated", rank=None, n=len(full))

        if ytd is not None and y == ytd["year"] and y not in full:
            hist, cur = ytd["hist"], float(ytd["cur"])
            med = float(hist.median())
            ratio = cur / med if med > 0 else None
            if ratio is None:
                level, yhead = ("more", YEAR_LEVELS[1][3]) if cur > 0 else ("typical", YEAR_LEVELS[2][3])
            else:
                level, _, yhead = level_for(ratio)
            rank = int((hist > cur).sum()) + 1
            rec.update(
                kind="ytd", ratio=ratio, level=level, headline=yhead, rank=rank,
                short=(short_text(ratio) + " so far").strip(),
                detail=(f"{fmt(cur)} detections from January 1 to {ytd['label']}, "
                        f"{size_text(ratio, 'the usual for these dates')}. "
                        f"{rank_text(rank, len(hist) + 1, 'years for the same dates')}"),
            )
        elif y in full and rated:
            ratio = total / baseline if baseline > 0 else None
            if ratio is None:
                level, head = ("more", YEAR_LEVELS[1][2]) if total > 0 else ("typical", YEAR_LEVELS[2][2])
            else:
                level, head, _ = level_for(ratio)
            rank = int((annual_total.loc[full] > total).sum()) + 1
            tail = f" ({fmt(baseline)})." if baseline > 0 else "."
            parts = [f"{fmt(total)} detections, {size_text(ratio)}{tail}",
                     rank_text(rank, len(full), "complete years")]

            row = pivot_cov.loc[y].astype(float) if y in pivot_cov.index else None
            if row is not None and row.notna().any() and float(row.max()) > 0 and typ_peak is not None:
                pm = int(row.idxmax())
                shift = ((pm - typ_peak + 6) % 12) - 6
                if shift == 0:
                    timing = "as usual"
                else:
                    timing = f"{abs(shift)} month{'s' if abs(shift) > 1 else ''} {'later' if shift > 0 else 'earlier'} than usual"
                parts.append(f"The season peaked in {calendar.month_name[pm]}, {timing}.")
            if row is not None and has_clim:
                best, best_r = None, 0.0
                for mo in range(1, 13):
                    med, val = med_by_month.get(mo, np.nan), row.get(mo, np.nan)
                    if pd.notna(med) and pd.notna(val) and med > 0 and med >= 0.03 * clim_total:
                        if val / med > best_r:
                            best, best_r = mo, float(val / med)
                if best is not None and best_r >= 1.5:
                    parts.append(f"{calendar.month_name[best]} ran {best_r:.1f}× its usual level.")

            direction = 1 if level in ("more", "much_more") else (-1 if level in ("less", "much_less") else 0)
            if direction != 0 and direction == run_dir and prev == y - 1:
                run_len += 1
            else:
                run_len = 1 if direction != 0 else 0
            run_dir, prev = direction, y
            if run_len >= 3:
                parts.append(f"That is the {ordinal(run_len)} year in a row {'above' if direction > 0 else 'below'} usual.")
            if float(modis_share.get(y, 0.0)) > 0.5:
                parts.append("Mostly estimated from scaled MODIS data, so treat the size as approximate.")
            rec.update(kind="full", ratio=ratio, level=level, headline=head, rank=rank,
                       short=short_text(ratio) or "no usual level", detail=" ".join(parts))
        else:
            valid = pivot_cov.loc[y].dropna() if y in pivot_cov.index else pd.Series(dtype=float)
            first_m = int(valid.index.min()) if len(valid) else None
            if y in full and not rated:
                rec["detail"] = "Fewer than four complete years are loaded, so years are not rated yet."
            elif y == years[0] and first_m and first_m > 1:
                rec["detail"] = f"Data for this year starts in {calendar.month_name[first_m]}, so it is not rated."
            else:
                rec["detail"] = "Only part of this year is covered, so it is not rated."
        out.append(rec)
    return out


def years_summary(vs, region_text, baseline):
    full = [v for v in vs if v["kind"] == "full"]
    if len(full) < 4 or baseline is None:
        return "Load at least four complete years of data to rate each year against the usual one."
    hi = max(full, key=lambda v: v["total"])
    lo = min(full, key=lambda v: v["total"])
    above = sum(v["level"] in ("more", "much_more") for v in full)
    below = sum(v["level"] in ("less", "much_less") for v in full)
    text = (
        f"Across {len(full)} complete years, the usual year in {region_text} has {fmt(baseline)} detections. "
        f"{hi['year']} was the heaviest, at {size_text(hi['ratio'])}, and {lo['year']} the quietest, "
        f"at {size_text(lo['ratio'])}. {above} {'year ran' if above == 1 else 'years ran'} above usual "
        f"and {below} below."
    )
    if any(v["year"] < 2012 for v in full):
        text += " Years before 2012 rest on scaled MODIS data, so their sizes are approximate."
    return text


# ---- HELPERS END -----------------------------------------------------------


def polygon_area_km2(poly) -> float:
    try:
        return float(gpd.GeoSeries([poly], crs="EPSG:4326").to_crs("EPSG:6933").area.iloc[0] / 1e6)
    except Exception:
        return float("nan")


# --------------------------------------------------------------------------
# Regions
# --------------------------------------------------------------------------
@st.cache_data
def load_regions():
    """Return ({name: (lat_min, lat_max, lon_min, lon_max)}, {alias: canonical name})."""
    world = gpd.read_file(NE_PATH)
    name_col = next(c for c in ("NAME", "name", "NAME_EN", "ADMIN") if c in world.columns)
    regs = {}
    for name, geom in zip(world[name_col], world.geometry):
        if geom is None or geom.is_empty:
            continue
        minx, miny, maxx, maxy = geom.bounds
        regs[str(name).lower()] = (miny, maxy, minx, maxx)

    canon = {}
    for alias, target in ALIASES.items():
        if target in regs:
            canon[alias] = target
            regs.setdefault(alias, regs[target])
    for name, bbox in MANUAL_REGIONS.items():
        regs.setdefault(name, bbox)
    regs.update(OVERRIDES)
    return regs, canon


def slug(key: str, canon: dict) -> str:
    s = unicodedata.normalize("NFKD", canon.get(key, key)).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")


def data_stems() -> set:
    if not DATA_DIR.exists():
        return set()
    return {p.stem for p in DATA_DIR.iterdir() if p.suffix.lower() in (".csv", ".parquet")}


def find_data_file(key: str, canon: dict):
    for stem in (slug(key, canon), key.replace(" ", "_")):
        for ext in (".parquet", ".csv"):
            p = DATA_DIR / f"{stem}{ext}"
            if p.exists():
                return p
    return None


# --------------------------------------------------------------------------
# Charts
# --------------------------------------------------------------------------
_PLOT_HAS_WIDTH = "width" in inspect.signature(st.plotly_chart).parameters


def show(fig, key):
    if _PLOT_HAS_WIDTH:
        st.plotly_chart(fig, width="stretch", key=key)
    else:
        st.plotly_chart(fig, use_container_width=True, key=key)


def tidy(fig, title, height=380):
    fig.update_layout(
        title=dict(text=title, x=0, xanchor="left", y=0.98, yanchor="top",
                   yref="container", font=dict(size=14)),
        height=height,
        margin=dict(l=10, r=10, t=88, b=10),
        legend=dict(orientation="h", x=0, xanchor="left", y=1.0, yanchor="bottom",
                    title=None, font=dict(size=11)),
        hoverlabel=dict(font_size=12),
        font=dict(family=FONT_PLOT, size=12),
    )
    fig.update_xaxes(showgrid=False, automargin=True)
    fig.update_yaxes(gridcolor="rgba(128,128,128,.18)", zeroline=False, automargin=True)
    return fig



FIRE_CMAP = colors.LinearSegmentedColormap.from_list(
    "fire_years", [(0.0, "#6d5aa8"), (0.5, "#3a4541"), (0.78, "#d9622b"), (1.0, "#ffb066")])


def stripe_color(ratio):
    t = 0.5 + np.log2(max(float(ratio), 0.05)) / 2.4
    return colors.to_hex(FIRE_CMAP(float(np.clip(t, 0.0, 1.0))))


def render_stripes(vs):
    cells = []
    for v in vs:
        bg = stripe_color(v["ratio"]) if v["ratio"] is not None else "#262d2b"
        tip = escape(f"{v['year']}: {v['headline']}. {v['short']}", quote=True)
        cells.append(f'<div class="stripe" title="{tip}" style="background:{bg}"></div>')
    st.markdown(
        f'<div class="stripes">{"".join(cells)}</div>'
        f'<div class="stripe-axis"><span>{vs[0]["year"]}</span><span>{vs[-1]["year"]}</span></div>',
        unsafe_allow_html=True,
    )


@st.cache_data
def to_csv(key, _df):
    out = _df.drop(columns=["year", "month", "ym"], errors="ignore").copy()
    out["acq_date"] = out["acq_date"].dt.strftime("%Y-%m-%d")
    out = out.rename(columns={"weight": "viirs_equiv_weight"})
    return out.to_csv(index=False).encode()


# --------------------------------------------------------------------------
# Map layers
# --------------------------------------------------------------------------
ESRI = "https://server.arcgisonline.com/ArcGIS/rest/services"
TILES = {
    "Dark": dict(tiles=f"{ESRI}/Canvas/World_Dark_Gray_Base/MapServer/tile/{{z}}/{{y}}/{{x}}",
                 attr="Tiles © Esri — Esri, HERE, Garmin, © OpenStreetMap contributors"),
    "Light": dict(tiles=f"{ESRI}/Canvas/World_Light_Gray_Base/MapServer/tile/{{z}}/{{y}}/{{x}}",
                  attr="Tiles © Esri — Esri, HERE, Garmin, © OpenStreetMap contributors"),
    "Streets": dict(tiles="OpenStreetMap", attr="© OpenStreetMap contributors"),
    "Satellite": dict(tiles=f"{ESRI}/World_Imagery/MapServer/tile/{{z}}/{{y}}/{{x}}",
                      attr="Tiles © Esri — Maxar, Earthstar Geographics"),
    "Blank (offline)": None,
}


def base_map(style: str) -> folium.Map:
    m = folium.Map(location=[20, 0], zoom_start=2, control_scale=True, prefer_canvas=True,
                   tiles=None, min_zoom=2, max_zoom=16)
    cfg = TILES[style]
    if cfg:
        folium.TileLayer(tiles=cfg["tiles"], attr=cfg["attr"], name=style, max_zoom=16).add_to(m)
    else:
        m.get_root().html.add_child(folium.Element(
            "<style>.leaflet-container{background:#1c1f24!important}</style>"))
    return m


def add_points(m, sub, max_points=5000):
    pts = sub[["latitude", "longitude", "acq_date", "sensor"]].copy()
    pts["sensor"] = pts["sensor"].astype(str)
    pts = pts.dropna(subset=["latitude", "longitude"])
    n_all = len(pts)
    if n_all > max_points:                                     # even split per sensor
        per = max_points // pts["sensor"].nunique()
        pts = pd.concat(
            [g.sample(min(len(g), per), random_state=0) for _, g in pts.groupby("sensor")]
        )
    dates = pts["acq_date"].dt.strftime("%Y-%m-%d").tolist()
    layer = folium.FeatureGroup(name="Detections")
    for la, lo, d, sn in zip(pts["latitude"], pts["longitude"], dates, pts["sensor"]):
        col = SENSOR_COLORS.get(sn, OTHER_COLOR)
        folium.CircleMarker(
            location=[float(la), float(lo)], radius=3, color=col, weight=0,
            fill=True, fill_color=col, fill_opacity=0.55, tooltip=f"{d}, {sn}",
        ).add_to(layer)
    layer.add_to(m)

    present = [sn for sn in ("MODIS", "VIIRS") if sn in set(pts["sensor"])]
    items = "".join(
        f'<span style="color:{SENSOR_COLORS[sn]}">&#9679;</span>&nbsp;{sn}&nbsp;&nbsp;' for sn in present
    )
    m.get_root().html.add_child(folium.Element(
        '<div style="position:fixed;bottom:26px;left:12px;z-index:9999;'
        'background:rgba(18,18,20,.8);color:#e8e8e8;padding:5px 10px;'
        f'border-radius:4px;font:12px sans-serif">{items}</div>'
    ))
    return n_all, len(pts)


def _grid(df, c, value_col="weight", denom=1.0):
    g = pd.DataFrame({
        "glat": np.floor(df["latitude"].to_numpy() / c) * c + c / 2,
        "glon": np.floor(df["longitude"].to_numpy() / c) * c + c / 2,
        "w": df[value_col].to_numpy() / denom,
    })
    return g.groupby(["glat", "glon"], as_index=False)["w"].sum()


def _cell_feature(la, lo, h, props):
    return {
        "type": "Feature", "properties": props,
        "geometry": {"type": "Polygon", "coordinates": [[
            [lo - h, la - h], [lo + h, la - h], [lo + h, la + h], [lo - h, la + h], [lo - h, la - h]
        ]]},
    }


def add_density(m, sub):
    cell = 0.05
    grid = _grid(sub, cell)
    while len(grid) > 2500 and cell < 5:
        cell *= 2
        grid = _grid(sub, cell)
    vmin = max(float(grid["w"].min()), 0.1)
    vmax = max(float(grid["w"].max()), vmin * 1.5)
    norm = colors.LogNorm(vmin=vmin, vmax=vmax, clip=True)
    cmap = colormaps["YlOrRd"]
    feats = [
        _cell_feature(la, lo, cell / 2,
                      {"count": round(float(w)), "color": colors.to_hex(cmap(0.2 + 0.8 * norm(w)))})
        for la, lo, w in zip(grid["glat"], grid["glon"], grid["w"])
    ]
    folium.GeoJson(
        {"type": "FeatureCollection", "features": feats},
        style_function=lambda f: {"fillColor": f["properties"]["color"], "fillOpacity": 0.65, "weight": 0},
        tooltip=folium.GeoJsonTooltip(fields=["count"], aliases=["Detections (VIIRS-eq.)"]),
    ).add_to(m)
    return cell, int(round(grid["w"].max()))


def add_change(m, sub, recent, base):
    """Mean annual detections per cell: recent years minus the earlier baseline."""
    cell = 0.1
    while True:
        gr = _grid(sub[sub["year"].isin(recent)], cell, denom=len(recent))
        gb = _grid(sub[sub["year"].isin(base)], cell, denom=len(base))
        mg = gr.merge(gb, on=["glat", "glon"], how="outer", suffixes=("_r", "_b")).fillna(0)
        if len(mg) <= 2500 or cell >= 5:
            break
        cell *= 2
    mg["d"] = mg["w_r"] - mg["w_b"]
    lim = max(float(np.percentile(np.abs(mg["d"]), 98)), 0.5)
    norm = colors.Normalize(vmin=-lim, vmax=lim, clip=True)
    cmap = colormaps["PuOr_r"]                                  # purple = fewer, orange = more
    mg = mg[mg["d"].abs() >= 0.05 * lim]
    feats = [
        _cell_feature(la, lo, cell / 2, {
            "change": round(float(d), 1), "recent": round(float(r), 1), "baseline": round(float(b), 1),
            "color": colors.to_hex(cmap(norm(d)))})
        for la, lo, d, r, b in zip(mg["glat"], mg["glon"], mg["d"], mg["w_r"], mg["w_b"])
    ]
    folium.GeoJson(
        {"type": "FeatureCollection", "features": feats},
        style_function=lambda f: {"fillColor": f["properties"]["color"], "fillOpacity": 0.7, "weight": 0},
        tooltip=folium.GeoJsonTooltip(
            fields=["change", "recent", "baseline"],
            aliases=["Change / yr", "Recent / yr", "Baseline / yr"]),
    ).add_to(m)
    return cell, lim


ATTRIBUTION_FIX = (
    "<style>.leaflet-control-attribution{background:rgba(255,255,255,.85)!important;"
    "color:#333!important;font-size:10px}.leaflet-control-attribution a{color:#06c!important}</style>"
)

PAGE_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Public+Sans:wght@400;500;600;700&display=swap');
:root {--lichen: #a8c08a; --line: rgba(190,205,195,.18); --muted: #98a39d;}
html, body, .stApp, .stApp p, .stApp label, .stApp li, .stApp h1, .stApp h2, .stApp h3, .stApp h4,
.stApp button, .stApp input, .stApp textarea, .stApp [data-baseweb="tab"] {font-family: __FONT__;}
.block-container {padding-top: 4.4rem; max-width: 1480px;}
footer, #MainMenu, [data-testid="stAppDeployButton"] {visibility: hidden;}

.masthead h1 {font-size: 2.1rem; font-weight: 700; letter-spacing: -0.025em; line-height: 1.1;
              margin: 0; padding: 0;}
.masthead p {color: var(--muted); font-size: 1rem; line-height: 1.5; max-width: 60ch;
             margin: .45rem 0 1.4rem 0;}

[data-testid="stSidebar"] {border-right: 1px solid var(--line);}
.kicker {font-size: .84rem; font-weight: 600; margin: 1.35rem 0 .35rem 0;}

[data-testid="stMetric"] {border-left: 2px solid var(--line); padding: .1rem 0 .15rem .9rem;}
[data-testid="stMetricLabel"] p {font-size: .82rem; color: var(--muted); white-space: normal;}
[data-testid="stMetricDelta"] {font-size: .8rem;}
[data-testid="stMetricValue"] {font-size: 1.4rem; font-weight: 600; letter-spacing: -0.015em;
                               font-variant-numeric: tabular-nums;}

.stTabs [data-baseweb="tab-list"] {gap: 1.6rem; border-bottom: 1px solid var(--line);}
.stTabs [data-baseweb="tab"] {padding: .55rem 0; background: transparent;}
.stTabs [data-baseweb="tab-highlight"] {background-color: var(--lichen);}

.callout {border-left: 4px solid; padding: .8rem 1.1rem; border-radius: 2px;
          margin: .2rem 0 1.3rem 0; font-size: 1.02rem; line-height: 1.6; max-width: 90ch;}
.callout.low, .callout.normal {border-color: #6f94b3; background: rgba(111,148,179,.09);}
.callout.elevated {border-color: #d9a441; background: rgba(217,164,65,.09);}
.callout.high {border-color: #ef6c33; background: rgba(239,108,51,.11);}

.hero {border-left: 4px solid; padding: .15rem 0 .15rem 1.1rem; margin: .4rem 0 1.3rem 0;
       animation: rise .6s ease-out;}
.hero.low, .hero.normal {border-color: #6f94b3;}
.hero.elevated {border-color: #d9a441;}
.hero.high {border-color: #ef6c33;}
.hero-line {font-size: 1.7rem; font-weight: 650; letter-spacing: -0.02em; line-height: 1.2;}
.hero-sub {color: var(--muted); font-size: .94rem; line-height: 1.55; margin-top: .55rem;}
@keyframes rise {from {opacity: 0; transform: translateY(8px);} to {opacity: 1; transform: none;}}
@media (prefers-reduced-motion: reduce) {.hero {animation: none;}}
.hero.much_more {border-color: #ef6c33;}
.hero.more {border-color: #d9a441;}
.hero.typical {border-color: #6f94b3;}
.hero.less {border-color: #8a78c0;}
.hero.much_less {border-color: #6d5aa8;}
.hero.partial {border-color: #59635f;}
.lede {font-size: 1.08rem; line-height: 1.65; max-width: 78ch; margin: .2rem 0 1.3rem 0;}
.stripes {display: flex; height: 4.6rem; gap: 1px; border-radius: 3px; overflow: hidden; margin: .5rem 0 .25rem 0;}
.stripe {flex: 1; transition: filter .15s;}
.stripe:hover {filter: brightness(1.35);}
.stripe-axis {display: flex; justify-content: space-between; color: var(--muted); font-size: .78rem;}
.yeargrid {display: grid; grid-template-columns: repeat(auto-fill, minmax(165px, 1fr)); gap: 8px; margin: 1.1rem 0 .5rem 0;}
.yearcard {border-top: 3px solid var(--line); background: rgba(190,205,195,.05); padding: .6rem .75rem .7rem; border-radius: 2px;}
.yearcard .y {font-size: 1.15rem; font-weight: 650; font-variant-numeric: tabular-nums;}
.yearcard .h {font-size: .84rem; line-height: 1.35; margin-top: .15rem;}
.yearcard .s {font-size: .78rem; color: var(--muted); margin-top: .3rem;}
.yearcard.much_more {border-color: #ef6c33;}
.yearcard.more {border-color: #d9a441;}
.yearcard.typical {border-color: #6f94b3;}
.yearcard.less {border-color: #8a78c0;}
.yearcard.much_less {border-color: #6d5aa8;}
.yearcard.partial {border-color: #59635f; opacity: .75;}
.ribbon-title {font-size: .9rem; font-weight: 600; margin: 1.2rem 0 .4rem 0;}
.ribbon {display: grid; grid-template-columns: repeat(6, 1fr); gap: 3px;}
.ribbon-cell {height: 2.3rem; border-radius: 2px; padding: .2rem .4rem;
              display: flex; align-items: flex-end; font-size: .72rem;}
.step {border-top: 2px solid var(--lichen); padding-top: .7rem;}
.step b {font-size: 1.02rem;}
.step p {color: var(--muted); font-size: .92rem; line-height: 1.55; margin: .3rem 0 0 0;}

.note {font-size: .88rem; color: var(--muted); line-height: 1.55;}
.credits {margin-top: 2.2rem; padding-top: .8rem; border-top: 1px solid var(--line);
          color: var(--muted); font-size: .8rem; line-height: 1.6;}

[data-testid="stCustomComponentV1"] iframe {border: 1px solid var(--line); border-radius: 3px;}
.stDownloadButton button, .stButton button {border-radius: 4px; border: 1px solid var(--line);}
</style>
"""

# --------------------------------------------------------------------------
# Page
# --------------------------------------------------------------------------
st.markdown(PAGE_CSS.replace("__FONT__", FONT_STACK), unsafe_allow_html=True)
st.markdown(
    '<div class="masthead"><h1>Fire Activity Explorer</h1>'
    "<p>Satellites have counted fire hotspots since 2000, but with two different instruments. "
    "Here MODIS and VIIRS are joined into one record, so you can compare any month, in any area, "
    "with everything that came before it.</p></div>",
    unsafe_allow_html=True,
)

try:
    regions, canon = load_regions()
except Exception as e:
    st.error(f"Could not load country boundaries from {NE_PATH}: {e}")
    st.stop()

stems = data_stems()


def has_data(key: str) -> bool:
    return slug(key, canon) in stems or key.replace(" ", "_") in stems


def region_label(key: str) -> str:
    return key.title() + ("  (data loaded)" if has_data(key) else "")


# ---- Sidebar --------------------------------------------------------------
sb = st.sidebar
sb.markdown('<div class="kicker">Region</div>', unsafe_allow_html=True)
match = sb.selectbox(
    "Region", sorted(regions), index=None, format_func=region_label,
    placeholder="Search, e.g. Egypt", label_visibility="collapsed", key="region_pick",
)
bbox = regions[match] if match else None
match_title = match.title() if match else ""

sb.markdown('<div class="kicker">Area of interest</div>', unsafe_allow_html=True)
aoi_mode = sb.radio("Area of interest", ["Whole region", "Draw on map", "Coordinates"],
                    label_visibility="collapsed")
aoi_poly = None
if match and aoi_mode == "Draw on map":
    saved = st.session_state.get("aoi_draw")
    if saved and saved["region"] == match:
        aoi_poly = shapely_wkt.loads(saved["wkt"])
    sb.caption("Use the polygon, rectangle or circle tool at the top left of the map.")
    sb.button("Clear drawn area", on_click=lambda: st.session_state.pop("aoi_draw", None))
elif match and aoi_mode == "Coordinates":
    kind = sb.radio("Shape", ["Bounding box", "Point and radius"], horizontal=True,
                    label_visibility="collapsed")
    if kind == "Bounding box":
        c1, c2 = sb.columns(2)
        lat0 = c1.number_input("South latitude", -90.0, 90.0, round(bbox[0], 3), 0.1,
                               format="%.3f", key=f"s_{match}")
        lat1 = c2.number_input("North latitude", -90.0, 90.0, round(bbox[1], 3), 0.1,
                               format="%.3f", key=f"n_{match}")
        lon0 = c1.number_input("West longitude", -180.0, 180.0, round(bbox[2], 3), 0.1,
                               format="%.3f", key=f"w_{match}")
        lon1 = c2.number_input("East longitude", -180.0, 180.0, round(bbox[3], 3), 0.1,
                               format="%.3f", key=f"e_{match}")
        if lat1 > lat0 and lon1 > lon0:
            aoi_poly = box(lon0, lat0, lon1, lat1)
        else:
            sb.warning("North must be above south, and east right of west.")
    else:
        c1, c2 = sb.columns(2)
        clat = c1.number_input("Latitude", -90.0, 90.0, round((bbox[0] + bbox[1]) / 2, 3), 0.1,
                               format="%.3f", key=f"clat_{match}")
        clon = c2.number_input("Longitude", -180.0, 180.0, round((bbox[2] + bbox[3]) / 2, 3), 0.1,
                               format="%.3f", key=f"clon_{match}")
        rad = sb.number_input("Radius (km)", 1.0, 2000.0, 50.0, 5.0, key=f"rad_{match}")
        aoi_poly = circle_polygon(clon, clat, rad * 1000.0)

sb.markdown('<div class="kicker">Period</div>', unsafe_allow_html=True)
date_box = sb.container()
sb.markdown('<div class="kicker">Map</div>', unsafe_allow_html=True)
layer = sb.selectbox("Map layer", ["Points", "Density grid", "Change vs baseline"],
                     label_visibility="collapsed")
style = sb.selectbox("Base map", list(TILES), label_visibility="collapsed")
sb.markdown('<div class="kicker">Data cleaning</div>', unsafe_allow_html=True)
with sb.expander("Quality and harmonization"):
    drop_low = st.checkbox(
        "Drop low-confidence detections", True,
        help="Removes MODIS confidence below 30 and VIIRS 'low'.")
    veg_only = st.checkbox(
        "Vegetation fires only", True,
        help="Keeps FIRMS type 0. Removes volcanoes, gas flares and other static sources "
             "(only applies when the file has a type column).")
    viirs_mode = st.radio(
        "VIIRS satellites", ["SNPP only", "All, normalized per satellite"],
        help="SNPP has flown since 2012, so it is the most consistent reference. "
             "NOAA-20 and NOAA-21 add overpasses that raise raw counts without more fire.")
    scaling = st.radio(
        "MODIS to VIIRS scaling", ["Single factor", "By calendar month"],
        help="Monthly factors follow the seasonal difference in what each sensor sees "
             "(VIIRS picks up many small agricultural fires MODIS misses).")
    min_monthly = st.number_input(
        "Min. detections per overlap month", 1, 500, 10,
        help="Months below this are ignored when computing the scaling factor.")

# ---- Data -----------------------------------------------------------------
sub, info = None, None
data_ready = False
dmax = None

if match:
    path = find_data_file(match, canon)
    if path is None:
        st.info(
            f"No fire data for {match_title} yet. Add `data/{slug(match, canon)}.csv` "
            "(a FIRMS archive download) to see hotspots."
        )
    else:
        try:
            hot, info = build_dataset(
                str(path), path.stat().st_mtime, drop_low, veg_only,
                "snpp" if viirs_mode.startswith("SNPP") else "all",
                scaling.startswith("By"), int(min_monthly),
            )
        except Exception as e:
            st.error(f"Could not process {path.name}: {e}")
            st.stop()

        if hot.empty:
            st.warning("No detections left after the quality filters.")
        else:
            data_ready = True
            dmin, dmax = hot["acq_date"].min().date(), hot["acq_date"].max().date()
            with date_box:
                if dmin < dmax:
                    d0, d1 = st.slider("Date range", dmin, dmax, (dmin, dmax),
                                       format="YYYY-MM-DD", label_visibility="collapsed")
                else:
                    d0, d1 = dmin, dmax
            sub = hot[(hot["acq_date"] >= pd.Timestamp(d0)) & (hot["acq_date"] <= pd.Timestamp(d1))]
            if aoi_poly is not None:
                sub = apply_aoi(sub, aoi_poly)
            if sub.empty:
                st.warning("No hotspots in this area and period." if aoi_poly is not None
                           else "No hotspots in this date range.")
                sub = None

aoi_area = polygon_area_km2(aoi_poly) if aoi_poly is not None else None
if aoi_poly is not None:
    minx, miny, maxx, maxy = aoi_poly.bounds
    sb.caption(f"About {aoi_area:,.0f} km², from {miny:.2f}° to {maxy:.2f}° N "
               f"and {minx:.2f}° to {maxx:.2f}° E.")

# ---- Statistics -----------------------------------------------------------
status = None
if sub is not None:
    lo, hi, tol = pd.Timestamp(d0), pd.Timestamp(d1), pd.Timedelta(days=3)
    first_y, last_y = int(sub["year"].min()), int(sub["year"].max())
    years = list(range(first_y, last_y + 1))

    monthly = sub.groupby(["year", "month"])["weight"].sum()
    pivot = monthly.unstack("month").reindex(index=years, columns=range(1, 13))
    covered = pd.DataFrame(False, index=pivot.index, columns=pivot.columns)
    full = covered.copy()
    for y in pivot.index:
        for mo in pivot.columns:
            s = pd.Timestamp(y, mo, 1)
            e = s + pd.offsets.MonthEnd(0)
            covered.loc[y, mo] = (e >= lo) and (s <= hi)
            full.loc[y, mo] = (s + tol >= lo) and (e - tol <= hi)
    pivot_cov = pivot.fillna(0).where(covered)
    pivot_full = pivot.fillna(0).where(full)
    med_by_month = pivot_full.median()

    annual_by_sensor = sub.groupby(["year", "sensor"], observed=True)["weight"].sum().unstack(fill_value=0)
    annual_total = annual_by_sensor.sum(axis=1)
    annual_raw = sub.groupby("year").size()
    full_years = [y for y in years if full.loc[y].all()]
    trend_h = rel_trend(annual_total, full_years)
    trend_raw = rel_trend(annual_raw, full_years)

    latest = sub["acq_date"].max()
    ly, lm, last_day = latest.year, latest.month, latest.day
    current = float(sub[(sub["year"] == ly) & (sub["month"] == lm)]["weight"].sum())
    past_years = [y for y in years if y < ly]
    hist = (
        sub[(sub["month"] == lm) & (sub["year"] < ly) & (sub["acq_date"].dt.day <= last_day)]
        .groupby("year")["weight"].sum().reindex(past_years, fill_value=0)
    )
    if len(hist) > 0:
        status = status_for(current, hist)

    modis_share = (annual_by_sensor["MODIS"] / annual_total) if "MODIS" in annual_by_sensor \
        else pd.Series(0.0, index=annual_total.index)
    ytd = None
    past_full = [y for y in full_years if y < ly]
    if ly not in full_years and len(past_full) >= 4:
        upto = (sub["month"] < lm) | ((sub["month"] == lm) & (sub["acq_date"].dt.day <= last_day))
        ytd_by_year = sub[upto].groupby("year")["weight"].sum()
        ytd = dict(year=ly, cur=float(ytd_by_year.get(ly, 0.0)),
                   hist=ytd_by_year.reindex(past_full, fill_value=0.0),
                   label=f"{calendar.month_abbr[lm]} {last_day}")
    verdicts = year_verdicts(annual_total, modis_share, pivot_cov, full_years, med_by_month, ytd)
    vmap = {v["year"]: v for v in verdicts}
    year_baseline = float(annual_total.loc[full_years].median()) if len(full_years) >= 4 else None

    region_text = match_title + (" (selected area)" if aoi_poly is not None else "")

# ---- Overview and map -----------------------------------------------------
draw_mode = bool(match) and aoi_mode == "Draw on map"
aoi_key = hashlib.md5(aoi_poly.wkt.encode()).hexdigest()[:8] if aoi_poly is not None else "none"

HERO_LINES = {
    "low": "Quieter than usual for this time of year.",
    "normal": "About normal for this time of year.",
    "elevated": "Busier than usual for this time of year.",
    "high": "Unusually busy for this time of year.",
}


def render_map(msub, tag="all", height=640):
    m = base_map(style)
    if msub is not None:
        if layer == "Points":
            n_all, n_shown = add_points(m, msub)
            if n_shown < n_all:
                st.caption(f"Showing {n_shown:,} of {n_all:,} detections, sampled evenly per sensor. "
                           "Switch to Density grid to see the full counts.")
        elif layer == "Density grid":
            cell, maxc = add_density(m, msub)
            st.caption(f"Grid cell about {cell:g}° (~{int(cell * 111)} km). "
                       f"Densest cell: {maxc:,} VIIRS-equivalent detections.")
        else:
            if len(full_years) >= 8:
                recent, base = full_years[-5:], full_years[:-5]
                cell, lim = add_change(m, sub, recent, base)
                st.caption(
                    f"Mean annual detections per cell, {recent[0]}–{recent[-1]} compared with "
                    f"{base[0]}–{base[-1]}. Orange means more fire than the baseline, purple means less "
                    f"(strongest change about {lim:,.1f} per year per cell). This layer ignores the "
                    "year and month filters."
                )
            else:
                st.info("The change layer needs at least eight complete years in the selected range.")
    if aoi_poly is not None:
        folium.GeoJson(
            mapping(aoi_poly),
            style_function=lambda f: {"color": AOI_COLOR, "weight": 2, "dashArray": "6 4",
                                      "fillColor": AOI_COLOR, "fillOpacity": 0.04},
        ).add_to(m)
    if draw_mode and data_ready:
        Draw(export=False, draw_options={"polyline": False, "marker": False, "circlemarker": False},
             edit_options={"edit": False}).add_to(m)

    if aoi_poly is not None:
        minx, miny, maxx, maxy = aoi_poly.bounds
        m.fit_bounds([[miny, minx], [maxy, maxx]], padding=(20, 20))
    elif bbox:
        m.fit_bounds([[bbox[0], bbox[2]], [bbox[1], bbox[3]]])
    m.get_root().html.add_child(folium.Element(ATTRIBUTION_FIX))

    result = st_folium(
        m, height=height, use_container_width=True,
        returned_objects=["all_drawings"] if (draw_mode and data_ready) else [],
        key=f"map-{match}-{layer}-{style}-{aoi_key}-{tag}",
    )
    if draw_mode and data_ready and result and result.get("all_drawings"):
        poly = drawing_to_polygon(result["all_drawings"][-1])
        if poly is not None:
            saved = st.session_state.get("aoi_draw")
            if not saved or saved["region"] != match or saved["wkt"] != poly.wkt:
                st.session_state["aoi_draw"] = {"region": match, "wkt": poly.wkt}
                st.rerun()


def render_overview():
    st.markdown('<div class="ribbon-title" style="margin-top:.2rem">Every year at a glance</div>',
                unsafe_allow_html=True)
    render_stripes(verdicts)
    st.caption("One stripe per year. Orange means more fire than the usual year, purple means less, "
               "and grey-green is about usual. Hover over a stripe for the verdict.")
    left, right = st.columns([4, 8], gap="large")

    with left:
        st.caption(f"Showing {region_text}.")
        if status is not None:
            if status["median"] > 0:
                ratio_txt = f"{current / status['median']:.1f}× the median ({fmt(status['median'])})"
            else:
                ratio_txt = "where the median year had none"
            st.markdown(
                f'<div class="hero {status["level"]}">'
                f'<div class="hero-line">{HERO_LINES[status["level"]]}</div>'
                f'<div class="hero-sub">{calendar.month_name[lm]} 1–{last_day}, {ly}: {fmt(current)} detections, '
                f'{ratio_txt} for the same days across the {len(hist)} earlier years. '
                f'It ranks {ordinal(status["rank"])} highest of {status["n"]}.</div></div>',
                unsafe_allow_html=True,
            )

        a, b = st.columns(2)
        a.metric("Hotspots detected", fmt(len(sub)),
                 help="Individual detections kept after the quality filters and the MODIS/VIIRS splice.")
        b.metric("Harmonized total", fmt(sub["weight"].sum()),
                 help="VIIRS-equivalent detections: per-satellite normalized, MODIS scaled.")
        c, d = st.columns(2)
        c.metric("Peak month", calendar.month_name[int(med_by_month.idxmax())] if med_by_month.notna().any() else "n/a",
                 help="Calendar month with the highest median across complete months.")
        label = f"{calendar.month_abbr[lm]} {ly}, days 1–{last_day}"
        if status is not None:
            d.metric(label, fmt(current), delta=f"{current - status['median']:+,.0f} vs median",
                     delta_color="inverse", help=f"Median of the same days in earlier years: {fmt(status['median'])}.")
        else:
            d.metric(label, fmt(current))
        e, g = st.columns(2)
        if trend_h is not None:
            e.metric("Annual trend", f"{trend_h[0]:+.1f}% / yr",
                     help=f"Theil-Sen slope over {len(full_years)} complete years, relative to the median year.")
        else:
            e.metric("Annual trend", "n/a", help="Needs at least five complete years in the selected range.")
        g.metric("Latest data", f"{dmax:%d %b %Y}")

        if med_by_month.notna().any() and med_by_month.max() > 0:
            top = float(med_by_month.max())
            cells = []
            for mo in range(1, 13):
                v = med_by_month.get(mo, np.nan)
                frac = 0.0 if pd.isna(v) else float(v) / top
                ring = "box-shadow: inset 0 0 0 2px #a8c08a;" if mo == lm else ""
                ink = "#171c1b" if frac > 0.6 else "#e8e6df"
                tip = f"{calendar.month_name[mo]}: median {fmt(0 if pd.isna(v) else v)} detections"
                cells.append(
                    f'<div class="ribbon-cell" title="{tip}" style="background: rgba(239,108,51,{0.07 + 0.88 * frac:.2f});'
                    f' color: {ink}; {ring}">{calendar.month_abbr[mo]}</div>'
                )
            st.markdown(
                '<div class="ribbon-title">When this area usually burns</div>'
                f'<div class="ribbon">{"".join(cells)}</div>'
                '<p class="note" style="margin-top:.4rem">Median detections per calendar month across complete '
                "years. The outlined month is the latest one in your data.</p>",
                unsafe_allow_html=True,
            )

        if info["has_both"]:
            ci = info["factor_ci"]
            ci_txt = f" (range {ci[0]:.2f}–{ci[1]:.2f})" if ci else ""
            st.caption(
                f"MODIS scaled ×{info['factor']:.2f}{ci_txt} to VIIRS-equivalent from {info['n_valid']} "
                f"overlap months. {fmt(info['n_final'])} of {fmt(info['n_raw'])} detections kept."
            )

    with right:
        years_all = sorted(int(y) for y in sub["year"].unique())
        st.caption("Pick a year, or a few months, to see only that window on the map. "
                   "The figures on the left always cover the whole period.")
        year_pick = st.select_slider(
            "Year on the map", options=["All years"] + years_all, value="All years",
            key=f"map_year_{match}_{years_all[0]}_{years_all[-1]}_{aoi_key}")
        month_opts = list(range(1, 13))
        if hasattr(st, "pills"):
            month_pick = st.pills("Months on the map", month_opts, selection_mode="multi", default=month_opts,
                                  format_func=lambda mo: calendar.month_abbr[mo], key="map_months")
        else:
            month_pick = st.multiselect("Months on the map", month_opts, default=month_opts,
                                        format_func=lambda mo: calendar.month_abbr[mo], key="map_months")
        month_pick = list(month_pick or month_opts)

        msub = sub
        if year_pick != "All years":
            msub = msub[msub["year"] == year_pick]
        if len(month_pick) < 12:
            msub = msub[msub["month"].isin(month_pick)]
        tag = hashlib.md5(f"{year_pick}-{sorted(month_pick)}".encode()).hexdigest()[:6]
        if msub.empty:
            st.info("No detections in that window. Try another year or more months.")
            render_map(None, tag)
        else:
            if len(msub) < len(sub):
                st.caption(f"{fmt(len(msub))} detections in this window.")
            render_map(msub, tag)

# ---- Tabs -----------------------------------------------------------------
if sub is not None:
    t_over, t_year, t_cal, t_trend, t_time, t_harm, t_data = st.tabs(
        ["Overview", "Year by year", "Calendar", "Trends", "Timelapse", "Harmonization", "Data"])

    with t_over:
        render_overview()

    with t_year:
        st.markdown(f'<p class="lede">{escape(years_summary(verdicts, region_text, year_baseline))}</p>',
                    unsafe_allow_html=True)
        render_stripes(verdicts)
        opts = [v["year"] for v in verdicts]
        pick = st.select_slider("Year in focus", options=opts, value=opts[-1],
                                key=f"year_pick_{match}_{aoi_key}_{opts[0]}_{opts[-1]}")
        vy = vmap[pick]
        st.markdown(
            f'<div class="hero {vy["level"]}"><div class="hero-line">{pick}: {escape(vy["headline"])}</div>'
            f'<div class="hero-sub">{escape(vy["detail"])}</div></div>',
            unsafe_allow_html=True,
        )
        mn = [calendar.month_abbr[c] for c in range(1, 13)]
        row = pivot_cov.loc[pick].astype(float)
        qs = pivot_full.quantile([0.25, 0.5, 0.75])
        mfig = go.Figure()
        if qs.notna().any().any():
            mfig.add_trace(go.Scatter(x=mn, y=qs.loc[0.75].to_numpy(), mode="lines", line=dict(width=0),
                                      showlegend=False, hoverinfo="skip"))
            mfig.add_trace(go.Scatter(x=mn, y=qs.loc[0.25].to_numpy(), mode="lines", line=dict(width=0),
                                      fill="tonexty", fillcolor="rgba(138,143,152,.25)",
                                      name="Middle 50% of years", hoverinfo="skip"))
            mfig.add_trace(go.Scatter(x=mn, y=qs.loc[0.5].to_numpy(), mode="lines",
                                      line=dict(color=OTHER_COLOR, width=2, dash="dash"), name="Usual month"))
        mfig.add_bar(x=mn, y=row.to_numpy(), name=str(pick), marker_color=LEVEL_COLORS[vy["level"]])
        show(tidy(mfig, f"{pick} month by month against the usual year", 340), "yearchart")

        cards = []
        for v in verdicts:
            cards.append(
                f'<div class="yearcard {v["level"]}" title="{escape(v["detail"], quote=True)}">'
                f'<div class="y">{v["year"]}</div><div class="h">{escape(v["headline"])}</div>'
                f'<div class="s">{escape(v["short"])}</div></div>'
            )
        st.markdown(f'<div class="yeargrid">{"".join(cards)}</div>', unsafe_allow_html=True)
        st.caption("Each year is rated against the usual (median) complete year in the selected area and period. "
                   "Hover over a card for the full sentence.")

    with t_cal:
        ca, cb = st.columns([2, 1])
        mode_c = ca.radio("Show", ["Detections", "Anomaly vs median"], horizontal=True, key="calmode")
        ramp = cb.selectbox("Colour ramp", ["Inferno", "Viridis", "Cividis"], key="ramp",
                            help="All three are perceptually uniform and readable with colour-vision deficiency.",
                            disabled=mode_c != "Detections")
        mnames = [calendar.month_abbr[c] for c in range(1, 13)]
        if mode_c == "Detections":
            heat_df = pivot_cov.copy()
            heat_df.columns = mnames
            heat = px.imshow(heat_df, color_continuous_scale=ramp, aspect="auto",
                             labels=dict(x="Month", y="Year", color="Detections"))
            heat.update_traces(texttemplate="%{z:.2~s}", textfont_size=9)
            heat_title = f"Monthly detections, {region_text}"
        else:
            anom = (pivot_full - med_by_month) / med_by_month.replace(0, np.nan) * 100
            anom.columns = mnames
            vals = np.abs(anom.to_numpy(float))
            lim = float(np.nanpercentile(vals, 95)) if np.isfinite(vals).any() else 100.0
            lim = min(max(lim, 50.0), 300.0)
            heat = px.imshow(anom, color_continuous_scale="PuOr_r", zmin=-lim, zmax=lim, aspect="auto",
                             labels=dict(x="Month", y="Year", color="% vs median"))
            heat.update_traces(texttemplate="%{z:+.0f}%", textfont_size=9)
            heat_title = f"Deviation from each calendar month's median, {region_text}"
        heat.update_yaxes(type="category", automargin=True)
        heat.update_xaxes(automargin=True)
        heat.update_layout(
            height=max(300, 30 * len(pivot_cov) + 140),
            margin=dict(l=10, r=10, t=56, b=10),
            title=dict(text=heat_title, x=0, xanchor="left", y=0.98, yanchor="top",
                       yref="container", font=dict(size=14)),
            coloraxis_colorbar=dict(title=None, thickness=12, len=0.9),
            font=dict(family=FONT_PLOT, size=12),
        )
        show(heat, "calendar")
        st.caption("Partial months at the edges of the range appear in the detections view and are "
                   "left blank in the anomaly view. Anomaly colours: orange above the median, purple below.")

        flat = (pivot_full.rename_axis("year").reset_index()
                .melt(id_vars="year", var_name="month", value_name="n").dropna())
        tl, tr_ = st.columns(2)
        if not flat.empty:
            top_m = flat.nlargest(8, "n").copy()
            top_m["Month"] = [f"{calendar.month_abbr[int(a)]} {int(b)}" for a, b in zip(top_m["month"], top_m["year"])]
            top_m["Detections"] = top_m["n"].round(0).astype(int)
            top_m["Times the median"] = [
                f"{n / med_by_month[int(mo)]:.1f}×" if med_by_month[int(mo)] > 0 else "n/a"
                for n, mo in zip(top_m["n"], top_m["month"])]
            tl.markdown("**Most active months**")
            tl.dataframe(top_m[["Month", "Detections", "Times the median"]], hide_index=True)
        if len(full_years) >= 1:
            top_y = annual_total.loc[full_years].nlargest(8)
            tr_.markdown("**Most active complete years**")
            tr_.dataframe(pd.DataFrame({"Year": top_y.index.astype(int), "Detections": top_y.round(0).astype(int).to_numpy()}),
                          hide_index=True)

        q = pivot_full.quantile([0.25, 0.5, 0.75])
        if q.notna().any().any():
            sc = go.Figure()
            sc.add_trace(go.Scatter(x=mnames, y=q.loc[0.75].to_numpy(), mode="lines",
                                    line=dict(width=0), showlegend=False, hoverinfo="skip"))
            sc.add_trace(go.Scatter(x=mnames, y=q.loc[0.25].to_numpy(), mode="lines", line=dict(width=0),
                                    fill="tonexty", fillcolor="rgba(138,143,152,.25)",
                                    name="Middle 50% of years", hoverinfo="skip"))
            sc.add_trace(go.Scatter(x=mnames, y=q.loc[0.5].to_numpy(), mode="lines",
                                    line=dict(color=OTHER_COLOR, width=2), name="Median"))
            sc.add_trace(go.Scatter(x=mnames, y=pivot_cov.loc[last_y].to_numpy(), mode="lines+markers",
                                    line=dict(color=VIIRS_COLOR, width=2.5), marker=dict(size=6),
                                    name=str(last_y)))
            show(tidy(sc, "Seasonal cycle against the latest year", 340), "season")

    with t_trend:
        ann = go.Figure()
        ci = info["factor_ci"]
        for s, nm in (("MODIS", "MODIS-derived"), ("VIIRS", "VIIRS"), ("UNKNOWN", "Unlabelled")):
            if s in annual_by_sensor:
                kw = {}
                if s == "MODIS" and ci and not scaling.startswith("By") and info["factor"] > 0:
                    v = annual_by_sensor[s]
                    kw["error_y"] = dict(type="data", symmetric=False, thickness=1, width=2,
                                         array=(v * (ci[1] / info["factor"] - 1)).to_numpy(),
                                         arrayminus=(v * (1 - ci[0] / info["factor"])).to_numpy())
                ann.add_bar(x=annual_by_sensor.index, y=annual_by_sensor[s], name=nm,
                            marker_color=SENSOR_COLORS[s], **kw)
        ann.update_layout(barmode="stack")
        if trend_h is not None:
            xs = np.array([min(full_years), max(full_years)])
            ann.add_scatter(x=xs, y=trend_h[1] * xs + trend_h[2], mode="lines", name="Trend (Theil-Sen)",
                            line=dict(color=OTHER_COLOR, dash="dash", width=2))
        ann.update_xaxes(dtick=2)
        show(tidy(ann, "Annual detections, VIIRS-equivalent (partial years included)"), "annual")
        if ci and not scaling.startswith("By"):
            st.caption("Whiskers on the MODIS-derived bars show the bootstrap range of the scaling factor.")

        if status is not None:
            vals = pd.concat([hist, pd.Series({ly: current})])
            bar = go.Figure(go.Bar(
                x=vals.index, y=vals.to_numpy(),
                marker_color=[OTHER_COLOR] * len(hist) + [VIIRS_COLOR], showlegend=False))
            bar.add_scatter(x=[vals.index.min(), vals.index.max()], y=[status["median"]] * 2,
                            mode="lines", name="Median of past years",
                            line=dict(color=OTHER_COLOR, dash="dash", width=2))
            bar.update_xaxes(dtick=2)
            show(tidy(bar, f"{calendar.month_name[lm]}, days 1–{last_day}, every year"), "monthcmp")

    with t_time:
        try:
            if aoi_poly is not None:
                bx0, by0, bx1, by1 = aoi_poly.bounds
                lat_lo, lat_hi, lon_lo, lon_hi = by0, by1, bx0, bx1
            elif bbox:
                lat_lo, lat_hi, lon_lo, lon_hi = bbox
            else:
                lat_lo, lat_hi = float(sub["latitude"].min()), float(sub["latitude"].max())
                lon_lo, lon_hi = float(sub["longitude"].min()), float(sub["longitude"].max())
            span = max(lon_hi - lon_lo, (lat_hi - lat_lo) * 1.6, 0.05)
            zoom_t = float(np.clip(np.log2(773 / span) - 0.3, 1, 11))

            def yearly_grid(df, c):
                g = pd.DataFrame({
                    "year": df["year"].to_numpy(),
                    "lat": np.floor(df["latitude"].to_numpy() / c) * c + c / 2,
                    "lon": np.floor(df["longitude"].to_numpy() / c) * c + c / 2,
                    "w": df["weight"].to_numpy(),
                })
                return g.groupby(["year", "lat", "lon"], as_index=False)["w"].sum()

            cell_t = 0.1
            tl = yearly_grid(sub, cell_t)
            while len(tl) > 60000 and cell_t < 5:
                cell_t *= 2
                tl = yearly_grid(sub, cell_t)
            tl = tl.sort_values("year")
            cap = max(float(tl["w"].quantile(0.99)), 1.0)

            tile_style = style if style in ("Dark", "Light", "Satellite") else "Dark"
            layer_def = dict(below="traces", sourcetype="raster", source=[TILES[tile_style]["tiles"]],
                             sourceattribution="Tiles © Esri")
            args = dict(lat="lat", lon="lon", z="w", radius=14, animation_frame="year",
                        center=dict(lat=(lat_lo + lat_hi) / 2, lon=(lon_lo + lon_hi) / 2), zoom=zoom_t,
                        color_continuous_scale="YlOrRd", range_color=(0, cap), opacity=0.85)
            if hasattr(px, "density_map"):
                fig_t = px.density_map(tl, map_style="white-bg", **args)
                fig_t.update_layout(map_layers=[layer_def])
            else:
                fig_t = px.density_mapbox(tl, mapbox_style="white-bg", **args)
                fig_t.update_layout(mapbox_layers=[layer_def])
            fig_t.update_layout(height=640, margin=dict(l=0, r=0, t=0, b=0),
                                coloraxis_colorbar=dict(title="Detections per cell", thickness=12))
            st.caption("Press play to watch the burning season by season. Each frame is one year of "
                       f"VIIRS-equivalent detections on a grid of about {cell_t:g}°. "
                       "The colour scale is the same in every frame, so years can be compared.")
            show(fig_t, "timelapse")
        except Exception as exc:
            st.info(f"The timelapse could not be drawn here ({exc}). Update plotly to 5.24 or newer and try again.")

    with t_harm:
        if not info["has_both"]:
            st.info("This file has one sensor only, so no MODIS/VIIRS harmonization is needed. "
                    "Detections are still normalized per satellite.")
        else:
            split = info["split"]
            st.markdown(
                f"""
<p class="note">
<b>1. Filter.</b> Low-confidence and non-vegetation detections are removed (see sidebar).<br>
<b>2. Per-satellite normalization.</b> Each detection is divided by the number of satellites of its
sensor already in orbit on that date, so Terra alone before Aqua, or SNPP alone before NOAA-20,
is on the same footing as the full constellation.<br>
<b>3. Scaling.</b> From {split:%B %Y} both sensors overlap. For every month with enough detections
the VIIRS / MODIS ratio is computed, and its median is the factor ({info['factor']:.2f}).
The factor is calibrated on the whole file, so it stays stable when you zoom to a small area.<br>
<b>4. Splice.</b> Scaled MODIS is used before {split:%B %Y} and VIIRS after it, so no month is counted twice.
</p>""",
                unsafe_allow_html=True,
            )
            ci = info["factor_ci"]
            h1, h2, h3, h4 = st.columns(4)
            h1.metric("Scaling factor", f"×{info['factor']:.2f}",
                      delta=f"range {ci[0]:.2f}–{ci[1]:.2f}" if ci else None, delta_color="off",
                      help="Bootstrap 95% range over the overlap months.")
            h2.metric("Overlap months used", info["n_valid"])
            h3.metric("Monthly correlation", "n/a" if np.isnan(info["corr"]) else f"{info['corr']:.2f}",
                      help="Pearson r between MODIS and VIIRS monthly counts in the overlap.")
            err = info["err_seasonal"] if scaling.startswith("By") else info["err_single"]
            h4.metric("Median monthly error", "n/a" if np.isnan(err) else f"{err * 100:.0f}%",
                      help="How far scaled MODIS lands from VIIRS in the overlap months.")

            v1, v2 = st.columns(2)
            v1.metric("Out-of-sample error", "n/a" if np.isnan(info["err_loyo"]) else f"{info['err_loyo'] * 100:.0f}%",
                      help="Leave-one-year-out: the factor is fitted on the other overlap years and "
                           "used to predict each held-out year's VIIRS monthly counts.")
            if trend_raw is not None and trend_h is not None:
                v2.metric("Trend without vs with scaling",
                          f"{trend_raw[0]:+.1f}% vs {trend_h[0]:+.1f}% / yr",
                          help="Same area and years. Without scaling, the jump from MODIS to VIIRS "
                               "shows up as a fake trend in the record.")
            else:
                v2.metric("Trend without vs with scaling", "n/a",
                          help="Needs at least five complete years in the selected range.")

            ov = info["overlap"]
            ovv = ov[ov["valid"]] if len(ov) else ov
            if len(ovv):
                sc = go.Figure()
                sc.add_trace(go.Scatter(
                    x=ovv["MODIS"], y=ovv["VIIRS"], mode="markers",
                    marker=dict(color=VIIRS_COLOR, size=7, opacity=0.8), name="Month",
                    text=ovv["ym"].dt.strftime("%b %Y"),
                    hovertemplate="%{text}<br>MODIS %{x:.0f}<br>VIIRS %{y:.0f}<extra></extra>",
                ))
                xmax = float(ovv["MODIS"].max())
                sc.add_trace(go.Scatter(
                    x=[0, xmax], y=[0, xmax * info["factor"]], mode="lines",
                    line=dict(color=OTHER_COLOR, dash="dash"), name=f"VIIRS = {info['factor']:.2f} × MODIS"))
                sc.update_xaxes(title="MODIS monthly detections (per-satellite)")
                sc.update_yaxes(title="VIIRS monthly detections")
                show(tidy(sc, "Overlap months (whole file): MODIS against VIIRS", 360), "overlap")

            rm = info["raw_monthly"]
            rm = rm[(rm["ym"] >= pd.Timestamp(d0).to_period("M").start_time) & (rm["ym"] <= pd.Timestamp(d1))]
            if aoi_poly is None:
                raw_year = (rm.assign(year=rm["ym"].dt.year)
                            .groupby(["year", "sensor"], observed=True)["n"].sum().unstack(fill_value=0))
                cmp_ = go.Figure()
                if "MODIS" in raw_year:
                    cmp_.add_scatter(x=raw_year.index, y=raw_year["MODIS"], mode="lines+markers", name="MODIS, raw",
                                     line=dict(color=MODIS_COLOR, width=1.5, dash="dot"), marker=dict(size=4))
                if "VIIRS" in raw_year:
                    cmp_.add_scatter(x=raw_year.index, y=raw_year["VIIRS"], mode="lines+markers",
                                     name="VIIRS, raw (all satellites)",
                                     line=dict(color=VIIRS_COLOR, width=1.5, dash="dot"), marker=dict(size=4))
                cmp_.add_scatter(x=annual_total.index, y=annual_total.to_numpy(), mode="lines+markers",
                                 name="Harmonized", line=dict(color="#e8e8e8", width=3), marker=dict(size=6))
                cmp_.update_xaxes(dtick=2)
                cmp_.update_yaxes(title="detections / year")
                show(tidy(cmp_, "Before and after: the raw records against the harmonized series"), "beforeafter")
            else:
                st.caption("The raw-sensor comparison is shown for the whole region. "
                           "Choose Whole region to see it.")

            by_plat = rm.groupby(["ym", "platform"], observed=True)["n"].sum().unstack(fill_value=0)
            if len(by_plat):
                pf = px.line(by_plat, labels=dict(ym="", value="detections", platform="Satellite"))
                show(tidy(pf, "Raw monthly detections by satellite, whole region (before any correction)", 320), "platforms")
                st.caption(
                    "Steps in this chart happen when a satellite joins the constellation, not when fire "
                    "changes. That is why the app normalizes per satellite and defaults to SNPP only."
                )

            if scaling.startswith("By"):
                tbl = pd.DataFrame({"Month": [calendar.month_abbr[mo] for mo in range(1, 13)],
                                    "MODIS to VIIRS factor": [round(info["seasonal_map"][mo], 2) for mo in range(1, 13)]})
                with st.expander("Monthly scaling factors"):
                    st.dataframe(tbl, hide_index=True)

    with t_data:
        key = (match, str(d0), str(d1), drop_low, veg_only, viirs_mode, scaling, int(min_monthly), aoi_key)
        mt = (
            sub.groupby(["year", "month"])
            .agg(detections_viirs_equiv=("weight", "sum"), raw_detections=("weight", "size"))
            .reset_index()
        )
        mt["median_same_month"] = mt["month"].map(med_by_month)
        mt["complete_month"] = [bool(full.loc[y, mo]) for y, mo in zip(mt["year"], mt["month"])]
        mt["anomaly_pct"] = np.where(
            mt["complete_month"] & (mt["median_same_month"] > 0),
            (mt["detections_viirs_equiv"] - mt["median_same_month"]) / mt["median_same_month"] * 100, np.nan)
        mt = mt.round({"detections_viirs_equiv": 1, "median_same_month": 1, "anomaly_pct": 1})

        report = [
            f"# Fire activity summary: {region_text}",
            "",
            f"- Period: {d0} to {d1} (data through {dmax})",
            f"- Area: {'whole region bounding data' if aoi_poly is None else f'custom area, about {aoi_area:,.0f} km2'}",
            f"- Detections kept: {len(sub):,} (VIIRS-equivalent total {sub['weight'].sum():,.0f})",
            f"- Filters: low confidence {'dropped' if drop_low else 'kept'}, "
            f"{'vegetation fires only' if veg_only else 'all fire types'}, "
            f"VIIRS {'SNPP only' if viirs_mode.startswith('SNPP') else 'all satellites, normalized'}",
        ]
        if info["has_both"]:
            ci = info["factor_ci"]
            report += [
                f"- MODIS to VIIRS factor: {info['factor']:.2f}"
                + (f" (bootstrap range {ci[0]:.2f}-{ci[1]:.2f})" if ci else "")
                + f" from {info['n_valid']} overlap months; first full VIIRS month {info['split']:%B %Y}",
                f"- Overlap correlation r = {info['corr']:.2f}; leave-one-year-out error "
                f"{info['err_loyo'] * 100:.0f}%" if not np.isnan(info["corr"]) and not np.isnan(info["err_loyo"])
                else "- Overlap statistics unavailable for this file",
            ]
        if status is not None:
            report += ["", "## Current period",
                       f"{calendar.month_name[lm]} 1-{last_day}, {ly}: {current:,.0f} detections against a median of "
                       f"{status['median']:,.0f} for the same days in {len(hist)} earlier years "
                       f"({ordinal(status['rank'])} highest of {status['n']}; {status['label']})."]
        if trend_h is not None:
            report += ["", f"Annual trend over {len(full_years)} complete years: {trend_h[0]:+.1f}% per year (Theil-Sen)."]
        if not flat.empty:
            report += ["", "## Most active months"] + [
                f"- {calendar.month_abbr[int(r.month)]} {int(r.year)}: {r.n:,.0f}" for r in flat.nlargest(5, "n").itertuples()]
        report += ["", "Source: NASA FIRMS active-fire detections (MODIS Terra/Aqua, VIIRS SNPP/NOAA-20/NOAA-21)."]
        report_txt = "\n".join(report)

        stem = slug(match, canon) + ("_aoi" if aoi_poly is not None else "")
        d1c, d2c, d3c = st.columns(3)
        d1c.download_button("Hotspots (CSV)", to_csv(key, sub), file_name=f"{stem}_hotspots.csv", mime="text/csv",
                            help="Every detection in the current area and period, with sensor, satellite and "
                                 "the VIIRS-equivalent weight.")
        d2c.download_button("Monthly series with anomalies (CSV)", mt.to_csv(index=False).encode(),
                            file_name=f"{stem}_monthly.csv", mime="text/csv")
        d3c.download_button("Summary (Markdown)", report_txt.encode(), file_name=f"{stem}_summary.md",
                            mime="text/markdown")
        if aoi_poly is not None:
            st.download_button("Area of interest (GeoJSON)",
                               json.dumps({"type": "Feature", "properties": {}, "geometry": mapping(aoi_poly)}).encode(),
                               file_name=f"{stem}.geojson", mime="application/geo+json")
        st.dataframe(mt, hide_index=True)

else:
    render_map(None)
    if match is None:
        st.markdown("&nbsp;", unsafe_allow_html=True)
        s1, s2, s3 = st.columns(3, gap="large")
        for col, head, body in (
            (s1, "1. Choose a region",
             "Search for a country in the sidebar. Those marked “data loaded” already have a hotspot file."),
            (s2, "2. Zoom in on your area",
             "Draw a shape on the map or type coordinates to look at a province, a park or a stretch of farmland."),
            (s3, "3. Read the calendar",
             "See which months and years burned the most, and whether this month looks unusual."),
        ):
            col.markdown(f'<div class="step"><b>{head}</b><p>{body}</p></div>', unsafe_allow_html=True)
        avail = sorted(x.replace("_", " ").title() for x in stems)
        if avail:
            st.markdown(f'<p class="note" style="margin-top:1.2rem">Ready to explore: {", ".join(avail)}.</p>',
                        unsafe_allow_html=True)
        else:
            st.markdown('<p class="note" style="margin-top:1.2rem">No hotspot files found yet. '
                        "Add a FIRMS download to the data folder to get started.</p>", unsafe_allow_html=True)

st.markdown(
    '<div class="credits">Fire detections come from NASA FIRMS (MODIS and VIIRS). Country boundaries are from '
    "Natural Earth, and base maps are from Esri and OpenStreetMap contributors. "
    "Made for the 2026 NASA Space Apps Challenge.</div>",
    unsafe_allow_html=True,
)
