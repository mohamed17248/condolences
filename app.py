"""

The Burn Record made by Ahmed sultan & Mohamed abbas (queriesonhold/egobreaking & mohamed17248)
Run:    streamlit run app.py
Data:   FIRMS archive downloads (https://firms.modaps.eosdis.nasa.gov/download/)
        saved in ./data/<country>.csv or ./data/<country>.parquet
        
"""

from __future__ import annotations

import calendar
import hashlib
import inspect
import json
import re
import unicodedata
from datetime import date
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
from shapely.geometry import Point, Polygon, box, mapping, shape
from streamlit_folium import st_folium

st.set_page_config(
    page_title="The Burn Record",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
CACHE_DIR = DATA_DIR / ".cache"
NE_PATH = BASE_DIR / "ne_countries.geojson"
if not NE_PATH.exists():
    NE_PATH = Path("ne_countries.geojson")

MODIS_COLOR = "#5ca4d6"
VIIRS_COLOR = "#f06428"
OTHER_COLOR = "#928d82"
AOI_COLOR = "#4ecdc4"
SAGE_COLOR = "#8fb369"
DAY_COLOR = "#f2b138"
NIGHT_COLOR = "#5b8fd9"

FONT_UI = '"Plus Jakarta Sans", -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif'
FONT_DISPLAY = '"Fraunces", Georgia, "Times New Roman", serif'
FONT_MONO = '"JetBrains Mono", ui-monospace, SFMono-Regular, Menlo, monospace'
FONT_PLOT = "Plus Jakarta Sans, -apple-system, BlinkMacSystemFont, Segoe UI, sans-serif"

SENSOR_COLORS = {"MODIS": MODIS_COLOR, "VIIRS": VIIRS_COLOR, "UNKNOWN": OTHER_COLOR}
LEVEL_COLORS = {
    "much_more": "#f06428",
    "more": "#e0a938",
    "typical": "#6892b5",
    "less": "#8976bd",
    "much_less": "#6b57a6",
    "partial": "#585c55",
}
LEVEL_LABELS = {
    "low": "Below normal",
    "normal": "Normal range",
    "elevated": "Above normal",
    "high": "Unusually high",
    "much_more": "Much heavier",
    "more": "Heavier than usual",
    "typical": "Typical year",
    "less": "Quieter than usual",
    "much_less": "Much quieter",
    "partial": "Partial record",
}

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
    "russian federation": "russia",
    "falkland islands": "falkland is.",
}

# Microstates left out of 110m Natural Earth polygons must be done
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

# Keeping overseas territories from blowing up country zoom boxes, will mess up cords
OVERRIDES = {
    "france": (41.0, 51.5, -5.5, 9.8),
    "united states of america": (24.0, 50.0, -125.0, -66.5),
    "russia": (41.0, 70.0, 27.0, 180.0),
    "fiji": (-20.3, -15.5, 176.5, 180.0),
    "new zealand": (-47.5, -34.0, 166.0, 179.0),
}

WANTED = {
    "latitude", "longitude", "acq_date", "acq_time", "satellite", "instrument",
    "confidence", "brightness", "bright_ti4", "type", "frp",
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


def normalize_platform(sat: pd.Series) -> pd.Series:
    # Category-level mapping: avoids materializing millions of Python strings on huge countries like russia
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


def infer_sensor(df: pd.DataFrame, platform: pd.Series) -> pd.Series:
    # avoiding a 25M-row object-dtype string column. (integer codes)
    codes = np.full(len(df), 2, dtype=np.int8)  # 0=MODIS 1=VIIRS 2=UNKNOWN
    if "brightness" in df:
        codes[df["brightness"].notna().to_numpy()] = 0
    if "bright_ti4" in df:
        codes[df["bright_ti4"].notna().to_numpy()] = 1
    codes[platform.isin(MODIS_PLATFORMS).to_numpy()] = 0
    codes[platform.isin(VIIRS_PLATFORMS).to_numpy()] = 1
    if "instrument" in df:
        ins = df["instrument"]
        if isinstance(ins.dtype, pd.CategoricalDtype):
            cats = ins.cat.categories.astype(str).str.upper()
            modis_cats = set(ins.cat.categories[np.asarray(cats.str.contains("MODIS", na=False))])
            viirs_cats = set(ins.cat.categories[np.asarray(cats.str.contains("VIIRS", na=False))])
            codes[ins.isin(modis_cats).to_numpy()] = 0
            codes[ins.isin(viirs_cats).to_numpy()] = 1
        else:
            ins_up = ins.astype(str).str.upper()
            codes[ins_up.str.contains("MODIS", na=False).to_numpy()] = 0
            codes[ins_up.str.contains("VIIRS", na=False).to_numpy()] = 1
    return pd.Categorical.from_codes(codes, categories=["MODIS", "VIIRS", "UNKNOWN"])


def _read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        df = pd.read_parquet(path)
        df.columns = [c.strip().lower() for c in df.columns]
        return df[[c for c in df.columns if c in WANTED]].copy()
    header = pd.read_csv(path, nrows=0).columns
    keep = [c for c in header if c.strip().lower() in WANTED]
    try:
        # The pyarrow reader parses multi-million-row CSVs several times faster
        df = pd.read_csv(path, usecols=keep, engine="pyarrow")
    except Exception:
        text_cols = {c: str for c in keep if c.strip().lower() in ("satellite", "instrument", "confidence")}
        df = pd.read_csv(path, usecols=keep, dtype=text_cols, low_memory=False)
    df.columns = [c.strip().lower() for c in df.columns]
    return df


def _to_date(s: pd.Series) -> pd.Series:
    if pd.api.types.is_datetime64_any_dtype(s):
        return s.dt.normalize()
    d = pd.to_datetime(s, format="%Y-%m-%d", errors="coerce")
    if d.isna().mean() > 0.01:
        d = pd.to_datetime(s, errors="coerce")
    return d.dt.normalize()


def _parse_file(path: Path):
    df = _read_table(path)
    df["acq_date"] = _to_date(df["acq_date"])
    df = df.dropna(subset=["latitude", "longitude", "acq_date"])
    for c in ("satellite", "instrument", "confidence"):
        if c in df:
            # category keeps multi-million-row files (Russia, Angola) in RAM
            df[c] = df[c].astype("category")

    subset = [c for c in ("latitude", "longitude", "acq_date", "acq_time", "satellite") if c in df]
    df = df.drop_duplicates(subset=subset)

    platform = normalize_platform(df["satellite"]) if "satellite" in df else pd.Series("UNKNOWN", index=df.index)
    sensor = infer_sensor(df, platform)

    # MODIS confidence is 0-100, VIIRS uses l/n/h
    lowconf = pd.Series(False, index=df.index)
    if "confidence" in df:
        c = df["confidence"]
        if isinstance(c.dtype, pd.CategoricalDtype):
            cats = c.cat.categories.astype(str).str.strip().str.lower()
            flags = np.asarray(
                (pd.to_numeric(cats, errors="coerce") < 30) | cats.isin(["l", "low"])
            ).astype(bool)
            codes = c.cat.codes.to_numpy()
            if len(flags):
                low = flags[np.clip(codes, 0, None)] & (codes >= 0)
            else:
                low = np.zeros(len(df), dtype=bool)
            lowconf = pd.Series(low, index=df.index)
        else:
            c = c.astype(str).str.strip().str.lower()
            lowconf = (pd.to_numeric(c, errors="coerce") < 30) | c.isin(["l", "low"])

    # FIRMS type: 0 = vegetation fire, 1-3 = volcano / industrial / offshore | dectating the type
    firms_type = pd.Series(np.nan, index=df.index, dtype="float32")
    nonveg = pd.Series(False, index=df.index)
    if "type" in df:
        t = pd.to_numeric(df["type"], errors="coerce")
        nonveg = t.notna() & (t != 0)
        firms_type = t.astype("float32")

    # Convert UTC acq_time + longitude into approximate local solar hour (0..23, -1 if missing##)
    local_hour = np.full(len(df), -1, dtype=np.int8)
    if "acq_time" in df:
        t_num = pd.to_numeric(df["acq_time"], errors="coerce")
        valid_t = t_num.notna() & df["longitude"].notna()
        if valid_t.any():
            hh = np.floor(t_num[valid_t].to_numpy(float) / 100.0).clip(0, 23)
            mm = (t_num[valid_t].to_numpy(float) % 100.0).clip(0, 59)
            lon = df.loc[valid_t, "longitude"].to_numpy(float)
            solar = np.floor((hh + mm / 60.0 + lon / 15.0) % 24.0).astype(np.int8)
            local_hour[valid_t.to_numpy()] = solar

    frp = (
        pd.to_numeric(df["frp"], errors="coerce")
        if "frp" in df
        else pd.Series(0.0, index=df.index)
    )
    frp = frp.clip(lower=0).fillna(0).astype("float32")

    out = pd.DataFrame({
        "latitude": df["latitude"].astype("float32"),
        "longitude": df["longitude"].astype("float32"),
        "acq_date": df["acq_date"],
        "sensor": sensor.astype("category"),
        "platform": platform.astype("category"),
        "local_hour": local_hour,
        "lowconf": lowconf.to_numpy(),
        "nonveg": nonveg.to_numpy(),
        "firms_type": firms_type.to_numpy(),
        "frp": frp.to_numpy(),
    }).reset_index(drop=True)
    flags = {
        "has_conf": "confidence" in df,
        "has_type": "type" in df,
        "has_platform": "satellite" in df,
        "has_time": bool((local_hour >= 0).any()),
        "n_raw": int(len(out)),
        "n_nonveg_total": int(nonveg.sum()),
        "n_lowconf_total": int(lowconf.sum()),
        "has_frp": bool((frp > 0).any()),
    }
    return out, flags


def _cache_paths(path: Path):
    stat = path.stat()
    tag = f"{path.stem}.{stat.st_size}.{int(stat.st_mtime)}"
    return CACHE_DIR / f"{tag}.parquet", CACHE_DIR / f"{tag}.json"


@st.cache_resource(show_spinner="Reading the hotspot file (first time only)…", max_entries=2)
def read_file(path_str: str, mtime: float):
    """Parse a FIRMS file once. A cleaned copy is kept in data/.cache so later
    launches of the app load in seconds instead of re-parsing the CSV."""
    path = Path(path_str)
    pq_path, js_path = _cache_paths(path)
    if pq_path.exists() and js_path.exists():
        try:
            return pd.read_parquet(pq_path), json.loads(js_path.read_text())
        except Exception:
            pass  # a damaged cache just means we parse again
    out, flags = _parse_file(path)
    try:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        for old in CACHE_DIR.glob(f"{path.stem}.*"):
            old.unlink(missing_ok=True)
        out.to_parquet(pq_path, index=False)
        js_path.write_text(json.dumps(flags))
    except Exception:
        pass  # read-only folder: the in-memory cache still works
    return out, flags


# Approximate FIRMS data start per platform (launch + commissioning). A quiet
# region's first detection can lag the real start by years, which would
# under-count the active constellation — weights use these, not first_seen.
PLATFORM_START = {
    "Terra": pd.Timestamp("2000-11-01"),
    "Aqua": pd.Timestamp("2002-07-01"),
    "SNPP": pd.Timestamp("2012-01-01"),
    "NOAA-20": pd.Timestamp("2018-01-01"),
    "NOAA-21": pd.Timestamp("2023-01-01"),
}


@st.cache_resource(show_spinner="Calibrating MODIS and VIIRS records…", max_entries=3)
def build_dataset(path_str, mtime, drop_low, veg_only, viirs_mode, seasonal, min_monthly, calib_end_year=None):
    raw, flags = read_file(path_str, mtime)
    first_seen = raw.groupby("platform", observed=True)["acq_date"].min()
    active_since = first_seen.copy()
    for plat, start in PLATFORM_START.items():
        if plat in active_since.index:
            active_since.loc[plat] = min(active_since.loc[plat], start)

    df = raw
    if drop_low:
        df = df[~df["lowconf"]]
    if veg_only:
        df = df[~df["nonveg"]]
    df = df.drop(columns=["lowconf", "nonveg"]).copy()
    n_after_qc = len(df)

    df["ym"] = df["acq_date"].values.astype("datetime64[M]").astype("datetime64[ns]")
    raw_monthly = df.groupby(["ym", "sensor", "platform"], observed=True).size().reset_index(name="n")

    if viirs_mode == "snpp" and ((df["sensor"] == "VIIRS") & (df["platform"] == "SNPP")).any():
        df = df[(df["sensor"] != "VIIRS") | (df["platform"] == "SNPP")]

    # Normalize each detection by active constellation size on that date
    w0 = np.ones(len(df))
    for sensor in ("MODIS", "VIIRS"):
        mask = (df["sensor"] == sensor).to_numpy()
        if not mask.any():
            continue
        plats = [str(p) for p in df.loc[mask, "platform"].unique()]
        firsts = np.sort(active_since.reindex(plats).to_numpy(dtype="datetime64[ns]"))
        dates = df.loc[mask, "acq_date"].to_numpy(dtype="datetime64[ns]")
        n_active = np.searchsorted(firsts, dates, side="right")
        w0[mask] = 1.0 / np.maximum(n_active, 1)
    df["w0"] = w0

    modis = (df["sensor"] == "MODIS").to_numpy()
    viirs = (df["sensor"] == "VIIRS").to_numpy()
    has_both = modis.any() and viirs.any()

    info = {
        "has_both": has_both,
        "factor": 1.0,
        "factor_ci": None,
        "seasonal_map": {m: 1.0 for m in range(1, 13)},
        "split": None,
        "overlap": pd.DataFrame(),
        "corr": np.nan,
        "err_single": np.nan,
        "err_seasonal": np.nan,
        "err_loyo": np.nan,
        "n_valid": 0,
        "raw_monthly": raw_monthly,
        "flags": flags,
        "n_raw": flags["n_raw"],
        "n_after_qc": n_after_qc,
        "first_seen": {str(k): v for k, v in first_seen.items() if str(k) != "UNKNOWN"},
        "platforms": {
            s: sorted(map(str, df.loc[df["sensor"] == s, "platform"].unique()))
            for s in ("MODIS", "VIIRS")
        },
    }

    f = np.ones(len(df))
    keep = np.ones(len(df), dtype=bool)

    if has_both:
        viirs_first = df.loc[viirs, "acq_date"].min()
        split = viirs_first + pd.offsets.MonthBegin(0)
        info["split"] = split

        monthly = df.groupby(["ym", "sensor"], observed=True)["w0"].sum().unstack(fill_value=0)
        for s in ("MODIS", "VIIRS"):
            if s not in monthly:
                monthly[s] = 0.0
        ov = monthly.loc[monthly.index >= split, ["MODIS", "VIIRS"]].copy()

        # Recent overlap months are excluded from the fit: Terra/Aqua orbit
        # drift since ~2020 shifts overpass times and biases the ratio.
        year_ok = ov.index.year <= calib_end_year if calib_end_year else True
        valid = (ov["MODIS"] >= min_monthly) & (ov["VIIRS"] >= min_monthly) & year_ok
        if valid.sum() < 6:
            valid = (ov["MODIS"] > 0) & (ov["VIIRS"] > 0) & year_ok
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

            # Leave one year out sanity check so we don't overfit
            errs = []
            for y in np.unique(yrs):
                te = yrs == y
                tr = ~te
                if tr.sum() >= 3:
                    fy = float(np.median(v_arr[tr] / m_arr[tr]))
                    errs.append(float(np.median(np.abs(m_arr[te] * fy - v_arr[te]) / v_arr[te])))
            if errs:
                info["err_loyo"] = float(np.median(errs))

        info.update(
            factor=factor,
            seasonal_map=seasonal_map,
            overlap=ov.reset_index(),
            n_valid=int(valid.sum()),
        )

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
    # FRP is fire intensity (MW), not a detection count: average it across the active constellation via w0 but do not apply the MODIS to VIIRS rate factor.
    df["frp_w"] = df["frp"].to_numpy(float) * df["w0"].to_numpy(float)
    df = df[keep].drop(columns=["w0"])
    df["year"] = df["acq_date"].dt.year
    df["month"] = df["acq_date"].dt.month
    df = df.reset_index(drop=True)
    info["n_final"] = len(df)
    info["dmin"] = df["acq_date"].min().date() if len(df) else None
    info["dmax"] = df["acq_date"].max().date() if len(df) else None
    return df, info


@st.cache_resource(show_spinner=False, max_entries=6)
def get_sub(path_str, mtime, drop_low, veg_only, viirs_mode, seasonal, min_monthly, d0, d1, aoi_wkt, calib_end_year=None):
    """The cleaned record restricted to a date range and (optionally) an area.
    Cached as a resource so reruns reuse the same frame instead of re-filtering
    millions of rows. Treat the result as read-only."""
    hot, info = build_dataset(path_str, mtime, drop_low, veg_only, viirs_mode, seasonal, min_monthly, calib_end_year)
    lo, hi = pd.Timestamp(d0), pd.Timestamp(d1)
    if info["dmin"] is not None and lo.date() <= info["dmin"] and hi.date() >= info["dmax"]:
        sub = hot  # full range: no copy
    else:
        sub = hot[(hot["acq_date"] >= lo) & (hot["acq_date"] <= hi)]
    if aoi_wkt:
        sub = apply_aoi(sub, shapely_wkt.loads(aoi_wkt))
    return sub


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


def circle_polygon(lon, lat, radius_m, n=72):
    ang = np.linspace(0, 2 * np.pi, n, endpoint=False)
    dlat = (radius_m / 111320.0) * np.sin(ang)
    dlon = (radius_m / (111320.0 * max(np.cos(np.radians(lat)), 0.05))) * np.cos(ang)
    return Polygon(list(zip(lon + dlon, lat + dlat)))


def drawing_to_polygon(feat):
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
    return shapely.contains_xy(poly, x, y)


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
MIN_RATED_YEARS = 4  #complete years needed before a year for it to be rated usual


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


def _month_masks(pivot, lo, hi, tol=pd.Timedelta(3, unit="D")):
    """Which year/month cells are touched by, and fully inside, a date range."""
    covered = pd.DataFrame(False, index=pivot.index, columns=pivot.columns)
    full = covered.copy()
    for y in pivot.index:
        for mo in pivot.columns:
            s = pd.Timestamp(y, mo, 1)
            e = s + pd.offsets.MonthEnd(0)
            covered.loc[y, mo] = (e >= lo) and (s <= hi)
            full.loc[y, mo] = (s + tol >= lo) and (e - tol <= hi)
    return covered, full


@st.cache_data(show_spinner="Checking the full archive…", max_entries=6)
def archive_year_stats(path_str, mtime, aoi_wkt, drop_low, veg_only, viirs_mode, seasonal, min_monthly, calib_end_year=None):
    """Year-verdict stats over the whole cleaned archive, ignoring the date
    slider, so the annual stripes stay meaningful for short windows."""
    hot, _ = build_dataset(path_str, mtime, drop_low, veg_only, viirs_mode, seasonal, min_monthly, calib_end_year)
    arch = apply_aoi(hot, shapely_wkt.loads(aoi_wkt)) if aoi_wkt else hot
    if arch.empty:
        return None
    first_y, last_y = int(arch["year"].min()), int(arch["year"].max())
    years = list(range(first_y, last_y + 1))
    ad0, ad1 = arch["acq_date"].min(), arch["acq_date"].max()

    monthly = arch.groupby(["year", "month"])["weight"].sum()
    pivot = monthly.unstack("month").reindex(index=years, columns=range(1, 13))
    covered, full = _month_masks(pivot, ad0, ad1)
    pivot_cov = pivot.fillna(0).where(covered)
    pivot_full = pivot.fillna(0).where(full)
    med_by_month = pivot_full.median()
    full_years = [y for y in years if full.loc[y].all()]

    abs_year = (
        arch.groupby(["year", "sensor"], observed=True)["weight"]
        .sum().unstack(fill_value=0).reindex(years, fill_value=0)
    )
    annual_total = abs_year.sum(axis=1)
    modis_share = (
        (abs_year["MODIS"] / annual_total)
        if "MODIS" in abs_year
        else pd.Series(0.0, index=annual_total.index)
    )

    latest = arch["acq_date"].max()
    aly, alm, ald = latest.year, latest.month, latest.day
    ytd = None
    past_full = [y for y in full_years if y < aly]
    if aly not in full_years and len(past_full) >= MIN_RATED_YEARS:
        upto = (arch["month"] < alm) | ((arch["month"] == alm) & (arch["acq_date"].dt.day <= ald))
        yb = arch[upto].groupby("year")["weight"].sum()
        ytd = dict(year=aly, cur=float(yb.get(aly, 0.0)),
                   hist=yb.reindex(past_full, fill_value=0.0),
                   label=f"{calendar.month_abbr[alm]} {ald}")

    verdicts = year_verdicts(annual_total, modis_share, pivot, full_years, med_by_month, ytd)
    baseline = float(annual_total.loc[full_years].median()) if len(full_years) >= MIN_RATED_YEARS else None
    return verdicts, baseline, pivot_cov, pivot_full


@st.cache_data(show_spinner="Checking unusual-activity days…", max_entries=6)
def archive_alert_days(path_str, mtime, aoi_wkt, drop_low, veg_only, viirs_mode, seasonal, min_monthly, calib_end_year=None):
    """Days whose activity ran far above the same day-of-year in other years.
    Archive-scoped like archive_year_stats so short windows still get alerts."""
    hot, _ = build_dataset(path_str, mtime, drop_low, veg_only, viirs_mode, seasonal, min_monthly, calib_end_year)
    arch = apply_aoi(hot, shapely_wkt.loads(aoi_wkt)) if aoi_wkt else hot
    if arch.empty:
        return None

    first_y, last_y = int(arch["year"].min()), int(arch["year"].max())
    years = list(range(first_y, last_y + 1))
    ad0, ad1 = arch["acq_date"].min(), arch["acq_date"].max()

    monthly = arch.groupby(["year", "month"])["weight"].sum()
    pivot = monthly.unstack("month").reindex(index=years, columns=range(1, 13))
    _, full = _month_masks(pivot, ad0, ad1)
    full_years = [y for y in years if full.loc[y].all()]
    if len(full_years) < 3:
        return None

    # Plain arrays instead of frame.assign(): no copy of every column on big files
    doy = np.minimum(arch["acq_date"].dt.dayofyear.to_numpy(), 365)
    daily = (
        pd.DataFrame({"year": arch["year"].to_numpy(), "doy": doy, "w": arch["weight"].to_numpy()})
        .groupby(["year", "doy"])["w"]
        .sum()
        .unstack("doy")
        .reindex(index=years, columns=range(1, 366), fill_value=0.0)
    )

    latest = arch["acq_date"].max()
    aly = int(latest.year)
    aly_doy = int(min(latest.dayofyear, 365))
    cand_years = list(full_years)
    if aly not in full_years:
        cand_years.append(aly)

    rows = []
    for y in cand_years:
        others = [o for o in full_years if o != y]
        if not others or y not in daily.index:
            continue
        med = daily.loc[others].median()
        vals = daily.loc[y]
        lim = aly_doy if (y == aly and y not in full_years) else 365
        v, m = vals.iloc[:lim], med.iloc[:lim]
        mask = (v >= 10) & (m > 0) & (v >= 2.0 * m)
        for d in np.nonzero(mask.to_numpy())[0]:
            rows.append(dict(year=y, doy=int(d) + 1,
                             detections=float(v.iloc[d]), usual=float(m.iloc[d])))
    if not rows:
        return None

    out = pd.DataFrame(rows)
    out["date"] = pd.to_datetime("2021-01-01") + pd.to_timedelta(out["doy"] - 1, unit="D")
    out["ratio"] = out["detections"] / out["usual"]
    out["excess"] = out["detections"] - out["usual"]
    out = out.sort_values("excess", ascending=False).head(12)
    out["Year"] = out["year"].astype(int)
    out["Date"] = out["date"].dt.strftime("%d %b") + " " + out["Year"].astype(str)
    out["Detections"] = out["detections"].round(0).astype(int)
    out["Usual for that day"] = out["usual"].round(0).astype(int)
    out["Times usual"] = out["ratio"].round(1).map(lambda r: f"{r:.1f}×")
    return out[["Date", "Detections", "Usual for that day", "Times usual"]]


def year_verdicts(annual_total, modis_share, pivot_cov, full_years, med_by_month, ytd=None):
    years = sorted(int(y) for y in annual_total.index)
    full = [y for y in full_years if y in annual_total.index]
    rated = len(full) >= MIN_RATED_YEARS
    baseline = float(annual_total.loc[full].median()) if rated else None
    has_clim = bool(med_by_month.notna().any()) and float(med_by_month.max()) > 0
    typ_peak = int(med_by_month.idxmax()) if has_clim else None
    clim_total = float(med_by_month.sum()) if has_clim else 0.0

    out, run_dir, run_len, prev = [], 0, 0, None
    for y in years:
        total = float(annual_total.loc[y])
        rec = dict(
            year=y, kind="partial", total=total, ratio=None, level="partial",
            headline="Partial year", detail="", short="not rated", rank=None, n=len(full),
        )

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
                kind="ytd",
                ratio=ratio,
                level=level,
                headline=yhead,
                rank=rank,
                short=(short_text(ratio) + " so far").strip(),
                detail=(
                    f"{fmt(cur)} detections from January 1 to {ytd['label']}, "
                    f"{size_text(ratio, 'the usual for these dates')}. "
                    f"{rank_text(rank, len(hist) + 1, 'years for the same dates')}"
                ),
            )
        elif y in full and rated:
            ratio = total / baseline if baseline > 0 else None
            if ratio is None:
                level, head = ("more", YEAR_LEVELS[1][2]) if total > 0 else ("typical", YEAR_LEVELS[2][2])
            else:
                level, head, _ = level_for(ratio)
            rank = int((annual_total.loc[full] > total).sum()) + 1
            tail = f" ({fmt(baseline)})." if baseline > 0 else "."
            parts = [
                f"{fmt(total)} detections, {size_text(ratio)}{tail}",
                rank_text(rank, len(full), "complete years"),
            ]

            row = pivot_cov.loc[y].astype(float) if y in pivot_cov.index else None
            if row is not None and row.notna().any() and float(row.max()) > 0 and typ_peak is not None:
                pm = int(row.idxmax())
                # Shortest signed month offset on a 12-month circle
                shift = ((pm - typ_peak + 6) % 12) - 6
                if shift == 0:
                    timing = "as usual"
                else:
                    timing = (
                        f"{abs(shift)} month{'s' if abs(shift) > 1 else ''} "
                        f"{'later' if shift > 0 else 'earlier'} than usual"
                    )
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
                parts.append(
                    f"That is the {ordinal(run_len)} year in a row {'above' if direction > 0 else 'below'} usual."
                )
            if float(modis_share.get(y, 0.0)) > 0.5:
                parts.append("Mostly estimated from scaled MODIS data, so treat the size as approximate.")
            rec.update(
                kind="full",
                ratio=ratio,
                level=level,
                headline=head,
                rank=rank,
                short=short_text(ratio) or "no usual level",
                detail=" ".join(parts),
            )
        else:
            valid = pivot_cov.loc[y].dropna() if y in pivot_cov.index else pd.Series(dtype=float)
            first_m = int(valid.index.min()) if len(valid) else None
            if y in full and not rated:
                rec["detail"] = f"Fewer than {MIN_RATED_YEARS} complete years are loaded, so years are not rated yet."
            elif y == years[0] and first_m and first_m > 1:
                rec["detail"] = f"Data for this year starts in {calendar.month_name[first_m]}, so it is not rated."
            else:
                rec["detail"] = "Only part of this year is covered, so it is not rated."
        out.append(rec)
    return out


def years_summary(vs, region_text, baseline):
    full = [v for v in vs if v["kind"] == "full"]
    if len(full) < MIN_RATED_YEARS or baseline is None:
        return f"Load at least {MIN_RATED_YEARS} complete years of data to rate each year against the usual one."
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


def _arch_span(verdicts, sub) -> int:
    if verdicts:
        return max(v["year"] for v in verdicts) - min(v["year"] for v in verdicts)
    return int(sub["acq_date"].max().year - sub["acq_date"].min().year)


def derive_field_notes(sub: pd.DataFrame, med_by_month: pd.Series, info: dict, verdicts: list, region_text: str, veg_only: bool = True):
    """Synthesize plain-English fire regime clues, diurnal clock stats, and archive records from live data."""
    notes = []
    tot_w = float(sub["weight"].sum())
    if tot_w <= 0:
        return notes

    # 1. Fire regime & likely driver signature (small-fire VIIRS/MODIS jump + seasonal sharpness + spatial clustering)
    clim = med_by_month.fillna(0.0)
    clim_sum = float(clim.sum())
    top2_share = float(clim.nlargest(2).sum() / clim_sum) if clim_sum > 0 else 0.0
    factor = float(info.get("factor", 1.0))

    # Check if there is a distinct secondary peak separated by >= 4 months
    bimodal_pair = None
    if clim_sum > 0:
        m1 = int(clim.idxmax())
        v1 = float(clim.loc[m1])
        if v1 > 0:
            for m2 in range(1, 13):
                dist = min(abs(m2 - m1), 12 - abs(m2 - m1))
                if dist >= 4 and float(clim.loc[m2]) >= 0.45 * v1:
                    bimodal_pair = (calendar.month_name[m1], calendar.month_name[m2])
                    break

    # Spatial concentration on a 0.25-deg grid
    g = pd.DataFrame({
        "la": np.floor(sub["latitude"].to_numpy() * 4) / 4,
        "lo": np.floor(sub["longitude"].to_numpy() * 4) / 4,
        "w": sub["weight"].to_numpy(),
    }).groupby(["la", "lo"])["w"].sum().sort_values(ascending=False)
    top_cells_n = max(1, int(np.ceil(len(g) * 0.10)))
    spatial_conc = float(g.iloc[:top_cells_n].sum() / g.sum()) if len(g) > 0 else 0.0

    # Heuristic only: VIIRS catches tiny burns MODIS misses, so a high factor plus a
    # sharp seasonal window is *consistent with* crop-residue burning, not proof of it.
    if info.get("has_both") and factor >= 3.2 and top2_share >= 0.50:
        driver_tag = "Heuristic: likely crop-residue burning"
        driver_head = "Small, brief fires clustered around harvest windows"
        driver_body = (
            f"VIIRS picks up {factor:.1f}× more hotspots here than 1 km MODIS, and {top2_share * 100:.0f}% "
            f"of typical annual burning lands in just two peak months — a pattern consistent with "
            f"post-harvest agricultural stubble clearing, though other small-fire regimes can look similar."
        )
    elif bimodal_pair is not None:
        driver_tag = "Two-season burn cycle"
        driver_head = f"Twin activity peaks in {bimodal_pair[0]} and {bimodal_pair[1]}"
        driver_body = (
            f"Instead of a single dry-season spike, burning rises twice a year—typically reflecting "
            f"two distinct crop harvest rotations or split rainfall regimes across {region_text}."
        )
    elif info.get("has_both") and factor >= 2.7:
        driver_tag = "Small-scale & fragmented fires"
        driver_head = "Dominated by small burns below MODIS resolution"
        driver_body = (
            f"With a {factor:.1f}× calibration jump between MODIS and VIIRS, most detections here come from "
            f"small, fragmented burns—common in rotational farming, pasture maintenance, and mosaic landscapes."
        )
    elif top2_share >= 0.62:
        driver_tag = "Tight dry-season fire regime"
        driver_head = f"{top2_share * 100:.0f}% of burning happens in two peak months"
        driver_body = (
            f"Activity stays quiet most of the year before surging sharply once vegetation dries out. "
            f"In landscapes like this, year-to-year swings usually track rainfall deficits and fuel dryness."
        )
    else:
        driver_tag = "Broad seasonal regime"
        conc_txt = (
            f"concentrated heavily in a few local belts ({spatial_conc * 100:.0f}% of activity in 10% of active cells)"
            if spatial_conc >= 0.55
            else "spread widely across the landscape"
        )
        driver_head = "Burning stretches across multiple months"
        driver_body = (
            f"Detections in {region_text} follow a gradual seasonal curve rather than a single brief spike, "
            f"and hotspots are {conc_txt}."
        )
    notes.append(dict(key="regime", tag=driver_tag, head=driver_head, body=driver_body))

    # 2. Local solar clock (daytime vs overnight burning)
    lh = sub["local_hour"].to_numpy()
    valid_lh = lh >= 0
    if valid_lh.any():
        hrs = lh[valid_lh]
        wts = sub["weight"].to_numpy()[valid_lh]
        by_hr = np.bincount(hrs, weights=wts, minlength=24)
        peak_hr = int(np.argmax(by_hr))
        day_share = float(by_hr[6:18].sum() / by_hr.sum())
        night_share = 1.0 - day_share
        # This histogram mostly reflects fixed satellite overpass times, not fire behaviour.
        if night_share >= 0.28:
            clock_tag = "Overpass diagnostic: night-weighted"
            clock_head = f"{night_share * 100:.0f}% of weight sits in the 18:00–06:00 window"
            clock_body = (
                f"MODIS and VIIRS only observe at fixed sun-synchronous times (~10:30/13:30 and "
                f"~22:30/01:30 local), so this mostly tracks which overpasses see more fire, not when "
                f"fires actually burn. Treat night weighting as a sensor-geometry artefact."
            )
        else:
            clock_tag = "Overpass diagnostic: day-weighted"
            clock_head = f"{day_share * 100:.0f}% of weight sits in the 06:00–18:00 window"
            clock_body = (
                f"As with the night-weighted case, the 06:00–18:00 share largely reflects the fixed "
                f"afternoon overpass schedule of the constellation rather than real diurnal fire behaviour."
            )
        notes.append(dict(key="clock", tag=clock_tag, head=clock_head, body=clock_body))

    # 3. Single-day archive spike & multi-year streaks
    daily_totals = sub.groupby("acq_date")["weight"].sum()
    if len(daily_totals) > 0:
        busiest_day = daily_totals.idxmax()
        busiest_val = float(daily_totals.max())
        full_v = [v for v in verdicts if v["kind"] == "full"]
        streak_txt = ""
        if len(full_v) >= 4:
            # Longest consecutive run above or below median
            best_len, best_dir = 0, ""
            cur_len, cur_dir = 0, ""
            for v in full_v:
                d = "above" if v["level"] in ("more", "much_more") else ("below" if v["level"] in ("less", "much_less") else "")
                if d and d == cur_dir:
                    cur_len += 1
                else:
                    cur_len = 1 if d else 0
                    cur_dir = d
                if cur_len > best_len:
                    best_len, best_dir = cur_len, cur_dir
            if best_len >= 3:
                streak_txt = f" The longest multi-year run in the archive is {best_len} consecutive years {best_dir} the median."
        notes.append(
            dict(
                key="record",
                tag="Archive extreme",
                head=f"Single busiest day: {busiest_day:%d %b %Y}",
                body=(
                    f"Satellites logged {fmt(busiest_val)} VIIRS-equivalent detections in a single 24-hour UTC window "
                    f"on {busiest_day:%B %d, %Y}.{streak_txt}"
                ),
            )
        )

    # 4. Orbital & sensor physics tidbit
    flags = info.get("flags", {})
    n_nonveg = int(flags.get("n_nonveg_total", 0))
    first_seen = info.get("first_seen", {})
    if veg_only and n_nonveg > 0:
        orb_tag = "Static heat masked"
        orb_head = f"{fmt(n_nonveg)} non-vegetation heat sources filtered out"
        orb_body = (
            f"NASA's FIRMS classifier flagged {fmt(n_nonveg)} static thermal anomalies in this archive "
            f"(gas flares, cement kilns, smelters, or volcanic vents); the vegetation-only filter removed them."
        )
    elif "Terra" in first_seen and "SNPP" in first_seen:
        orb_tag = "Orbital handoff"
        orb_head = f"{_arch_span(verdicts, sub)}+ years bridged across {len(first_seen)} spacecraft"
        orb_body = (
            f"NASA Terra began logging hotspots here in {first_seen['Terra']:%b %Y}, joined by Aqua in "
            f"{first_seen.get('Aqua', first_seen['Terra']):%b %Y}, before Suomi-NPP VIIRS took over in "
            f"{first_seen['SNPP']:%b %Y} at 375 m resolution."
        )
    else:
        lat_mid = float(sub["latitude"].mean())
        hemi = "Northern" if lat_mid >= 0 else "Southern"
        orb_tag = "Orbital geometry"
        orb_head = f"Sun-synchronous passes at {abs(lat_mid):.1f}° {'N' if lat_mid >= 0 else 'S'}"
        orb_body = (
            f"Polar-orbiting MODIS and VIIRS instruments cross this {hemi}-Hemisphere latitude on fixed "
            f"sun-synchronous schedules (Terra ~10:30/22:30, Aqua & VIIRS ~13:30/01:30 local), capturing "
            f"active flame fronts from 700+ km up."
        )
    notes.append(dict(key="orbital", tag=orb_tag, head=orb_head, body=orb_body))
    return notes


def polygon_area_km2(poly) -> float:
    try:
        return float(gpd.GeoSeries([poly], crs="EPSG:4326").to_crs("EPSG:6933").area.iloc[0] / 1e6)
    except Exception:
        return float("nan")


@st.cache_resource(show_spinner=False)
def load_regions():
    # cache_resource hands back the same objects every rerun; cache_data would
    # unpickle ~250 geometries on each click.
    world = gpd.read_file(NE_PATH)
    name_col = next(c for c in ("NAME", "name", "NAME_EN", "ADMIN") if c in world.columns)
    regs = {}
    geoms_wkb = {}
    for name, geom in zip(world[name_col], world.geometry):
        if geom is None or geom.is_empty:
            continue
        key = str(name).lower()
        minx, miny, maxx, maxy = geom.bounds
        regs[key] = (miny, maxy, minx, maxx)
        geoms_wkb[key] = geom.wkb

    canon = {}
    for alias, target in ALIASES.items():
        if target in regs:
            canon[alias] = target
            regs.setdefault(alias, regs[target])
            if target in geoms_wkb:
                geoms_wkb.setdefault(alias, geoms_wkb[target])
    for name, bbox in MANUAL_REGIONS.items():
        regs.setdefault(name, bbox)
    regs.update(OVERRIDES)
    return regs, canon, geoms_wkb


def slug(key: str, canon: dict) -> str:
    s = unicodedata.normalize("NFKD", canon.get(key, key)).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")


def data_stems() -> set:
    if not DATA_DIR.exists():
        return set()
    return {p.stem for p in DATA_DIR.iterdir() if p.is_file() and p.suffix.lower() in (".csv", ".parquet")}


def find_data_file(key: str, canon: dict):
    for stem in (slug(key, canon), key.replace(" ", "_")):
        for ext in (".parquet", ".csv"):
            p = DATA_DIR / f"{stem}{ext}"
            if p.exists():
                return p
    return None


_PLOT_HAS_WIDTH = "width" in inspect.signature(st.plotly_chart).parameters


def show(fig, key):
    if _PLOT_HAS_WIDTH:
        st.plotly_chart(fig, width="stretch", key=key)
    else:
        st.plotly_chart(fig, use_container_width=True, key=key)


def tidy(fig, title, height=380):
    fig.update_layout(
        title=dict(
            text=title,
            x=0,
            xanchor="left",
            y=0.98,
            yanchor="top",
            yref="container",
            font=dict(size=14, color="#f4f0e6"),
        ),
        height=height,
        margin=dict(l=6, r=6, t=82, b=8),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(
            orientation="h",
            x=0,
            xanchor="left",
            y=1.0,
            yanchor="bottom",
            title=None,
            font=dict(size=11, color="#cbc5b9"),
        ),
        hoverlabel=dict(
            bgcolor="#171916",
            bordercolor="rgba(244,240,230,0.28)",
            font=dict(size=12, family=FONT_PLOT, color="#f4f0e6"),
        ),
        font=dict(family=FONT_PLOT, size=12, color="#cbc5b9"),
    )
    fig.update_xaxes(
        showgrid=False,
        automargin=True,
        linecolor="rgba(244,240,230,0.16)",
        tickfont=dict(color="#928d82"),
    )
    fig.update_yaxes(
        gridcolor="rgba(244,240,230,0.09)",
        zeroline=False,
        automargin=True,
        tickfont=dict(color="#928d82"),
    )
    return fig


FIRE_CMAP = colors.LinearSegmentedColormap.from_list(
    "fire_years", [(0.0, "#6b57a6"), (0.5, "#363b34"), (0.78, "#dc6027"), (1.0, "#ffb469")]
)


def stripe_color(ratio):
    t = 0.5 + np.log2(max(float(ratio), 0.05)) / 2.4
    return colors.to_hex(FIRE_CMAP(float(np.clip(t, 0.0, 1.0))))


def render_stripes(vs, title="Every year at a glance", active_year=None):
    if not vs:
        return
    cells = []
    for v in vs:
        bg = stripe_color(v["ratio"]) if v["ratio"] is not None else "#21241f"
        tip = escape(f"{v['year']}: {v['headline']}. {v['short']}", quote=True)
        cls = "stripe active" if (active_year is not None and v["year"] == active_year) else "stripe"
        cells.append(f'<div class="{cls}" title="{tip}" style="background:{bg}"></div>')

    swatches = "".join(
        f'<span style="background:{stripe_color(r)}"></span>'
        for r in (0.45, 0.72, 1.0, 1.4, 2.1)
    )
    mid_html = f"<span>{vs[len(vs) // 2]['year']}</span>" if len(vs) >= 8 else ""
    st.markdown(
        f'<div class="stripes-head">'
        f'<span class="stripes-title">{escape(title)}</span>'
        f'<span class="stripes-key"><span>Quieter</span>'
        f'<span class="stripes-swatches">{swatches}</span>'
        f'<span>Heavier</span></span></div>'
        f'<div class="stripes">{"".join(cells)}</div>'
        f'<div class="stripe-axis"><span>{vs[0]["year"]}</span>{mid_html}<span>{vs[-1]["year"]}</span></div>',
        unsafe_allow_html=True,
    )


@st.cache_data(max_entries=6)
def to_csv(key, _df):
    out = _df.drop(columns=["year", "month", "ym", "local_hour"], errors="ignore").copy()
    out["acq_date"] = out["acq_date"].dt.strftime("%Y-%m-%d")
    out = out.rename(columns={"weight": "viirs_equiv_weight"})
    return out.to_csv(index=False).encode()


ESRI = "https://server.arcgisonline.com/ArcGIS/rest/services"
TILES = {
    "Dark": dict(
        tiles=f"{ESRI}/Canvas/World_Dark_Gray_Base/MapServer/tile/{{z}}/{{y}}/{{x}}",
        attr="Tiles © Esri — Esri, HERE, Garmin, © OpenStreetMap contributors",
    ),
    "Light": dict(
        tiles=f"{ESRI}/Canvas/World_Light_Gray_Base/MapServer/tile/{{z}}/{{y}}/{{x}}",
        attr="Tiles © Esri — Esri, HERE, Garmin, © OpenStreetMap contributors",
    ),
    "Streets": dict(tiles="OpenStreetMap", attr="© OpenStreetMap contributors"),
    "Satellite": dict(
        tiles=f"{ESRI}/World_Imagery/MapServer/tile/{{z}}/{{y}}/{{x}}",
        attr="Tiles © Esri — Maxar, Earthstar Geographics",
    ),
    "Blank (offline)": None,
}

MAP_LEGEND_STYLE = (
    "position:fixed;bottom:24px;left:12px;z-index:9999;"
    "background:rgba(12,13,11,0.92);backdrop-filter:blur(6px);"
    "border:1px solid rgba(244,240,230,0.22);color:#f4f0e6;"
    "padding:7px 11px;border-radius:3px;"
    "font:11.5px/1.4 'JetBrains Mono',ui-monospace,monospace;"
    "box-shadow:0 8px 22px rgba(0,0,0,0.45);"
)


def base_map(style: str) -> folium.Map:
    m = folium.Map(
        location=[18, 12],
        zoom_start=2,
        control_scale=True,
        prefer_canvas=True,
        tiles=None,
        min_zoom=2,
        max_zoom=16,
        # Keep the viewport inside the world so panning can't drift into the
        # grey void left of/right of the dateline.
        max_bounds=[[-85.06, -180.0], [85.06, 180.0]],
        max_bounds_viscosity=0.5,
    )
    cfg = TILES[style]
    # Dark canvas behind tiles on every style: the iframe shows this while
    # Leaflet and tiles are still loading instead of a white flash.
    m.get_root().html.add_child(
        folium.Element("<style>.leaflet-container{background:#0c0d0b!important}</style>")
    )
    if cfg:
        folium.TileLayer(tiles=cfg["tiles"], attr=cfg["attr"], name=style, max_zoom=16,
                         no_wrap=True).add_to(m)
    return m


def _group_codes(series: pd.Series) -> np.ndarray:
    if isinstance(series.dtype, pd.CategoricalDtype):
        return series.cat.codes.to_numpy()
    return pd.factorize(series)[0]


def add_points(m, sub, color_by="sensor", max_points=3000):
    # Pick the sample with array positions BEFORE touching the frame: the old
    # version copied seven columns of every row just to show a few thousand
    # dots, which is what made the map slow on big countries.
    lh = sub["local_hour"].to_numpy()
    if color_by == "solar":
        ok = lh >= 0
        group = np.where((lh >= 6) & (lh < 18), 0, 1)
    else:
        ok = np.ones(len(sub), dtype=bool)
        group = _group_codes(sub["sensor"])

    idx_all = np.nonzero(ok)[0]
    n_all = int(len(idx_all))
    if n_all == 0:
        return 0, 0
    g_all = group[idx_all]
    groups = np.unique(g_all)
    per = max(1, max_points // len(groups))
    rng = np.random.default_rng(0)
    picks = []
    for g in groups:
        ids = idx_all[g_all == g]
        if len(ids) > per:
            ids = rng.choice(ids, per, replace=False)
        picks.append(ids)
    sel = np.sort(np.concatenate(picks))

    pts = sub.iloc[sel][["latitude", "longitude", "acq_date", "sensor", "platform", "weight", "local_hour"]].copy()
    pts["sensor"] = pts["sensor"].astype(str)
    pts["platform"] = pts["platform"].astype(str)
    dates = pts["acq_date"].dt.strftime("%Y-%m-%d").tolist()
    layer_grp = folium.FeatureGroup(name="Detections")

    if color_by == "solar":
        tod_colors = {"Daytime (06–18h)": DAY_COLOR, "Overnight (18–06h)": NIGHT_COLOR}
        tods = np.where((pts["local_hour"] >= 6) & (pts["local_hour"] < 18), "Daytime (06–18h)", "Overnight (18–06h)")
        for la, lo, d, sn, lh_, td in zip(
            pts["latitude"], pts["longitude"], dates, pts["sensor"], pts["local_hour"], tods
        ):
            col = tod_colors[td]
            folium.CircleMarker(
                location=[float(la), float(lo)],
                radius=3,
                color=col,
                weight=0,
                fill=True,
                fill_color=col,
                fill_opacity=0.68,
                tooltip=f"{d} · ~{int(lh_):02d}:00 solar ({sn})",
            ).add_to(layer_grp)
        layer_grp.add_to(m)
        items = "&nbsp;&nbsp;".join(
            f'<span style="display:inline-flex;align-items:center;gap:5px;">'
            f'<span style="width:8px;height:8px;border-radius:50%;background:{c};display:inline-block;"></span>'
            f"{lbl}</span>"
            for lbl, c in tod_colors.items()
        )
        m.get_root().html.add_child(folium.Element(f'<div style="{MAP_LEGEND_STYLE}">{items}</div>'))
    else:
        for la, lo, d, sn, pl, wt in zip(
            pts["latitude"], pts["longitude"], dates, pts["sensor"], pts["platform"], pts["weight"]
        ):
            col = SENSOR_COLORS.get(sn, OTHER_COLOR)
            plat_txt = f" ({pl})" if pl and pl != "UNKNOWN" else ""
            folium.CircleMarker(
                location=[float(la), float(lo)],
                radius=3,
                color=col,
                weight=0,
                fill=True,
                fill_color=col,
                fill_opacity=0.65,
                tooltip=f"{d} · {sn}{plat_txt} · {wt:.2f} eq.",
            ).add_to(layer_grp)
        layer_grp.add_to(m)
        present = [sn for sn in ("MODIS", "VIIRS") if sn in set(pts["sensor"])] or sorted(set(pts["sensor"]))
        items = "&nbsp;&nbsp;".join(
            f'<span style="display:inline-flex;align-items:center;gap:5px;">'
            f'<span style="width:8px;height:8px;border-radius:50%;background:{SENSOR_COLORS.get(sn, OTHER_COLOR)};display:inline-block;"></span>'
            f"{sn}</span>"
            for sn in present
        )
        m.get_root().html.add_child(folium.Element(f'<div style="{MAP_LEGEND_STYLE}">{items}</div>'))

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
        "type": "Feature",
        "properties": props,
        "geometry": {
            "type": "Polygon",
            "coordinates": [[[lo - h, la - h], [lo + h, la - h], [lo + h, la + h], [lo - h, la + h], [lo - h, la - h]]],
        },
    }


def _frame_sig(df):
    return (len(df), round(float(df["weight"].sum()), 3),
            str(df["acq_date"].min()), str(df["acq_date"].max()))


def add_density(m, sub):
    # Memoize the O(n) grid aggregation — it is identical across reruns that
    # don't change the underlying detections (e.g. map layer switches).
    sig = ("dens",) + _frame_sig(sub)
    cache = st.session_state.setdefault("_dens_cache", {})
    if sig not in cache:
        cell = 0.05
        grid = _grid(sub, cell)
        while len(grid) > 2500 and cell < 5:
            cell *= 2
            grid = _grid(sub, cell)
        if len(cache) > 8:
            cache.clear()
        cache[sig] = (cell, grid)
    cell, grid = cache[sig]
    vmin = max(float(grid["w"].min()), 0.1)
    vmax = max(float(grid["w"].max()), vmin * 1.5)
    norm = colors.LogNorm(vmin=vmin, vmax=vmax, clip=True)
    cmap = colormaps["YlOrRd"]
    feats = [
        _cell_feature(
            la, lo, cell / 2,
            {"count": round(float(w)), "color": colors.to_hex(cmap(0.2 + 0.8 * norm(w)))},
        )
        for la, lo, w in zip(grid["glat"], grid["glon"], grid["w"])
    ]
    folium.GeoJson(
        {"type": "FeatureCollection", "features": feats},
        style_function=lambda f: {"fillColor": f["properties"]["color"], "fillOpacity": 0.70, "weight": 0},
        tooltip=folium.GeoJsonTooltip(fields=["count"], aliases=["Detections (VIIRS-eq.)"]),
    ).add_to(m)

    max_val = int(round(grid["w"].max()))
    ramp_cols = [colors.to_hex(cmap(t)) for t in np.linspace(0.2, 1.0, 6)]
    grad = f"linear-gradient(90deg, {', '.join(ramp_cols)})"
    m.get_root().html.add_child(
        folium.Element(
            f'<div style="{MAP_LEGEND_STYLE}">'
            f'<div style="color:#928d82;font-size:10px;margin-bottom:4px;">VIIRS-EQ. DENSITY ({cell:g}° CELL)</div>'
            f'<div style="display:flex;align-items:center;gap:8px;">'
            f"<span>Low</span>"
            f'<span style="width:92px;height:7px;border-radius:2px;background:{grad};display:inline-block;"></span>'
            f"<span>{max_val:,}</span>"
            f"</div></div>"
        )
    )
    return cell, max_val


def add_change(m, sub, recent, base):
    sig = ("chg",) + _frame_sig(sub) + (tuple(recent), tuple(base))
    cache = st.session_state.setdefault("_chg_cache", {})
    if sig not in cache:
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
        mg = mg[mg["d"].abs() >= 0.05 * lim]
        if len(cache) > 8:
            cache.clear()
        cache[sig] = (cell, lim, mg)
    cell, lim, mg = cache[sig]
    norm = colors.Normalize(vmin=-lim, vmax=lim, clip=True)
    # Purple = fewer fires than baseline, orange = more
    cmap = colormaps["PuOr_r"]
    feats = [
        _cell_feature(
            la, lo, cell / 2,
            {
                "change": round(float(d), 1),
                "recent": round(float(r), 1),
                "baseline": round(float(b), 1),
                "color": colors.to_hex(cmap(norm(d))),
            },
        )
        for la, lo, d, r, b in zip(mg["glat"], mg["glon"], mg["d"], mg["w_r"], mg["w_b"])
    ]
    folium.GeoJson(
        {"type": "FeatureCollection", "features": feats},
        style_function=lambda f: {"fillColor": f["properties"]["color"], "fillOpacity": 0.72, "weight": 0},
        tooltip=folium.GeoJsonTooltip(
            fields=["change", "recent", "baseline"],
            aliases=["Change / yr", "Recent / yr", "Baseline / yr"],
        ),
    ).add_to(m)

    ramp_cols = [colors.to_hex(cmap(t)) for t in np.linspace(0.05, 0.95, 7)]
    grad = f"linear-gradient(90deg, {', '.join(ramp_cols)})"
    m.get_root().html.add_child(
        folium.Element(
            f'<div style="{MAP_LEGEND_STYLE}">'
            f'<div style="color:#928d82;font-size:10px;margin-bottom:4px;">ANNUAL CHANGE VS BASELINE</div>'
            f'<div style="display:flex;align-items:center;gap:8px;">'
            f"<span>−{lim:,.0f}</span>"
            f'<span style="width:92px;height:7px;border-radius:2px;background:{grad};display:inline-block;"></span>'
            f"<span>+{lim:,.0f}/yr</span>"
            f"</div></div>"
        )
    )
    return cell, lim


ATTRIBUTION_FIX = (
    "<style>.leaflet-control-attribution{background:rgba(12,13,11,.85)!important;"
    "color:#928d82!important;font-size:10px;border-top-left-radius:3px}"
    ".leaflet-control-attribution a{color:#8fb369!important}</style>"
)


# ---- Source classification and the heavier per-section tables ----------------
SRC_CLASSES = [
    "Vegetation / agricultural (unresolved)",
    "Industrial / static source",
    "Non-vegetation (FIRMS type)",
]
SRC_COLORS = {
    "Vegetation / agricultural (unresolved)": "#f06428",
    "Industrial / static source": "#7a8ba8",
    "Non-vegetation (FIRMS type)": "#5b564f",
}


def source_class_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate detections into fire-source classes at ~1 km cell level.

    Deliberately conservative: only persistent year-round recurring cells
    (gas flares, industrial stacks) are split out as 'Industrial / static'.
    Everything else stays 'Vegetation / agricultural (unresolved)' because
    FIRMS alone cannot separate field burning from wildfire without a
    land-cover join.
    """
    ft_col = df["firms_type"].to_numpy() if "firms_type" in df else np.full(len(df), np.nan)
    # Integer cell id: far cheaper than building two strings per row
    cell_id = (
        np.round(df["latitude"].to_numpy(float) * 100).astype("int64") * 100000
        + np.round(df["longitude"].to_numpy(float) * 100).astype("int64")
    )
    work = pd.DataFrame({
        "cell": cell_id,
        "date": df["acq_date"].to_numpy(),
        "month": df["month"].to_numpy(),
        "year": df["year"].to_numpy(),
        "w": df["weight"].to_numpy(),
        "ft": ft_col,
    })
    cell = work.groupby("cell", sort=False).agg(
        days=("date", "nunique"),
        months=("month", "nunique"),
        years=("year", "nunique"),
        n=("w", "size"),
        w=("w", "sum"),
        ft=("ft", "first"),
    )

    span_days = max((df["acq_date"].max() - df["acq_date"].min()).days, 1)

    cls = pd.Series("Vegetation / agricultural (unresolved)", index=cell.index, dtype=object)
    ft = cell["ft"]
    for code, name in ((1.0, "Volcano (FIRMS type 1)"), (2.0, "Industrial / static (FIRMS type 2)"),
                       (3.0, "Offshore (FIRMS type 3)")):
        cls[ft == code] = name
    # Fires on most days across many months and years => fixed industrial/flaring source
    persistent = ((cell["days"] >= 15) & (cell["months"] >= 8) & (cell["years"] >= 3)
                  & (cell["days"] >= 0.15 * span_days))
    cls[persistent & (cls == "Vegetation / agricultural (unresolved)")] = "Industrial / static source"
    cell["cls"] = cls

    return (cell.groupby("cls")
            .agg(detections=("n", "sum"), viirs_eq=("w", "sum"))
            .reset_index())


@st.cache_data(show_spinner="Classifying fire sources…", max_entries=6)
def source_summary(skey):
    return source_class_summary(get_sub(*skey))


@st.cache_data(show_spinner=False, max_entries=6)
def hour_profile(skey):
    """Weighted detections by local solar hour and sensor (None without acq_time)."""
    sub = get_sub(*skey)
    lh = sub["local_hour"].to_numpy()
    ok = lh >= 0
    if not ok.any():
        return None
    cats = list(sub["sensor"].cat.categories) if isinstance(sub["sensor"].dtype, pd.CategoricalDtype) else None
    codes = _group_codes(sub["sensor"])
    frame = pd.DataFrame({"hour": lh[ok], "code": codes[ok], "w": sub["weight"].to_numpy()[ok]})
    piv = frame.groupby(["hour", "code"])["w"].sum().unstack(fill_value=0.0)
    if cats is not None:
        piv.columns = [str(cats[c]) if 0 <= c < len(cats) else "UNKNOWN" for c in piv.columns]
    return piv.reindex(index=range(24), fill_value=0.0)


@st.cache_data(show_spinner=False, max_entries=6)
def cumulative_daily(skey, years):
    """Cumulative weighted detections by day of year, one row per year."""
    sub = get_sub(*skey)
    doy = np.minimum(sub["acq_date"].dt.dayofyear.to_numpy(), 365)
    d = pd.DataFrame({"year": sub["year"].to_numpy(), "doy": doy, "w": sub["weight"].to_numpy()})
    return (
        d.groupby(["year", "doy"])["w"].sum().unstack("doy")
        .reindex(index=list(years), columns=range(1, 366), fill_value=0.0)
        .cumsum(axis=1)
    )


@st.cache_data(show_spinner="Building the timelapse…", max_entries=4)
def timelapse_grid(skey):
    sub = get_sub(*skey)

    def yearly_grid(c):
        g = pd.DataFrame({
            "year": sub["year"].to_numpy(),
            "lat": np.floor(sub["latitude"].to_numpy() / c) * c + c / 2,
            "lon": np.floor(sub["longitude"].to_numpy() / c) * c + c / 2,
            "w": sub["weight"].to_numpy(),
        })
        return g.groupby(["year", "lat", "lon"], as_index=False)["w"].sum()

    cell_t = 0.1
    tl = yearly_grid(cell_t)
    while len(tl) > 60000 and cell_t < 5:
        cell_t *= 2
        tl = yearly_grid(cell_t)
    return tl.sort_values("year"), cell_t


# ---- One cached pass over the selected area and period -----------------------
@st.cache_data(show_spinner="Analysing the record…", max_entries=6)
def analyze(skey, region_text):
    """Every summary number the page needs for one (file, filters, period, area)
    combination. Cached, so clicking around the interface no longer re-runs
    group-bys over millions of rows."""
    (path_str, mtime, drop_low, veg_only, viirs_mode, seasonal, min_monthly, d0, d1, aoi_wkt, calib_end_year) = skey
    sub = get_sub(*skey)
    _, info = build_dataset(path_str, mtime, drop_low, veg_only, viirs_mode, seasonal, min_monthly, calib_end_year)

    lo, hi = pd.Timestamp(d0), pd.Timestamp(d1)
    first_y, last_y = int(sub["year"].min()), int(sub["year"].max())
    years = list(range(first_y, last_y + 1))

    monthly = sub.groupby(["year", "month"])["weight"].sum()
    pivot = monthly.unstack("month").reindex(index=years, columns=range(1, 13))
    covered, full = _month_masks(pivot, lo, hi)
    pivot_cov = pivot.fillna(0).where(covered)
    pivot_full = pivot.fillna(0).where(full)
    med_by_month = pivot_full.median()

    annual_by_sensor = (
        sub.groupby(["year", "sensor"], observed=True)["weight"]
        .sum().unstack(fill_value=0).reindex(years, fill_value=0)
    )
    annual_by_sensor = annual_by_sensor.loc[:, (annual_by_sensor != 0).any()]  # drop empty sensors
    annual_total = annual_by_sensor.sum(axis=1)
    annual_raw = sub.groupby("year").size().reindex(years, fill_value=0)
    full_years = [y for y in years if full.loc[y].all()]

    latest = sub["acq_date"].max()
    ly, lm, last_day = int(latest.year), int(latest.month), int(latest.day)
    current = float(sub.loc[(sub["year"] == ly) & (sub["month"] == lm), "weight"].sum())
    past_years = [y for y in years if y < ly]
    hist = (
        sub[(sub["month"] == lm) & (sub["year"] < ly) & (sub["acq_date"].dt.day <= last_day)]
        .groupby("year")["weight"]
        .sum()
        .reindex(past_years, fill_value=0)
    )
    status = status_for(current, hist) if len(hist) > 0 else None

    arch = archive_year_stats(path_str, mtime, aoi_wkt, drop_low, veg_only, viirs_mode, seasonal, min_monthly, calib_end_year)
    if arch is None:
        verdicts, year_baseline, arch_cov, arch_full = [], None, None, None
    else:
        verdicts, year_baseline, arch_cov, arch_full = arch
    alert_days = archive_alert_days(path_str, mtime, aoi_wkt, drop_low, veg_only, viirs_mode, seasonal, min_monthly, calib_end_year)

    has_frp = bool(info["flags"].get("has_frp"))
    field_notes = derive_field_notes(sub, med_by_month, info, verdicts, region_text, veg_only)

    # Honesty check: a trend fitted only on VIIRS-era years does not depend on
    # the MODIS->VIIRS scaling factor. The split year itself still contains
    # scaled MODIS months, so start strictly after it.
    first_clean = (
        info["split"].year + (1 if info["split"].month > 1 else 0)
        if info["split"] is not None else None
    )
    viirs_years = [y for y in full_years if first_clean is not None and y >= first_clean]
    trend_viirs = rel_trend(annual_total, viirs_years)
    dmax_dt = sub["acq_date"].max()
    nrt_recent = bool((pd.Timestamp.today().normalize() - dmax_dt) < pd.Timedelta(120, unit="D"))

    monthly_agg = (
        sub.groupby(["year", "month"])
        .agg(detections_viirs_equiv=("weight", "sum"), raw_detections=("weight", "size"),
             frp_mw=("frp_w", "sum"))
        .reset_index()
    )

    return dict(
        years=years, first_y=first_y, last_y=last_y,
        pivot_cov=pivot_cov, pivot_full=pivot_full, med_by_month=med_by_month, full=full,
        annual_by_sensor=annual_by_sensor, annual_total=annual_total, annual_raw=annual_raw,
        full_years=full_years,
        trend_h=rel_trend(annual_total, full_years), trend_raw=rel_trend(annual_raw, full_years),
        trend_viirs=trend_viirs, viirs_years=viirs_years, nrt_recent=nrt_recent,
        latest=latest, ly=ly, lm=lm, last_day=last_day, current=current, hist=hist, status=status,
        verdicts=verdicts, year_baseline=year_baseline, arch_cov=arch_cov, arch_full=arch_full,
        alert_days=alert_days, has_frp=has_frp, field_notes=field_notes, monthly_agg=monthly_agg,
        n_rows=int(len(sub)), w_sum=float(sub["weight"].sum()),
        frp_sum=float(sub["frp_w"].sum()), frp_max=float(sub["frp_w"].max()),
        sensors_present=[str(s) for s in sub["sensor"].unique()],
        years_present=sorted(int(y) for y in sub["year"].unique()),
        lat_c=float(sub["latitude"].mean()), lon_c=float(sub["longitude"].mean()),
    )


PAGE_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,500;9..144,600&family=JetBrains+Mono:wght@400;500;600&family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap');

:root {
  --bg: #0c0d0b;
  --sidebar: #111310;
  --surface: #151714;
  --surface-2: #1c1f1a;
  --ink: #f4f0e6;
  --ink-soft: #cbc5b9;
  --muted: #928d82;
  --line: rgba(244, 240, 230, 0.11);
  --line-strong: rgba(244, 240, 230, 0.26);
  --sage: #8fb369;
  --ember: #f06428;
  --amber: #e0a938;
  --slate: #6892b5;
  --violet: #8976bd;
}

html, body, .stApp {
  background-color: var(--bg);
  color: var(--ink);
}

.stApp, .stApp p, .stApp label, .stApp li, .stApp h2, .stApp h3, .stApp h4,
.stApp button, .stApp input, .stApp textarea, .stApp [data-baseweb="tab"] {
  font-family: __UI__;
}

[data-testid="stHeader"] {
  background: transparent;
}

.block-container {
  padding-top: 1.85rem;
  padding-bottom: 3rem;
  max-width: 1500px;
}

footer, #MainMenu, [data-testid="stAppDeployButton"] {
  visibility: hidden;
}

/* Cartographic masthead with subtle orbital graticule lines */
.masthead {
  position: relative;
  display: flex;
  justify-content: space-between;
  align-items: flex-end;
  flex-wrap: wrap;
  gap: 1rem 2rem;
  border-bottom: 1px solid var(--line-strong);
  padding: .35rem 0 1.15rem 0;
  margin-bottom: 1.25rem;
  background-image:
    linear-gradient(to right, rgba(244, 240, 230, 0.03) 1px, transparent 1px),
    linear-gradient(to bottom, rgba(244, 240, 230, 0.03) 1px, transparent 1px);
  background-size: 36px 36px;
}
.masthead h1 {
  font-family: __DISPLAY__;
  font-size: 2.38rem;
  font-weight: 600;
  letter-spacing: -0.025em;
  line-height: 1.05;
  color: var(--ink);
  margin: 0;
  padding: 0;
}
.masthead .sub {
  color: var(--ink-soft);
  font-size: .95rem;
  line-height: 1.52;
  max-width: 62ch;
  margin: .42rem 0 0 0;
}
.telemetry {
  display: flex;
  flex-wrap: wrap;
  gap: .55rem 1.25rem;
  font-family: __MONO__;
  font-size: .74rem;
  color: var(--muted);
  padding: .5rem .8rem;
  background: var(--surface);
  border: 1px solid var(--line);
  border-left: 2px solid var(--ember);
  border-radius: 3px;
}
.telemetry b {
  color: var(--ink);
  font-weight: 500;
}
.telemetry .stale b {
  color: var(--amber);
}

/* Sidebar */
[data-testid="stSidebar"] {
  background-color: var(--sidebar);
  border-right: 1px solid var(--line);
}
.sb-label {
  font-family: __MONO__;
  font-size: .70rem;
  font-weight: 600;
  letter-spacing: .07em;
  text-transform: uppercase;
  color: var(--muted);
  border-top: 1px solid var(--line);
  padding-top: .8rem;
  margin: 1rem 0 .4rem 0;
}
.sb-label.top {
  border-top: none;
  padding-top: 0;
  margin-top: .1rem;
}
.aoi-box {
  font-family: __MONO__;
  font-size: .73rem;
  line-height: 1.5;
  color: var(--ink-soft);
  border-left: 2px solid #4ecdc4;
  padding: .2rem 0 .2rem .65rem;
  margin-top: .5rem;
}

/* Metrics */
[data-testid="stMetric"] {
  background: transparent;
  border: none;
  border-top: 1px solid var(--line);
  border-radius: 0;
  padding: .52rem 0 .32rem 0;
  margin-bottom: .25rem;
}
[data-testid="stMetricLabel"] p {
  font-size: .78rem;
  color: var(--muted);
  white-space: normal;
  line-height: 1.3;
}
[data-testid="stMetricDelta"] {
  font-family: __MONO__;
  font-size: .74rem;
}
[data-testid="stMetricValue"] {
  font-family: __MONO__;
  font-size: 1.32rem;
  font-weight: 600;
  letter-spacing: -0.02em;
  color: var(--ink);
  font-variant-numeric: tabular-nums;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
  gap: 1.65rem;
  border-bottom: 1px solid var(--line);
  margin-bottom: .75rem;
}
.stTabs [data-baseweb="tab"] {
  padding: .5rem 0;
  background: transparent;
  color: var(--muted);
  font-size: .91rem;
  font-weight: 500;
}
.stTabs [aria-selected="true"] {
  color: var(--ink) !important;
  font-weight: 600;
}
.stTabs [data-baseweb="tab-highlight"] {
  background-color: var(--ember);
  height: 2px;
}

/* Verdict statement */
.verdict {
  border-left: 3px solid var(--slate);
  padding: .15rem 0 .2rem .95rem;
  margin: .35rem 0 1rem 0;
  animation: fadeSlide .35s ease-out;
}
@keyframes fadeSlide {
  from {opacity: 0; transform: translateY(5px);}
  to {opacity: 1; transform: none;}
}
@media (prefers-reduced-motion: reduce) {
  .verdict {animation: none;}
}
.verdict-meta {
  font-family: __MONO__;
  font-size: .72rem;
  color: var(--muted);
  margin-bottom: .25rem;
}
.verdict-head {
  font-family: __DISPLAY__;
  font-size: 1.58rem;
  font-weight: 600;
  letter-spacing: -0.015em;
  line-height: 1.18;
  color: var(--ink);
}
.verdict-body {
  color: var(--ink-soft);
  font-size: .91rem;
  line-height: 1.56;
  margin-top: .38rem;
}
.verdict.low, .verdict.normal, .verdict.typical {border-left-color: #6892b5;}
.verdict.low .verdict-meta, .verdict.normal .verdict-meta, .verdict.typical .verdict-meta {color: #8cb4d4;}
.verdict.elevated, .verdict.more {border-left-color: #e0a938;}
.verdict.elevated .verdict-meta, .verdict.more .verdict-meta {color: #ecc063;}
.verdict.high, .verdict.much_more {border-left-color: #f06428;}
.verdict.high .verdict-meta, .verdict.much_more .verdict-meta {color: #f48656;}
.verdict.less {border-left-color: #8976bd;}
.verdict.less .verdict-meta {color: #a695d4;}
.verdict.much_less {border-left-color: #6b57a6;}
.verdict.much_less .verdict-meta {color: #9683cc;}
.verdict.partial {border-left-color: #585c55;}

/* Field Notes cards */
.notes-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
  gap: 10px;
  margin: .45rem 0 1.15rem 0;
}
.fnote {
  background: var(--surface);
  border: 1px solid var(--line);
  border-left: 3px solid var(--sage);
  border-radius: 3px;
  padding: .68rem .85rem .72rem;
  transition: background-color .14s;
}
.fnote:hover {
  background: var(--surface-2);
}
.fnote.regime {border-left-color: var(--ember);}
.fnote.clock {border-left-color: var(--amber);}
.fnote.record {border-left-color: var(--slate);}
.fnote.orbital {border-left-color: var(--sage);}
.fnote-tag {
  font-family: __MONO__;
  font-size: .68rem;
  font-weight: 600;
  letter-spacing: .05em;
  text-transform: uppercase;
  color: var(--muted);
  margin-bottom: .22rem;
}
.fnote.regime .fnote-tag {color: #f48656;}
.fnote.clock .fnote-tag {color: #ecc063;}
.fnote.record .fnote-tag {color: #8cb4d4;}
.fnote.orbital .fnote-tag {color: #a8c786;}
.fnote-head {
  font-size: .92rem;
  font-weight: 600;
  color: var(--ink);
  line-height: 1.32;
}
.fnote-body {
  font-size: .83rem;
  line-height: 1.52;
  color: var(--ink-soft);
  margin-top: .28rem;
}

.lede {
  font-size: 1.03rem;
  line-height: 1.65;
  color: var(--ink-soft);
  max-width: 78ch;
  margin: .2rem 0 1.1rem 0;
}

/* Annual fire stripes */
.stripes-head {
  display: flex;
  justify-content: space-between;
  align-items: center;
  flex-wrap: wrap;
  gap: .5rem;
  margin: .1rem 0 .35rem 0;
}
.stripes-title {
  font-size: .87rem;
  font-weight: 600;
  color: var(--ink);
}
.stripes-key {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  font-family: __MONO__;
  font-size: .70rem;
  color: var(--muted);
}
.stripes-swatches {
  display: inline-flex;
  gap: 2px;
}
.stripes-swatches span {
  width: 12px;
  height: 8px;
  border-radius: 1px;
  display: inline-block;
}
.stripes {
  display: flex;
  height: 3.9rem;
  gap: 2px;
  padding: 2px;
  background: var(--surface);
  border: 1px solid var(--line);
  border-radius: 3px;
  margin: 0 0 .25rem 0;
}
.stripe {
  flex: 1;
  border-radius: 1px;
  transition: filter .14s, transform .14s;
}
.stripe:hover {
  filter: brightness(1.28);
  transform: scaleY(1.05);
}
.stripe.active {
  outline: 2px solid var(--ink);
  outline-offset: 1px;
  z-index: 2;
}
.stripe-axis {
  display: flex;
  justify-content: space-between;
  font-family: __MONO__;
  color: var(--muted);
  font-size: .73rem;
  margin-bottom: .35rem;
}

/* Year cards */
.yeargrid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(170px, 1fr));
  gap: 8px;
  margin: 1.1rem 0 .5rem 0;
}
.yearcard {
  display: flex;
  flex-direction: column;
  justify-content: space-between;
  border: 1px solid var(--line);
  border-top: 3px solid var(--line);
  background: var(--surface);
  padding: .62rem .75rem .68rem;
  border-radius: 3px;
  transition: background-color .14s;
}
.yearcard:hover {
  background: var(--surface-2);
}
.yearcard.active {
  background: var(--surface-2);
  border-color: var(--line-strong);
  box-shadow: inset 0 0 0 1px rgba(244, 240, 230, 0.24);
}
.yearcard .y {
  font-family: __MONO__;
  font-size: 1.06rem;
  font-weight: 600;
  color: var(--ink);
  font-variant-numeric: tabular-nums;
}
.yearcard .h {
  font-size: .82rem;
  line-height: 1.35;
  color: var(--ink-soft);
  margin-top: .2rem;
}
.yearcard .s {
  font-family: __MONO__;
  font-size: .72rem;
  color: var(--muted);
  margin-top: .42rem;
  padding-top: .32rem;
  border-top: 1px solid var(--line);
}
.yearcard.much_more {border-top-color: #f06428;}
.yearcard.more {border-top-color: #e0a938;}
.yearcard.typical {border-top-color: #6892b5;}
.yearcard.less {border-top-color: #8976bd;}
.yearcard.much_less {border-top-color: #6b57a6;}
.yearcard.partial {border-top-color: #585c55; opacity: .74;}

/* Month strip */
.ribbon-title {
  font-size: .87rem;
  font-weight: 600;
  color: var(--ink);
  margin: .95rem 0 .4rem 0;
}
.ribbon {
  display: grid;
  grid-template-columns: repeat(6, 1fr);
  gap: 4px;
}
.ribbon-cell {
  height: 2.3rem;
  border-radius: 2px;
  padding: .22rem .4rem;
  display: flex;
  align-items: flex-end;
  font-family: __MONO__;
  font-size: .71rem;
  font-weight: 500;
  border: 1px solid rgba(255, 255, 255, 0.05);
}

/* Harmonization method notes */
.method-list {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(230px, 1fr));
  gap: 14px;
  border-top: 1px solid var(--line);
  border-bottom: 1px solid var(--line);
  padding: .9rem 0;
  margin: .2rem 0 1rem 0;
}
.method-item b {
  font-size: .92rem;
  color: var(--ink);
}
.method-item p {
  color: var(--ink-soft);
  font-size: .86rem;
  line-height: 1.52;
  margin: .25rem 0 0 0;
}

.region-head {
  display: flex;
  align-items: baseline;
  justify-content: space-between;
  flex-wrap: wrap;
  gap: .4rem;
  padding-bottom: .4rem;
  margin-bottom: .3rem;
  border-bottom: 1px solid var(--line);
}
.region-head .title {
  font-family: __DISPLAY__;
  font-size: 1.28rem;
  font-weight: 600;
  color: var(--ink);
}
.region-head .span {
  font-family: __MONO__;
  font-size: .74rem;
  color: var(--muted);
}

.landing-caption {
  display: flex;
  justify-content: space-between;
  align-items: baseline;
  flex-wrap: wrap;
  gap: .5rem 1.5rem;
  margin: .75rem 0 .5rem 0;
  color: var(--ink-soft);
  font-size: .93rem;
}

.note {
  font-size: .84rem;
  color: var(--muted);
  line-height: 1.55;
}
.credits {
  margin-top: 2.3rem;
  padding-top: .8rem;
  border-top: 1px solid var(--line);
  color: var(--muted);
  font-size: .78rem;
  line-height: 1.6;
}

[data-testid="stCustomComponentV1"] > div {
  position: relative;
  min-height: 560px;
}
[data-testid="stCustomComponentV1"] > div::after {
  content: "Loading map…";
  position: absolute;
  inset: 0;
  z-index: 0;
  display: flex;
  align-items: center;
  justify-content: center;
  background: #0c0d0b;
  color: #928d82;
  font-size: .82rem;
  letter-spacing: .12em;
  text-transform: uppercase;
}
[data-testid="stCustomComponentV1"] iframe {
  position: relative;
  z-index: 1;
  border: 1px solid var(--line);
  border-radius: 3px;
  background: #0c0d0b;
}
.stDownloadButton button, .stButton button {
  border-radius: 3px;
  border: 1px solid var(--line-strong);
  background-color: var(--surface);
  color: var(--ink);
  font-size: .85rem;
}
.stDownloadButton button:hover, .stButton button:hover {
  border-color: var(--ember);
  color: var(--ember);
}
</style>
"""

st.markdown(
    PAGE_CSS.replace("__UI__", FONT_UI)
    .replace("__DISPLAY__", FONT_DISPLAY)
    .replace("__MONO__", FONT_MONO),
    unsafe_allow_html=True,
)

try:
    regions, canon, geoms_wkb = load_regions()
except Exception as e:
    st.error(f"Could not load country boundaries from {NE_PATH}: {e}")
    st.stop()

stems = data_stems()


def has_data(key: str) -> bool:
    return slug(key, canon) in stems or key.replace(" ", "_") in stems


def region_label(key: str) -> str:
    return key.title() + ("  · loaded" if has_data(key) else "")


# Sidebar controls
_pending_region = st.session_state.pop("_pending_region", None)
if _pending_region is not None:
    st.session_state["region_pick"] = _pending_region
if st.session_state.pop("_clear_region", False):
    st.session_state["region_pick"] = None
st.session_state.setdefault("region_pick", None)

sb = st.sidebar
sb.markdown('<div class="sb-label top">Region</div>', unsafe_allow_html=True)
match = sb.selectbox(
    "Region",
    sorted(regions),
    index=None,
    format_func=region_label,
    placeholder="Type any country or territory…",
    label_visibility="collapsed",
    key="region_pick",
)
if match is not None:
    # _land_obj is a per-map click signature that must persist so stale
    # component values can't replay when returning to a previous map.
    st.session_state["_had_region"] = True
    if sb.button(
        "← All countries",
        key="back_to_world",
        help="Return to the world map to open a different country.",
    ):
        st.session_state["_clear_region"] = True
        st.session_state.pop("_pending_region", None)
        st.rerun()
elif st.session_state.pop("_had_region", False):
    # Returning to the world map: bump the epoch so the Folium component
    # remounts with a clean slate — st_folium's click values are sticky.
    st.session_state["_map_epoch"] = st.session_state.get("_map_epoch", 0) + 1
bbox = regions[match] if match else None
match_title = match.title() if match else ""

sb.markdown('<div class="sb-label">Area of interest</div>', unsafe_allow_html=True)
aoi_mode = sb.radio(
    "Area of interest",
    ["Whole region", "Draw on map", "Coordinates"],
    label_visibility="collapsed",
)
aoi_poly = None
if match and aoi_mode == "Draw on map":
    saved = st.session_state.get("aoi_draw")
    if saved and saved["region"] == match:
        aoi_poly = shapely_wkt.loads(saved["wkt"])
    sb.caption("Use the polygon, rectangle, or circle tool at the top left of the map.")
    sb.button("Clear drawn area", on_click=lambda: st.session_state.pop("aoi_draw", None))
elif match and aoi_mode == "Coordinates":
    kind = sb.radio("Shape", ["Bounding box", "Point and radius"], horizontal=True, label_visibility="collapsed")
    if kind == "Bounding box":
        c1, c2 = sb.columns(2)
        lat0 = c1.number_input("South latitude", -90.0, 90.0, round(bbox[0], 3), 0.1, format="%.3f", key=f"s_{match}")
        lat1 = c2.number_input("North latitude", -90.0, 90.0, round(bbox[1], 3), 0.1, format="%.3f", key=f"n_{match}")
        lon0 = c1.number_input("West longitude", -180.0, 180.0, round(bbox[2], 3), 0.1, format="%.3f", key=f"w_{match}")
        lon1 = c2.number_input("East longitude", -180.0, 180.0, round(bbox[3], 3), 0.1, format="%.3f", key=f"e_{match}")
        if lat1 > lat0 and lon1 > lon0:
            aoi_poly = box(lon0, lat0, lon1, lat1)
        else:
            sb.warning("North must be above south, and east right of west.")
    else:
        c1, c2 = sb.columns(2)
        clat = c1.number_input("Latitude", -90.0, 90.0, round((bbox[0] + bbox[1]) / 2, 3), 0.1, format="%.3f", key=f"clat_{match}")
        clon = c2.number_input("Longitude", -180.0, 180.0, round((bbox[2] + bbox[3]) / 2, 3), 0.1, format="%.3f", key=f"clon_{match}")
        rad = sb.number_input("Radius (km)", 1.0, 2000.0, 50.0, 5.0, key=f"rad_{match}")
        aoi_poly = circle_polygon(clon, clat, rad * 1000.0)

sb.markdown('<div class="sb-label">Period</div>', unsafe_allow_html=True)
date_box = sb.container()
sb.markdown('<div class="sb-label">Map display</div>', unsafe_allow_html=True)
layer = sb.selectbox(
    "Map layer",
    ["Points", "Density grid", "Change vs baseline", "Day vs night (solar)"],
    label_visibility="collapsed",
)
style = sb.selectbox("Base map", list(TILES), label_visibility="collapsed")
sb.markdown('<div class="sb-label">Data cleaning</div>', unsafe_allow_html=True)
with sb.expander("Quality and harmonization"):
    drop_low = st.checkbox(
        "Drop low-confidence detections",
        True,
        help="Removes MODIS confidence below 30 and VIIRS 'low'.",
    )
    veg_only = st.checkbox(
        "Vegetation fires only",
        True,
        help="Keeps FIRMS type 0 and drops volcanoes, gas flares, and static industrial heat sources.",
    )
    viirs_mode = st.radio(
        "VIIRS satellites",
        ["SNPP only", "All, normalized per satellite"],
        help="SNPP has flown since 2012. Adding NOAA-20 and NOAA-21 increases raw overpasses without more fire.",
    )
    scaling = st.radio(
        "MODIS to VIIRS scaling",
        ["Single factor", "By calendar month"],
        help="Monthly factors track seasonal differences in small agricultural fires missed by MODIS.",
    )
    min_monthly = st.number_input(
        "Min. detections per overlap month",
        1,
        500,
        10,
        help="Months below this threshold are skipped when estimating the scaling ratio.",
    )
    calib_year = st.number_input(
        "Calibrate MODIS→VIIRS through year",
        2012,
        pd.Timestamp.today().year,
        2019,
        help="Overlap months after this year are excluded from the scaling fit. "
        "Terra/Aqua orbit drift in recent years can bias the ratio.",
    )

# ---- Load the record --------------------------------------------------------
viirs_key = "snpp" if viirs_mode.startswith("SNPP") else "all"
seasonal_on = scaling.startswith("By")
aoi_wkt = aoi_poly.wkt if aoi_poly is not None else None
region_text = match_title + (" (selected area)" if aoi_poly is not None else "")

sub, info, S, skey, path = None, None, None, None, None
data_ready = False
dmax = None

if match:
    path = find_data_file(match, canon)
    if path is None:
        st.info(
            f"No hotspot file found for {match_title} yet. Drop a FIRMS archive file at "
            f"`data/{slug(match, canon)}.csv` (or `.parquet`) to open its record."
        )
    else:
        mtime = path.stat().st_mtime
        try:
            hot, info = build_dataset(
                str(path), mtime, drop_low, veg_only, viirs_key, seasonal_on, int(min_monthly), int(calib_year)
            )
        except Exception as e:
            st.error(f"Could not process {path.name}: {e}")
            st.stop()

        if hot.empty:
            st.warning("No detections left after the current quality filters.")
        else:
            data_ready = True
            dmin, dmax = info["dmin"], info["dmax"]
            with date_box:
                if dmin < dmax:
                    d0, d1 = st.slider(
                        "Date range",
                        dmin,
                        dmax,
                        (dmin, dmax),
                        format="YYYY-MM-DD",
                        label_visibility="collapsed",
                    )
                else:
                    d0, d1 = dmin, dmax
            skey = (str(path), mtime, drop_low, veg_only, viirs_key, seasonal_on,
                    int(min_monthly), str(d0), str(d1), aoi_wkt, int(calib_year))
            sub = get_sub(*skey)
            if sub.empty:
                st.warning(
                    "No hotspots in this area and period." if aoi_poly is not None else "No hotspots in this date range."
                )
                sub = None

aoi_area = polygon_area_km2(aoi_poly) if aoi_poly is not None else None
if aoi_poly is not None:
    minx, miny, maxx, maxy = aoi_poly.bounds
    sb.markdown(
        f'<div class="aoi-box">AOI ~{aoi_area:,.0f} km²<br>'
        f"{miny:.2f}° to {maxy:.2f}° N<br>{minx:.2f}° to {maxx:.2f}° E</div>",
        unsafe_allow_html=True,
    )

# ---- Summary numbers (cached; the page only unpacks them) --------------------
status = None
field_notes = []
alert_days = None
has_frp = False
verdicts, vmap, year_baseline = [], {}, None
if sub is not None:
    S = analyze(skey, region_text)
    years, first_y, last_y = S["years"], S["first_y"], S["last_y"]
    pivot_cov, pivot_full, med_by_month, full = S["pivot_cov"], S["pivot_full"], S["med_by_month"], S["full"]
    annual_by_sensor, annual_total, annual_raw = S["annual_by_sensor"], S["annual_total"], S["annual_raw"]
    full_years, trend_h, trend_raw = S["full_years"], S["trend_h"], S["trend_raw"]
    trend_viirs, viirs_years, nrt_recent = S["trend_viirs"], S["viirs_years"], S["nrt_recent"]
    latest, ly, lm, last_day = S["latest"], S["ly"], S["lm"], S["last_day"]
    current, hist, status = S["current"], S["hist"], S["status"]
    verdicts, year_baseline = S["verdicts"], S["year_baseline"]
    arch_cov, arch_full = S["arch_cov"], S["arch_full"]
    vmap = {v["year"]: v for v in verdicts}
    alert_days, has_frp, field_notes = S["alert_days"], S["has_frp"], S["field_notes"]

# ---- Masthead with live telemetry -------------------------------------------
telemetry_html = ""
if sub is not None:
    sensor_str = " + ".join(s for s in ("MODIS", "VIIRS") if s in S["sensors_present"]) or "FIRMS"
    scope_str = f"AOI {aoi_area:,.0f} km²" if aoi_poly is not None else "Whole region"
    lat_c, lon_c = S["lat_c"], S["lon_c"]
    coord_str = f"{abs(lat_c):.1f}°{'N' if lat_c >= 0 else 'S'} {abs(lon_c):.1f}°{'E' if lon_c >= 0 else 'W'}"
    age_days = (date.today() - dmax).days
    if age_days <= 0:
        age_txt = "today"
    elif age_days == 1:
        age_txt = "1 day ago"
    else:
        age_txt = f"{age_days} days ago"
    stale_cls = ' class="stale"' if age_days > 30 else ""
    stale_tip = ' title="The newest detection in this file is over 30 days old. Download a fresh FIRMS extract."' if age_days > 30 else ""
    telemetry_html = (
        f'<div class="telemetry">'
        f"<span>REGION <b>{escape(match_title)}</b></span>"
        f"<span>CENTROID <b>{coord_str}</b></span>"
        f"<span>SCOPE <b>{escape(scope_str)}</b></span>"
        f"<span>RECORD <b>{escape(sensor_str)}</b></span>"
        f"<span{stale_cls}{stale_tip}>LATEST <b>{dmax:%d %b %Y}</b> ({age_txt})</span>"
        f"</div>"
    )
elif stems:
    telemetry_html = (
        f'<div class="telemetry">'
        f"<span>COUNTRIES MAPPED <b>{len(regions)}</b></span>"
        f"<span>ARCHIVES LOADED <b>{len(stems)}</b></span>"
        f"</div>"
    )

_span_txt = "Decades of satellite fire detections"
if sub is not None:
    _sy = last_y - first_y + 1
    _span_txt = f"{_sy} year{'' if _sy == 1 else 's'} of satellite fire detections"
st.markdown(
    f'<div class="masthead">'
    f"<div><h1>The Burn Record</h1>"
    f'<p class="sub">{_span_txt} calibrated across MODIS and VIIRS, '
    f"so any month, country, or custom polygon can be read against its historical baseline.</p></div>"
    f"{telemetry_html}"
    f"</div>",
    unsafe_allow_html=True,
)

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

    # Clickable country overlay (loaded archives) — mounted under the detection
    # layers on every map so another country can be opened with one click,
    # even after zooming into a region.
    loaded_keys = tuple(k for k in regions if has_data(k) and k in geoms_wkb and k not in ALIASES)
    if loaded_keys:
        cc = st.session_state.setdefault("_ccache", {})
        if loaded_keys not in cc:
            if len(cc) > 4:
                cc.clear()
            lf = []
            for k in loaded_keys:
                g = shapely.from_wkb(geoms_wkb[k])
                lf.append({
                    "type": "Feature",
                    "properties": {"region": k.title(), "key": k, "status": "Ready ● · click to open"},
                    "geometry": mapping(g),
                })
            cc[loaded_keys] = lf
        feats = cc[loaded_keys]
        folium.GeoJson(
            {"type": "FeatureCollection", "features": feats},
            style_function=lambda f: {
                "fillColor": "#f06428",
                "fillOpacity": 0.16 if match is not None else 0.28,
                "color": "#f06428",
                "weight": 1.2 if match is not None else 1.6,
            },
            highlight_function=lambda f: {
                "fillColor": "#f06428",
                "fillOpacity": 0.45,
                "color": "#f4f0e6",
                "weight": 2.2,
            },
            tooltip=folium.GeoJsonTooltip(
                fields=["region", "status"],
                aliases=["Country", "Status"],
                sticky=True,
            ),
        ).add_to(m)

    if msub is not None:
        if layer == "Points":
            n_all, n_shown = add_points(m, msub, color_by="sensor")
            if n_shown < n_all:
                st.caption(
                    f"Showing {n_shown:,} of {n_all:,} detections from the harmonized record "
                    "(post-2012 MODIS rows are spliced out), sampled evenly per sensor. "
                    "Switch to Density grid to see full counts."
                )
        elif layer == "Day vs night (solar)":
            n_all, n_shown = add_points(m, msub, color_by="solar")
            if n_all == 0:
                st.info("This archive does not contain acquisition timestamps (`acq_time`) for day/night splitting.")
            elif n_shown < n_all:
                st.caption(
                    f"Showing {n_shown:,} of {n_all:,} timestamped detections split by local solar clock."
                )
        elif layer == "Density grid":
            cell, maxc = add_density(m, msub)
            st.caption(
                f"Grid cell about {cell:g}° (~{int(cell * 111)} km). "
                f"Densest cell: {maxc:,} VIIRS-equivalent detections."
            )
        else:
            if len(full_years) >= 8:
                recent, base = full_years[-5:], full_years[:-5]
                cell, lim = add_change(m, sub, recent, base)
                st.caption(
                    f"Mean annual detections per cell, {recent[0]}–{recent[-1]} compared with "
                    f"{base[0]}–{base[-1]}. Orange means more fire than baseline, purple means less "
                    f"(strongest change about {lim:,.1f} per year per cell). Ignores map year/month filters."
                )
            else:
                st.info("The change layer needs at least eight complete years in the selected range.")

    # Country boundary hairline for geographic framing
    if match and aoi_poly is None and match in geoms_wkb:
        country_geom = shapely.from_wkb(geoms_wkb[match])
        folium.GeoJson(
            mapping(country_geom),
            style_function=lambda f: {
                "color": "#cbc5b9",
                "weight": 1.2,
                "opacity": 0.42,
                "fillOpacity": 0.0,
            },
        ).add_to(m)

    if aoi_poly is not None:
        folium.GeoJson(
            mapping(aoi_poly),
            style_function=lambda f: {
                "color": AOI_COLOR,
                "weight": 2,
                "dashArray": "6 4",
                "fillColor": AOI_COLOR,
                "fillOpacity": 0.05,
            },
        ).add_to(m)

    if draw_mode and data_ready:
        Draw(
            export=False,
            draw_options={"polyline": False, "marker": False, "circlemarker": False},
            edit_options={"edit": False},
        ).add_to(m)

    if aoi_poly is not None:
        minx, miny, maxx, maxy = aoi_poly.bounds
        m.fit_bounds([[miny, minx], [maxy, maxx]], padding=(20, 20))
    elif bbox:
        m.fit_bounds([[bbox[0], bbox[2]], [bbox[1], bbox[3]]])
    m.get_root().html.add_child(folium.Element(ATTRIBUTION_FIX))

    ret_objs = []
    if draw_mode and data_ready:
        ret_objs = ["all_drawings"]
    else:
        # Country picks come only from clicks on the highlighted country layer.
        # last_object_clicked is the click's fresh lat/lng (last_clicked goes
        # stale because feature clicks don't always fire the map handler), and
        # the tooltip/count identify which layer was hit and dedupe repeats.
        ret_objs = ["last_object_clicked", "last_object_clicked_tooltip",
                    "last_object_clicked_count"]

    map_key = f"map-{match}-{layer}-{style}-{aoi_key}-{tag}-{st.session_state.get('_map_epoch', 0)}"
    result = st_folium(
        m,
        height=height,
        use_container_width=True,
        returned_objects=ret_objs,
        key=map_key,
    )

    if result and not (draw_mode and data_ready):
        tip = str(result.get("last_object_clicked_tooltip") or "")
        loc = result.get("last_object_clicked")
        cnt = result.get("last_object_clicked_count")
        # Only our country features carry "click to open" in their tooltip —
        # fire points, density cells and AOI borders are ignored.
        if loc and cnt is not None and "click to open" in tip.lower():
            sig = (map_key, cnt)
            if sig != st.session_state.get("_land_obj"):
                st.session_state["_land_obj"] = sig
                pt = Point(float(loc["lng"]), float(loc["lat"]))
                hits = []
                for k in [k for k in regions if k in geoms_wkb and k not in ALIASES]:
                    g = shapely.from_wkb(geoms_wkb[k])
                    if g.covers(pt):
                        hits.append((g.area, k))
                if hits:
                    # Prefer countries with a loaded archive; break ties by
                    # smallest polygon for overlapping boundaries.
                    with_data = [h for h in hits if has_data(h[1])]
                    pool = with_data or hits
                    picked = min(pool)[1]
                    # Re-clicking the country already open would just remount
                    # the map — skip the rerun entirely.
                    if picked != match:
                        st.session_state["_pending_region"] = picked
                        st.rerun()

    if draw_mode and data_ready and result and result.get("all_drawings"):
        poly = drawing_to_polygon(result["all_drawings"][-1])
        if poly is not None:
            saved = st.session_state.get("aoi_draw")
            if not saved or saved["region"] != match or saved["wkt"] != poly.wkt:
                st.session_state["aoi_draw"] = {"region": match, "wkt": poly.wkt}
                st.rerun()


def render_field_notes(notes):
    if not notes:
        return
    cards = []
    for n in notes:
        cards.append(
            f'<div class="fnote {n["key"]}">'
            f'<div class="fnote-tag">{escape(n["tag"])}</div>'
            f'<div class="fnote-head">{escape(n["head"])}</div>'
            f'<div class="fnote-body">{escape(n["body"])}</div>'
            f"</div>"
        )
    st.markdown(f'<div class="notes-grid">{"".join(cards)}</div>', unsafe_allow_html=True)


def render_predict(sub, style, match, bbox, aoi_poly, region_text, skey):
    st.markdown(
        f'<p class="lede">Two tools that look at recent and unusual activity: a conservative '
        f"<b>fire-source classifier</b> that flags persistent industrial/static heat sources so "
        f"they are not mistaken for landscape fire, and a <b>recent-activity footprint</b> that "
        f"buffers the latest ignition clusters outward. Both are approximations — read the caveats.</p>",
        unsafe_allow_html=True,
    )

    # ---- Source classification ----
    st.markdown('<div class="sb-label">Fire source classification</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns([2, 1, 1])
    c1.caption(
        "Classified per ~1 km cell across the whole selected period. Only cells that recur "
        "year-round on many separate days are flagged as industrial/static sources; everything "
        "else stays 'Vegetation / agricultural (unresolved)' — FIRMS cannot separate field "
        "burning from wildfire without land-cover data. FIRMS type 1–3 detections are labelled "
        "directly when the vegetation-only filter is off."
    )
    show_classes = c2.checkbox("Classify sources", True, key="src_class_on")
    if show_classes:
        summ = source_summary(skey)
        counts = summ.set_index("cls")["viirs_eq"]
        det_counts = summ.set_index("cls")["detections"]
    else:
        counts = pd.Series({"Vegetation / agricultural (unresolved)": float(sub["weight"].sum())}, dtype=float)
        det_counts = pd.Series({"Vegetation / agricultural (unresolved)": float(len(sub))}, dtype=float)

    order = [c for c in SRC_CLASSES if c in counts.index] + [
        c for c in counts.index if c not in SRC_CLASSES]
    if len(counts):
        chart_df = pd.DataFrame({
            "cls": order,
            "viirs_eq": counts.reindex(order).fillna(0).to_numpy(),
            "detections": det_counts.reindex(order).fillna(0).round(0).astype(int).to_numpy(),
        })
        fig = px.bar(
            chart_df, x="viirs_eq", y="cls", orientation="h",
            labels=dict(viirs_eq="VIIRS-equivalent detections", cls="", detections="Raw detections"),
            color="cls",
            color_discrete_map={c: SRC_COLORS.get(c, "#8a857c") for c in order},
            hover_data={"detections": True, "viirs_eq": ":.0f"},
        )
        fig.update_layout(height=max(220, 34 * len(order)), showlegend=False, margin=dict(l=8, r=8, t=8, b=8))
        show(tidy(fig, "", 240), "srctypes")

    # ---- Recent activity footprint ----
    st.markdown('<div class="sb-label">Recent activity footprint (heuristic)</div>', unsafe_allow_html=True)
    p1, p2, p3 = st.columns(3)
    window = p1.select_slider("Recent window (days)", options=[7, 14, 30, 60, 120], value=30,
                              key="spread_window", format_func=lambda d: f"{d} d")
    rate = p2.select_slider("Spread rate (km/day)", options=[0.5, 1, 2, 3, 5], value=2,
                            key="spread_rate", format_func=lambda r: f"{r:g} km/d")
    horizon = p3.select_slider("Project ahead (days)", options=[1, 3, 7, 14], value=7,
                               key="spread_horizon", format_func=lambda d: f"{d} d")

    cutoff = sub["acq_date"].max() - pd.Timedelta(window, unit="D")
    src = sub[sub["acq_date"] >= cutoff]
    st.caption(
        f"Using {fmt(len(src))} detections ending {sub['acq_date'].max():%d %b %Y} "
        f"(the 'recent' window is relative to the end of your selected period). Circles buffer "
        f"each active cluster by {rate * horizon:g} km at {rate:g} km/day — an upper-bound "
        f"envelope, not a prediction. No wind, terrain, fuel, or still-burning assumption."
    )
    if src.empty:
        st.info("No detections in that recent window — widen it or pick another period.")
        return

    cell = 0.1
    cl = src.assign(
        glat=(src["latitude"] / cell).round() * cell,
        glon=(src["longitude"] / cell).round() * cell,
    ).groupby(["glat", "glon"], as_index=False).agg(
        n=("weight", "size"), w=("weight", "sum"),
        la=("latitude", "mean"), lo=("longitude", "mean"),
    ).sort_values("w", ascending=False).head(30)

    m = base_map(style if style in ("Dark", "Light", "Satellite") else "Dark")
    max_w = float(cl["w"].max()) if len(cl) else 1.0
    for _, r in cl.iterrows():
        intensity = r["w"] / max_w
        col = "#f06428" if intensity > 0.66 else ("#e8a13c" if intensity > 0.33 else "#c96f2e")
        folium.Circle(
            location=[r["la"], r["lo"]],
            radius=rate * horizon * 1000,
            color=col, weight=1.5, opacity=0.85,
            fill=True, fill_color=col, fill_opacity=0.10 + 0.18 * intensity,
            popup=(f"{int(r['n']):,} detections in window<br>"
                   f"Projected reach: {rate * horizon:g} km"),
        ).add_to(m)
        folium.CircleMarker(
            location=[r["la"], r["lo"]], radius=4, color="#f4f0e6", weight=1,
            fill=True, fill_color=col, fill_opacity=0.95,
            tooltip=f"{int(r['n']):,} recent detections",
        ).add_to(m)

    legend_html = (
        f'<div style="{MAP_LEGEND_STYLE}">'
        "<b>Recent activity footprint</b><br>"
        '<span style="color:#f06428">●</span> high-activity cluster &nbsp;'
        '<span style="color:#e8a13c">●</span> medium &nbsp;'
        '<span style="color:#c96f2e">●</span> lower<br>'
        f"circles = {rate * horizon:g} km reach at {rate:g} km/day over {horizon} d"
        "</div>"
    )
    m.get_root().html.add_child(folium.Element(legend_html))

    if aoi_poly is not None:
        folium.GeoJson(
            mapping(aoi_poly),
            style_function=lambda f: {"color": AOI_COLOR, "weight": 2, "dashArray": "6 4",
                                      "fillColor": AOI_COLOR, "fillOpacity": 0.05},
        ).add_to(m)
    elif bbox:
        m.fit_bounds([[bbox[0], bbox[2]], [bbox[1], bbox[3]]])
    else:
        m.fit_bounds([[cl["la"].min() - 0.5, cl["lo"].min() - 0.5],
                      [cl["la"].max() + 0.5, cl["lo"].max() + 0.5]])
    st_folium(m, height=560, use_container_width=True, returned_objects=[],
              key=f"spread-{match}-{window}-{rate}-{horizon}-{aoi_key}")

    st.caption(
        f"Top {len(cl)} ignition clusters from {fmt(len(src))} recent detections. "
        "Treat the circles as an upper-bound envelope, not a prediction of perimeter."
    )


def render_overview():
    render_stripes(verdicts, title="Every year at a glance")
    st.caption(
        "One stripe per year. Orange marks heavier fire years than usual, purple marks quieter years, "
        "and dark grey-green sits near the median. Hover any stripe for its verdict."
    )

    # Field notes derived from the live region/AOI data
    render_field_notes(field_notes)

    left, right = st.columns([4, 8], gap="large")

    with left:
        st.markdown(
            f'<div class="region-head">'
            f'<span class="title">{escape(region_text)}</span>'
            f'<span class="span">{d0:%Y}–{d1:%Y}</span>'
            f"</div>",
            unsafe_allow_html=True,
        )
        if status is not None:
            if status["median"] > 0:
                ratio_txt = f"{current / status['median']:.1f}× the median ({fmt(status['median'])})"
            else:
                ratio_txt = "where the median year had none"
            badge = LEVEL_LABELS.get(status["level"], "Status")
            st.markdown(
                f'<div class="verdict {status["level"]}">'
                f'<div class="verdict-meta">{escape(badge)} · Rank {status["rank"]} of {status["n"]}</div>'
                f'<div class="verdict-head">{HERO_LINES[status["level"]]}</div>'
                f'<div class="verdict-body">{calendar.month_name[lm]} 1–{last_day}, {ly}: {fmt(current)} detections, '
                f"{ratio_txt} for the same days across the {len(hist)} earlier years.</div></div>",
                unsafe_allow_html=True,
            )

        a, b = st.columns(2)
        a.metric(
            "Hotspots detected",
            fmt(S["n_rows"]),
            help="Individual detections kept after quality filters and the MODIS/VIIRS splice.",
        )
        b.metric(
            "Harmonized total",
            fmt(S["w_sum"]),
            help="VIIRS-equivalent detections: per-satellite normalized, MODIS scaled.",
        )
        c, d = st.columns(2)
        c.metric(
            "Peak month",
            calendar.month_name[int(med_by_month.idxmax())] if med_by_month.notna().any() else "n/a",
            help="Calendar month with the highest median across complete months.",
        )
        label = f"{calendar.month_abbr[lm]} {ly}, days 1–{last_day}"
        if status is not None:
            d.metric(
                label,
                fmt(current),
                delta=f"{current - status['median']:+,.0f} vs median",
                delta_color="inverse",
                help=f"Median of the same days in earlier years: {fmt(status['median'])}.",
            )
        else:
            d.metric(label, fmt(current))
        e, g = st.columns(2)
        if trend_h is not None:
            e.metric(
                "Annual trend",
                f"{trend_h[0]:+.1f}% / yr",
                help=f"Theil-Sen slope over {len(full_years)} complete years, relative to the median year.",
            )
        else:
            e.metric("Annual trend", "n/a", help="Needs at least five complete years in the selected range.")
        g.metric("Latest data", f"{dmax:%d %b %Y}")

        if has_frp and S["frp_sum"] > 0:
            fr1, fr2 = st.columns(2)
            fr1.metric(
                "Total FRP",
                f"{S['frp_sum']:,.0f} MW",
                help="Sum of per-detection fire radiative power (MW) across the period, "
                "averaged over the active satellite constellation. An intensity proxy, "
                "not a burned-area measurement.",
            )
            fr2.metric(
                "Peak detection",
                f"{S['frp_max']:,.0f} MW",
                help="Largest single-detection FRP in the period — often a major fire front "
                "or an industrial heat source.",
            )

        if med_by_month.notna().any() and med_by_month.max() > 0:
            top = float(med_by_month.max())
            cells = []
            for mo in range(1, 13):
                v = med_by_month.get(mo, np.nan)
                frac = 0.0 if pd.isna(v) else float(v) / top
                ring = "box-shadow: inset 0 0 0 2px #8fb369;" if mo == lm else ""
                ink = "#0c0d0b" if frac > 0.56 else "#f4f0e6"
                tip = f"{calendar.month_name[mo]}: median {fmt(0 if pd.isna(v) else v)} detections"
                cells.append(
                    f'<div class="ribbon-cell" title="{tip}" style="background: rgba(240,100,40,{0.08 + 0.88 * frac:.2f});'
                    f' color: {ink}; {ring}">{calendar.month_abbr[mo]}</div>'
                )
            st.markdown(
                '<div class="ribbon-title">When this area usually burns</div>'
                f'<div class="ribbon">{"".join(cells)}</div>'
                '<p class="note" style="margin-top:.4rem">Median detections per calendar month across complete '
                "years. The outlined cell marks the latest month in the archive.</p>",
                unsafe_allow_html=True,
            )

        if info["has_both"]:
            ci = info["factor_ci"]
            ci_txt = f" (range {ci[0]:.2f}–{ci[1]:.2f})" if ci else ""
            st.caption(
                f"MODIS scaled ×{info['factor']:.2f}{ci_txt} to VIIRS-equivalent from {info['n_valid']} "
                f"overlap months (fit through {calib_year}). {fmt(info['n_final'])} of {fmt(info['n_raw'])} detections kept."
            )
        if nrt_recent:
            st.caption(
                "The tail of this record may be near-real-time (NRT) detections rather than the "
                "standard reprocessed product, so the most recent weeks carry extra uncertainty."
            )

    with right:
        years_all = S["years_present"]
        m_ctrl1, m_ctrl2 = st.columns([3, 1], gap="small")
        m_ctrl1.caption(
            "Filter the map by year or months. Summary figures on the left always cover the full period."
        )
        peak_3 = (
            sorted(int(m) for m in med_by_month.dropna().nlargest(3).index)
            if med_by_month.notna().any() and med_by_month.max() > 0
            else list(range(1, 13))
        )
        if m_ctrl2.button("Peak season", key=f"peak_btn_{match}_{aoi_key}", help="Isolate the 3 busiest calendar months on the map"):
            st.session_state["map_months"] = peak_3
            st.rerun()

        year_pick = st.select_slider(
            "Year on the map",
            options=["All years"] + years_all,
            value="All years",
            key=f"map_year_{match}_{years_all[0]}_{years_all[-1]}_{aoi_key}",
        )
        month_opts = list(range(1, 13))
        st.session_state.setdefault("map_months", month_opts)
        if hasattr(st, "pills"):
            month_pick = st.pills(
                "Months on the map",
                month_opts,
                selection_mode="multi",
                format_func=lambda mo: calendar.month_abbr[mo],
                key="map_months",
            )
        else:
            month_pick = st.multiselect(
                "Months on the map",
                month_opts,
                format_func=lambda mo: calendar.month_abbr[mo],
                key="map_months",
            )
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
                st.caption(f"{fmt(len(msub))} detections in this map window.")
            render_map(msub, tag)

    if alert_days is not None and len(alert_days):
        with st.expander(f"Unusual activity days — top {len(alert_days)} across the archive", expanded=False):
            st.dataframe(alert_days, hide_index=True)
            st.caption(
                "Days where detections ran at least 2× (and 10+ detections) the median of the "
                "same day-of-year across other complete years. Computed over the whole record "
                "for the current area and filters, so it stays meaningful for short date ranges. "
                "A spike can mean a real outbreak, a sensor artefact, or persistent industrial heat."
            )


def month_row(piv, y):
    """One year's twelve monthly values, or an all-empty row if that year is absent."""
    if piv is not None and y in piv.index:
        return piv.loc[y].astype(float)
    return pd.Series(np.nan, index=range(1, 13))


# Main workspace
if sub is not None:
    # Section nav: only the active section's body executes on each rerun,
    # which keeps the heavy tabs (timelapse, predict) from slowing every click.
    _nav_labels = ["Overview", "Year by year", "Calendar & clock", "Trends & pacing",
                   "Timelapse", "Recent activity & sources", "Harmonization", "Data"]
    _nav_keys = ["overview", "year", "cal", "trend", "time", "pred", "harm", "data"]
    _qp = st.query_params.get("tab", "overview")
    _nav_i = _nav_keys.index(_qp) if _qp in _nav_keys else 0
    st.session_state.setdefault("section_nav", _nav_labels[_nav_i])
    if hasattr(st, "pills"):
        _sel = st.pills("Section", _nav_labels, selection_mode="single",
                        key="section_nav", label_visibility="collapsed")
        if isinstance(_sel, (list, tuple)):
            _sel = _sel[0] if _sel else _nav_labels[_nav_i]
    else:
        _sel = st.radio("Section", _nav_labels,
                        index=_nav_labels.index(st.session_state["section_nav"])
                        if st.session_state["section_nav"] in _nav_labels else _nav_i,
                        horizontal=True, label_visibility="collapsed", key="section_nav")
    _sel = _sel if _sel in _nav_labels else "Overview"
    _tab_key = _nav_keys[_nav_labels.index(_sel)]
    if st.query_params.get("tab") != _tab_key:
        st.query_params["tab"] = _tab_key

    if _sel == "Overview":
        render_overview()

    elif _sel == "Year by year":
        st.markdown(
            f'<p class="lede">{escape(years_summary(verdicts, region_text, year_baseline))}</p>',
            unsafe_allow_html=True,
        )
        if not verdicts:
            st.info("There is no archive record for this area yet, so years cannot be rated.")
        else:
            opts = [v["year"] for v in verdicts]
            yc1, yc2 = st.columns([3, 1], gap="medium")
            pick = yc1.select_slider(
                "Year in focus",
                options=opts,
                value=opts[-1],
                key=f"year_pick_{match}_{aoi_key}_{opts[0]}_{opts[-1]}",
            )
            comp_opts = ["None"] + [str(y) for y in reversed(opts) if y != pick]
            comp_pick = yc2.selectbox("Compare with year", comp_opts, index=0, key=f"year_comp_{match}_{aoi_key}")

            render_stripes(verdicts, title="Annual anomaly strip (selected year outlined)", active_year=pick)
            vy = vmap[pick]
            y_badge = LEVEL_LABELS.get(vy["level"], "Year verdict")
            rank_suffix = f" · Rank {vy['rank']} of {vy['n']}" if vy.get("rank") else ""
            st.markdown(
                f'<div class="verdict {vy["level"]}">'
                f'<div class="verdict-meta">{escape(y_badge)}{escape(rank_suffix)}</div>'
                f'<div class="verdict-head">{pick}: {escape(vy["headline"])}</div>'
                f'<div class="verdict-body">{escape(vy["detail"])}</div></div>',
                unsafe_allow_html=True,
            )

            # The verdicts cover the whole archive, so the chart reads the archive's
            # monthly table too (the date slider only covers part of it).
            mn = [calendar.month_abbr[c] for c in range(1, 13)]
            row = month_row(arch_cov, pick)
            qs = arch_full.quantile([0.25, 0.5, 0.75])
            mfig = go.Figure()
            if qs.notna().any().any():
                mfig.add_trace(
                    go.Scatter(x=mn, y=qs.loc[0.75].to_numpy(), mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip")
                )
                mfig.add_trace(
                    go.Scatter(
                        x=mn,
                        y=qs.loc[0.25].to_numpy(),
                        mode="lines",
                        line=dict(width=0),
                        fill="tonexty",
                        fillcolor="rgba(146,141,130,.18)",
                        name="Middle 50% of years",
                        hoverinfo="skip",
                    )
                )
                mfig.add_trace(
                    go.Scatter(
                        x=mn,
                        y=qs.loc[0.5].to_numpy(),
                        mode="lines",
                        line=dict(color=OTHER_COLOR, width=2, dash="dash"),
                        name="Usual month",
                    )
                )
            mfig.add_bar(x=mn, y=row.to_numpy(), name=str(pick), marker_color=LEVEL_COLORS[vy["level"]])
            if comp_pick != "None":
                cy = int(comp_pick)
                mfig.add_bar(
                    x=mn,
                    y=month_row(arch_cov, cy).to_numpy(),
                    name=str(cy),
                    marker_color="rgba(244,240,230,0.34)",
                )
                mfig.update_layout(barmode="group")

            chart_title = (
                f"{pick} vs {comp_pick} month by month"
                if comp_pick != "None"
                else f"{pick} month by month against the usual year"
            )
            show(tidy(mfig, chart_title, 340), "yearchart")

            cards = []
            for v in verdicts:
                active_cls = " active" if v["year"] == pick else ""
                cards.append(
                    f'<div class="yearcard {v["level"]}{active_cls}" title="{escape(v["detail"], quote=True)}">'
                    f'<div><div class="y">{v["year"]}</div><div class="h">{escape(v["headline"])}</div></div>'
                    f'<div class="s">{escape(v["short"])}</div></div>'
                )
            st.markdown(f'<div class="yeargrid">{"".join(cards)}</div>', unsafe_allow_html=True)
            st.caption(
                "Each year is rated against the usual (median) complete year in the whole archive for the "
                "selected area, so the date slider does not change these verdicts. Hover any card for the full breakdown."
            )

    elif _sel == "Calendar & clock":
        ca, cb = st.columns([2, 1])
        mode_c = ca.radio("Show", ["Detections", "Anomaly vs median"], horizontal=True, key="calmode")
        ramp = cb.selectbox(
            "Colour ramp",
            ["Inferno", "Viridis", "Cividis"],
            key="ramp",
            help="Perceptually uniform ramps readable with colour-vision deficiency.",
            disabled=mode_c != "Detections",
        )
        mnames = [calendar.month_abbr[c] for c in range(1, 13)]
        if mode_c == "Detections":
            heat_df = pivot_cov.copy()
            heat_df.columns = mnames
            heat = px.imshow(
                heat_df,
                color_continuous_scale=ramp,
                aspect="auto",
                labels=dict(x="Month", y="Year", color="Detections"),
            )
            heat.update_traces(texttemplate="%{z:.2~s}", textfont_size=9)
            heat_title = f"Monthly detections, {region_text}"
        else:
            anom = (pivot_full - med_by_month) / med_by_month.replace(0, np.nan) * 100
            anom.columns = mnames
            vals = np.abs(anom.to_numpy(float))
            lim = float(np.nanpercentile(vals, 95)) if np.isfinite(vals).any() else 100.0
            lim = min(max(lim, 50.0), 300.0)
            heat = px.imshow(
                anom,
                color_continuous_scale="PuOr_r",
                zmin=-lim,
                zmax=lim,
                aspect="auto",
                labels=dict(x="Month", y="Year", color="% vs median"),
            )
            heat.update_traces(texttemplate="%{z:+.0f}%", textfont_size=9)
            heat_title = f"Deviation from each calendar month's median, {region_text}"
        heat.update_yaxes(type="category", automargin=True, tickfont=dict(color="#928d82"))
        heat.update_xaxes(automargin=True, tickfont=dict(color="#928d82"))
        heat.update_layout(
            height=max(300, 30 * len(pivot_cov) + 140),
            margin=dict(l=6, r=6, t=56, b=8),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            title=dict(
                text=heat_title,
                x=0,
                xanchor="left",
                y=0.98,
                yanchor="top",
                yref="container",
                font=dict(size=14, color="#f4f0e6"),
            ),
            coloraxis_colorbar=dict(title=None, thickness=12, len=0.9, tickfont=dict(color="#928d82")),
            hoverlabel=dict(
                bgcolor="#171916",
                bordercolor="rgba(244,240,230,0.28)",
                font=dict(size=12, family=FONT_PLOT, color="#f4f0e6"),
            ),
            font=dict(family=FONT_PLOT, size=12, color="#cbc5b9"),
        )
        show(heat, "calendar")
        st.caption(
            "Partial months at the edges of the range appear in the detections view and are "
            "left blank in the anomaly view. Anomaly colours: orange above the median, purple below."
        )

        # Diurnal solar clock chart if acq_time was present
        hp = hour_profile(skey)
        if hp is not None:
            hfig = go.Figure()
            for s, nm in (("MODIS", "MODIS"), ("VIIRS", "VIIRS"), ("UNKNOWN", "Other")):
                if s in hp.columns and hp[s].sum() > 0:
                    hfig.add_bar(
                        x=[f"{h:02d}:00" for h in range(24)],
                        y=hp[s].to_numpy(),
                        name=nm,
                        marker_color=SENSOR_COLORS[s],
                    )
            hfig.update_layout(barmode="stack")
            show(tidy(hfig, "24-hour local solar burn clock (sun-synchronous overpass timing)", 300), "diurnal")
            st.caption(
                "Approximate local solar hour derived from UTC acquisition time and longitude. "
                "Peaks reflect when fires are active during Terra (~10:30/22:30) and Aqua/VIIRS (~13:30/01:30) passes."
            )

        flat = (
            pivot_full.rename_axis("year")
            .reset_index()
            .melt(id_vars="year", var_name="month", value_name="n")
            .dropna()
        )
        tl, tr_ = st.columns(2)
        if not flat.empty:
            top_m = flat.nlargest(8, "n").copy()
            top_m["Month"] = [f"{calendar.month_abbr[int(a)]} {int(b)}" for a, b in zip(top_m["month"], top_m["year"])]
            top_m["Detections"] = top_m["n"].round(0).astype(int)
            top_m["Times the median"] = [
                f"{n / med_by_month[int(mo)]:.1f}×" if med_by_month[int(mo)] > 0 else "n/a"
                for n, mo in zip(top_m["n"], top_m["month"])
            ]
            tl.markdown("**Most active months**")
            tl.dataframe(top_m[["Month", "Detections", "Times the median"]], hide_index=True)
        if len(full_years) >= 1:
            top_y = annual_total.loc[full_years].nlargest(8)
            tr_.markdown("**Most active complete years**")
            tr_.dataframe(
                pd.DataFrame({"Year": top_y.index.astype(int), "Detections": top_y.round(0).astype(int).to_numpy()}),
                hide_index=True,
            )

        q = pivot_full.quantile([0.25, 0.5, 0.75])
        if q.notna().any().any():
            sc = go.Figure()
            sc.add_trace(
                go.Scatter(x=mnames, y=q.loc[0.75].to_numpy(), mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip")
            )
            sc.add_trace(
                go.Scatter(
                    x=mnames,
                    y=q.loc[0.25].to_numpy(),
                    mode="lines",
                    line=dict(width=0),
                    fill="tonexty",
                    fillcolor="rgba(146,141,130,.18)",
                    name="Middle 50% of years",
                    hoverinfo="skip",
                )
            )
            sc.add_trace(
                go.Scatter(x=mnames, y=q.loc[0.5].to_numpy(), mode="lines", line=dict(color=OTHER_COLOR, width=2), name="Median")
            )
            sc.add_trace(
                go.Scatter(
                    x=mnames,
                    y=pivot_cov.loc[last_y].to_numpy(),
                    mode="lines+markers",
                    line=dict(color=VIIRS_COLOR, width=2.5),
                    marker=dict(size=6),
                    name=str(last_y),
                )
            )
            show(tidy(sc, "Seasonal cycle against the latest year", 340), "season")

    elif _sel == "Trends & pacing":
        cmp_mode = st.selectbox(
            "Benchmark this period against",
            ["Nothing", "Same dates, previous year", "Median year, same months"],
            key=f"cmp_{match}_{aoi_key}",
            help="Adds a comparison baseline to the annual and monthly views below.",
        )

        ann = go.Figure()
        ci = info["factor_ci"]
        for s, nm in (("MODIS", "MODIS-derived"), ("VIIRS", "VIIRS"), ("UNKNOWN", "Unlabelled")):
            if s in annual_by_sensor:
                kw = {}
                if s == "MODIS" and ci and not scaling.startswith("By") and info["factor"] > 0:
                    v = annual_by_sensor[s]
                    kw["error_y"] = dict(
                        type="data",
                        symmetric=False,
                        thickness=1,
                        width=2,
                        array=(v * (ci[1] / info["factor"] - 1)).to_numpy(),
                        arrayminus=(v * (1 - ci[0] / info["factor"])).to_numpy(),
                    )
                ann.add_bar(
                    x=annual_by_sensor.index,
                    y=annual_by_sensor[s],
                    name=nm,
                    marker_color=SENSOR_COLORS[s],
                    **kw,
                )
        ann.update_layout(barmode="stack")
        if trend_h is not None:
            xs = np.array([min(full_years), max(full_years)])
            ann.add_scatter(
                x=xs,
                y=trend_h[1] * xs + trend_h[2],
                mode="lines",
                name="Trend (Theil-Sen)",
                line=dict(color=SAGE_COLOR, dash="dash", width=2),
            )
        ann.update_xaxes(dtick=2)
        show(tidy(ann, "Annual detections, VIIRS-equivalent (partial years included)"), "annual")
        if ci and not scaling.startswith("By"):
            st.caption("Whiskers on the MODIS-derived bars show the bootstrap range of the scaling factor.")
        trend_bits = []
        if trend_h is not None:
            trend_bits.append(f"all complete years {trend_h[0]:+.1f}%/yr")
        if trend_viirs is not None:
            trend_bits.append(
                f"VIIRS era only ({viirs_years[0]}+, {len(viirs_years)} yrs) {trend_viirs[0]:+.1f}%/yr"
            )
        if trend_bits:
            st.caption(
                "Theil-Sen trend — " + "; ".join(trend_bits) + ". "
                "The VIIRS-era figure uses only post-splice years, so it does not depend on the "
                "MODIS scaling factor; a large gap between the two is a red flag for calibration bias."
            )

        by_month_sel = S["monthly_agg"].groupby("month")["detections_viirs_equiv"].sum()
        if cmp_mode == "Same dates, previous year":
            d0p = (pd.Timestamp(d0) - pd.DateOffset(years=1)).date()
            d1p = (pd.Timestamp(d1) - pd.DateOffset(years=1)).date()
            comp = hot[(hot["acq_date"] >= pd.Timestamp(d0p)) & (hot["acq_date"] <= pd.Timestamp(d1p))]
            if aoi_poly is not None:
                comp = apply_aoi(comp, aoi_poly)
            if comp.empty:
                st.info("No detections in the same dates of the previous year for this area.")
            else:
                w_sub, w_cmp = S["w_sum"], float(comp["weight"].sum())
                pct = (w_sub - w_cmp) / w_cmp * 100 if w_cmp > 0 else np.nan
                m1, m2 = st.columns(2)
                m1.metric(
                    "Selected period",
                    f"{fmt(w_sub)}",
                    help="VIIRS-equivalent detections in the selected window.",
                )
                m2.metric(
                    "Previous year, same dates",
                    f"{fmt(w_cmp)}",
                    delta=(f"{pct:+.0f}%" if np.isfinite(pct) else None),
                    delta_color="inverse",
                    help="Same calendar dates one year earlier, same area and filters.",
                )
                comp_mon = comp.groupby("month")["weight"].sum()
                mon = pd.DataFrame({
                    "month": range(1, 13),
                    "Selected": [float(by_month_sel.get(mo, 0.0)) for mo in range(1, 13)],
                    "Previous year": [float(comp_mon.get(mo, 0.0)) for mo in range(1, 13)],
                })
                cfig = go.Figure()
                for col, nm, clr in (("Selected", "Selected period", VIIRS_COLOR), ("Previous year", "Previous year", "rgba(203,197,185,0.75)")):
                    cfig.add_bar(x=[calendar.month_abbr[m] for m in mon["month"]], y=mon[col], name=nm, marker_color=clr)
                cfig.update_layout(barmode="group")
                show(tidy(cfig, "Monthly detections: selected period vs same dates last year", 320), "cmp_year")
                st.caption(
                    "Both windows use the same area and filters. Months outside the selected "
                    "range are simply absent from the selected series."
                )
        elif cmp_mode == "Median year, same months":
            months_in = sorted(int(m) for m in pivot_cov.columns[pivot_cov.notna().any()])
            med = med_by_month.reindex(months_in)
            cfig = go.Figure()
            cfig.add_bar(
                x=[calendar.month_abbr[m] for m in months_in],
                y=[float(by_month_sel.get(m, 0.0)) for m in months_in],
                name="Selected period",
                marker_color=VIIRS_COLOR,
            )
            cfig.add_scatter(
                x=[calendar.month_abbr[m] for m in months_in],
                y=med.to_numpy(float),
                mode="lines+markers",
                name="Median of complete years",
                line=dict(color=SAGE_COLOR, width=2, dash="dash"),
            )
            show(tidy(cfig, "Selected period vs a median year, by calendar month", 320), "cmp_med")
            st.caption(
                "The dashed line is the median of the complete years inside the selected range "
                "for each calendar month — a 'typical year' benchmark, not a single past year."
            )

        if has_frp and S["frp_sum"] > 0:
            fr = S["monthly_agg"].groupby("year")["frp_mw"].sum()
            ffig = go.Figure(
                go.Scatter(
                    x=fr.index,
                    y=fr.to_numpy(),
                    mode="lines+markers",
                    line=dict(color="#f06428", width=2),
                    fill="tozeroy",
                    fillcolor="rgba(240,100,40,0.14)",
                    name="FRP",
                )
            )
            show(tidy(ffig, "Annual fire radiative power (FRP)", 300), "frp_year")
            st.caption(
                "FRP is satellite-measured fire intensity in megawatts, summed per year and "
                "averaged across the active constellation. It is an intensity proxy — not a "
                "burned-area measurement — and can be dominated by a few large fires or "
                "persistent industrial heat sources."
            )

        # Cumulative year-over-year burn pacing
        if len(full_years) >= 3:
            daily = cumulative_daily(skey, tuple(years))
            month_starts = [pd.Timestamp(2021, m, 1).dayofyear for m in range(1, 13)]
            month_labels = [calendar.month_abbr[m] for m in range(1, 13)]

            spot_opts = sorted([int(y) for y in daily.index], reverse=True)
            spot_year = st.selectbox(
                "Spotlight year on cumulative curve",
                spot_opts,
                index=0,
                key=f"cum_spot_{match}_{aoi_key}",
            )

            cfig = go.Figure()
            for y in full_years:
                if y == spot_year:
                    continue
                cfig.add_trace(
                    go.Scatter(
                        x=list(range(1, 366)),
                        y=daily.loc[y].to_numpy(),
                        mode="lines",
                        line=dict(color="rgba(203,197,185,0.16)", width=1),
                        name=str(y),
                        showlegend=False,
                        hovertemplate=f"{y} · Day %{{x}}: %{{y:,.0f}}<extra></extra>",
                    )
                )

            med_cum = daily.loc[full_years].median()
            cfig.add_trace(
                go.Scatter(
                    x=list(range(1, 366)),
                    y=med_cum.to_numpy(),
                    mode="lines",
                    line=dict(color=SAGE_COLOR, width=2, dash="dash"),
                    name="Median year",
                )
            )

            spot_vals = daily.loc[spot_year].to_numpy()
            if spot_year == ly and ly not in full_years:
                cutoff = min(int(latest.dayofyear), 365)
                xs_spot = list(range(1, cutoff + 1))
                ys_spot = spot_vals[:cutoff]
            else:
                xs_spot = list(range(1, 366))
                ys_spot = spot_vals

            cfig.add_trace(
                go.Scatter(
                    x=xs_spot,
                    y=ys_spot,
                    mode="lines",
                    line=dict(color=VIIRS_COLOR, width=3),
                    name=str(spot_year),
                )
            )
            if len(xs_spot):
                cfig.add_trace(
                    go.Scatter(
                        x=[xs_spot[-1]],
                        y=[ys_spot[-1]],
                        mode="markers",
                        marker=dict(color=VIIRS_COLOR, size=8, line=dict(color="#0c0d0b", width=1.5)),
                        showlegend=False,
                        hoverinfo="skip",
                    )
                )
            cfig.update_xaxes(tickmode="array", tickvals=month_starts, ticktext=month_labels)
            show(tidy(cfig, f"Cumulative burn pacing: {spot_year} against all complete years", 360), "cumpace")

        if status is not None:
            vals = pd.concat([hist, pd.Series({ly: current})])
            bar = go.Figure(
                go.Bar(
                    x=vals.index,
                    y=vals.to_numpy(),
                    marker_color=["rgba(146,141,130,0.48)"] * len(hist) + [VIIRS_COLOR],
                    showlegend=False,
                )
            )
            bar.add_scatter(
                x=[vals.index.min(), vals.index.max()],
                y=[status["median"]] * 2,
                mode="lines",
                name="Median of past years",
                line=dict(color=SAGE_COLOR, dash="dash", width=2),
            )
            bar.update_xaxes(dtick=2)
            show(tidy(bar, f"{calendar.month_name[lm]}, days 1–{last_day}, every year"), "monthcmp")

    elif _sel == "Timelapse":
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

            tl, cell_t = timelapse_grid(skey)
            cap = max(float(tl["w"].quantile(0.99)), 1.0)

            tile_style = style if style in ("Dark", "Light", "Satellite") else "Dark"
            layer_def = dict(
                below="traces",
                sourcetype="raster",
                source=[TILES[tile_style]["tiles"]],
                sourceattribution="Tiles © Esri",
            )
            args = dict(
                lat="lat",
                lon="lon",
                z="w",
                radius=14,
                animation_frame="year",
                center=dict(lat=(lat_lo + lat_hi) / 2, lon=(lon_lo + lon_hi) / 2),
                zoom=zoom_t,
                color_continuous_scale="YlOrRd",
                range_color=(0, cap),
                opacity=0.85,
            )
            if hasattr(px, "density_map"):
                fig_t = px.density_map(tl, map_style="white-bg", **args)
                fig_t.update_layout(map_layers=[layer_def])
            else:
                fig_t = px.density_mapbox(tl, mapbox_style="white-bg", **args)
                fig_t.update_layout(mapbox_layers=[layer_def])
            fig_t.update_layout(
                height=640,
                margin=dict(l=0, r=0, t=0, b=0),
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(family=FONT_PLOT, color="#cbc5b9"),
                coloraxis_colorbar=dict(title="Detections / cell", thickness=12, tickfont=dict(color="#928d82")),
            )
            st.caption(
                "Press play to watch the burning season by season. Each frame is one year of "
                f"VIIRS-equivalent detections on a grid of about {cell_t:g}°. "
                "The colour scale is locked across every frame so years can be compared directly."
            )
            show(fig_t, "timelapse")
        except Exception as exc:
            st.info(f"The timelapse could not be drawn here ({exc}). Update plotly to 5.24 or newer and try again.")

    elif _sel == "Recent activity & sources":
        render_predict(sub, style, match, bbox, aoi_poly, region_text, skey)

    elif _sel == "Harmonization":
        if not info["has_both"]:
            st.info(
                "This file has one sensor only, so no MODIS/VIIRS harmonization is needed. "
                "Detections are still normalized per satellite."
            )
        else:
            split = info["split"]
            st.markdown(
                f'<div class="method-list">'
                f'<div class="method-item"><b>1. Filter</b>'
                f"<p>Low-confidence and non-vegetation detections are removed per your sidebar settings.</p></div>"
                f'<div class="method-item"><b>2. Per-satellite weight</b>'
                f"<p>Each detection is divided by the active satellite count of its sensor on that date.</p></div>"
                f'<div class="method-item"><b>3. Cross-calibration</b>'
                f'<p>From {split:%B %Y}, overlapping months up to {calib_year} give a median VIIRS / MODIS ratio of ×{info["factor"]:.2f} across the full file; later months are excluded to limit orbit-drift bias.</p></div>'
                f'<div class="method-item"><b>4. Splice</b>'
                f"<p>Scaled MODIS covers the record before {split:%B %Y}; VIIRS covers {split:%B %Y} onward without double-counting.</p></div>"
                f"</div>",
                unsafe_allow_html=True,
            )
            ci = info["factor_ci"]
            h1, h2, h3, h4 = st.columns(4)
            h1.metric(
                "Scaling factor",
                f"×{info['factor']:.2f}",
                delta=f"range {ci[0]:.2f}–{ci[1]:.2f}" if ci else None,
                delta_color="off",
                help="Bootstrap 95% range over the overlap months.",
            )
            h2.metric("Overlap months used", info["n_valid"])
            h3.metric(
                "Monthly correlation",
                "n/a" if np.isnan(info["corr"]) else f"{info['corr']:.2f}",
                help="Pearson r between MODIS and VIIRS monthly counts in the overlap.",
            )
            err = info["err_seasonal"] if scaling.startswith("By") else info["err_single"]
            h4.metric(
                "Median monthly error",
                "n/a" if np.isnan(err) else f"{err * 100:.0f}%",
                help="How far scaled MODIS lands from VIIRS in the overlap months.",
            )

            v1, v2 = st.columns(2)
            v1.metric(
                "Out-of-sample error",
                "n/a" if np.isnan(info["err_loyo"]) else f"{info['err_loyo'] * 100:.0f}%",
                help="Leave-one-year-out: the factor is fitted on the other overlap years and "
                "used to predict each held-out year's VIIRS monthly counts.",
            )
            if trend_raw is not None and trend_h is not None:
                v2.metric(
                    "Trend without vs with scaling",
                    f"{trend_raw[0]:+.1f}% vs {trend_h[0]:+.1f}% / yr",
                    help="Same area and years. Without scaling, the jump from MODIS to VIIRS "
                    "shows up as a fake trend in the record.",
                )
            else:
                v2.metric(
                    "Trend without vs with scaling",
                    "n/a",
                    help="Needs at least five complete years in the selected range.",
                )

            ov = info["overlap"]
            ovv = ov[ov["valid"]] if len(ov) else ov
            if len(ovv):
                sc = go.Figure()
                sc.add_trace(
                    go.Scatter(
                        x=ovv["MODIS"],
                        y=ovv["VIIRS"],
                        mode="markers",
                        marker=dict(color=VIIRS_COLOR, size=7, opacity=0.82),
                        name="Overlap month",
                        text=ovv["ym"].dt.strftime("%b %Y"),
                        hovertemplate="%{text}<br>MODIS %{x:.0f}<br>VIIRS %{y:.0f}<extra></extra>",
                    )
                )
                xmax = float(ovv["MODIS"].max())
                sc.add_trace(
                    go.Scatter(
                        x=[0, xmax],
                        y=[0, xmax * info["factor"]],
                        mode="lines",
                        line=dict(color=SAGE_COLOR, dash="dash", width=2),
                        name=f"VIIRS = {info['factor']:.2f} × MODIS",
                    )
                )
                sc.update_xaxes(title="MODIS monthly detections (per-satellite)")
                sc.update_yaxes(title="VIIRS monthly detections")
                show(tidy(sc, "Overlap months (whole file): MODIS against VIIRS", 360), "overlap")

            rm = info["raw_monthly"]
            rm = rm[(rm["ym"] >= pd.Timestamp(d0).to_period("M").start_time) & (rm["ym"] <= pd.Timestamp(d1))]
            if aoi_poly is None:
                raw_year = (
                    rm.assign(year=rm["ym"].dt.year)
                    .groupby(["year", "sensor"], observed=True)["n"]
                    .sum()
                    .unstack(fill_value=0)
                )
                cmp_ = go.Figure()
                if "MODIS" in raw_year:
                    cmp_.add_scatter(
                        x=raw_year.index,
                        y=raw_year["MODIS"],
                        mode="lines+markers",
                        name="MODIS, raw",
                        line=dict(color=MODIS_COLOR, width=1.5, dash="dot"),
                        marker=dict(size=4),
                    )
                if "VIIRS" in raw_year:
                    cmp_.add_scatter(
                        x=raw_year.index,
                        y=raw_year["VIIRS"],
                        mode="lines+markers",
                        name="VIIRS, raw (all satellites)",
                        line=dict(color=VIIRS_COLOR, width=1.5, dash="dot"),
                        marker=dict(size=4),
                    )
                cmp_.add_scatter(
                    x=annual_total.index,
                    y=annual_total.to_numpy(),
                    mode="lines+markers",
                    name="Harmonized",
                    line=dict(color="#f4f0e6", width=2.8),
                    marker=dict(size=6),
                )
                cmp_.update_xaxes(dtick=2)
                cmp_.update_yaxes(title="detections / year")
                show(tidy(cmp_, "Before and after: the raw records against the harmonized series"), "beforeafter")
            else:
                st.caption("The raw-sensor comparison is shown for the whole region. Choose Whole region to see it.")

            by_plat = rm.groupby(["ym", "platform"], observed=True)["n"].sum().unstack(fill_value=0)
            if len(by_plat):
                pf = px.line(by_plat, labels=dict(ym="", value="detections", platform="Satellite"))
                show(tidy(pf, "Raw monthly detections by satellite, whole region (before any correction)", 320), "platforms")
                st.caption(
                    "Steps in this chart happen when a satellite joins the constellation, not when fire "
                    "changes. That is why the app normalizes per satellite and defaults to SNPP only."
                )

            cont = (
                info["raw_monthly"]
                .groupby(["ym", "sensor"], observed=True)["n"]
                .sum()
                .unstack(fill_value=0)
                .sort_index()
            )
            if len(cont):
                seam = go.Figure()
                for s, nm, clr in (
                    ("MODIS", "MODIS (raw)", MODIS_COLOR),
                    ("VIIRS", "VIIRS (raw)", VIIRS_COLOR),
                    ("UNKNOWN", "Unlabelled (raw)", OTHER_COLOR),
                ):
                    if s in cont.columns and cont[s].sum() > 0:
                        seam.add_scatter(
                            x=cont.index, y=cont[s].to_numpy(), mode="lines",
                            name=nm, line=dict(color=clr, width=1.8),
                        )
                if info["split"] is not None:
                    seam.add_vline(x=info["split"], line_dash="dash", line_color="#f4f0e6")
                seam.update_xaxes(title="Month")
                seam.update_yaxes(title="Raw detections / month")
                show(tidy(seam, "Sensor continuity: raw monthly detections across the whole record", 340), "seam")
                st.caption(
                    "Raw counts before harmonization, for the whole file. The dashed line marks "
                    f"{info['split']:%B %Y}, when VIIRS becomes the primary record and scaled MODIS "
                    "is spliced out. Steps in either line are constellation changes (satellites "
                    "joining), not changes in fire activity — which is why the harmonized series "
                    "normalizes per satellite."
                )

            if scaling.startswith("By"):
                tbl = pd.DataFrame({
                    "Month": [calendar.month_abbr[mo] for mo in range(1, 13)],
                    "MODIS to VIIRS factor": [round(info["seasonal_map"][mo], 2) for mo in range(1, 13)],
                })
                with st.expander("Monthly scaling factors"):
                    st.dataframe(tbl, hide_index=True)

            if aoi_poly is not None:
                st.caption(
                    "The scaling factor is fitted on the whole region file, not this drawn area — "
                    "the true MODIS/VIIRS ratio can differ regionally, so treat scaled values here "
                    "as approximate."
                )

        with st.expander("Methods and limitations"):
            st.markdown(
                "- **Counts are detections, not area or emissions.** One fire produces many "
                "detections; FRP is an intensity proxy, not a burned-area measurement.\n"
                "- **The scaling factor is regional.** It is fitted on the whole country file, "
                "not each drawn AOI, and can differ between land-cover types.\n"
                "- **MODIS misses small fires.** Scaling matches monthly totals but cannot "
                "recover the sub-pixel detail VIIRS sees after 2012.\n"
                "- **Orbit drift is only partly handled.** Calibration can be capped at a "
                "chosen year, but Terra/Aqua overpass drift since ~2020 is not fully corrected.\n"
                "- **Recent data may be NRT.** The tail of a fresh download can be "
                "near-real-time rather than standard reprocessed quality.\n"
                "- **Source classes are conservative.** 'Industrial / static' requires a cell "
                "to recur year-round on many days; everything else stays 'unresolved' because "
                "FIRMS alone cannot separate field burning from wildfire.\n"
                "- **Sensor steps are constellation changes,** not fire changes — see the raw "
                "continuity chart above."
            )

    else:
        key = (match, str(d0), str(d1), drop_low, veg_only, viirs_mode, scaling, int(min_monthly),
               round(path.stat().st_mtime, 1), aoi_key)
        mt = S["monthly_agg"].copy()
        mt["median_same_month"] = mt["month"].map(med_by_month)
        mt["complete_month"] = [bool(full.loc[y, mo]) for y, mo in zip(mt["year"], mt["month"])]
        mt["anomaly_pct"] = np.where(
            mt["complete_month"] & (mt["median_same_month"] > 0),
            (mt["detections_viirs_equiv"] - mt["median_same_month"]) / mt["median_same_month"] * 100,
            np.nan,
        )
        mt = mt.round({"detections_viirs_equiv": 1, "median_same_month": 1, "anomaly_pct": 1, "frp_mw": 0})
        flat = (
            pivot_full.rename_axis("year")
            .reset_index()
            .melt(id_vars="year", var_name="month", value_name="n")
            .dropna()
        )

        report = [
            f"# Fire activity summary: {region_text}",
            "",
            f"- Period: {d0} to {d1} (data through {dmax})",
            f"- Area: {'whole region bounding data' if aoi_poly is None else f'custom area, about {aoi_area:,.0f} km2'}",
            f"- Detections kept: {S['n_rows']:,} (VIIRS-equivalent total {S['w_sum']:,.0f})",
            *([f"- Total FRP: {S['frp_sum']:,.0f} MW (summed per-detection fire radiative power; intensity proxy, not burned area)"]
              if has_frp and S["frp_sum"] > 0 else []),
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
                f"{info['err_loyo'] * 100:.0f}%"
                if not np.isnan(info["corr"]) and not np.isnan(info["err_loyo"])
                else "- Overlap statistics unavailable for this file",
            ]
        if status is not None:
            report += [
                "",
                "## Current period",
                f"{calendar.month_name[lm]} 1-{last_day}, {ly}: {current:,.0f} detections against a median of "
                f"{status['median']:,.0f} for the same days in {len(hist)} earlier years "
                f"({ordinal(status['rank'])} highest of {status['n']}; {status['label']}).",
            ]
        if field_notes:
            report += ["", "## Field notes"] + [f"- **{n['tag']}** — {n['head']}: {n['body']}" for n in field_notes]
        if verdicts:
            report += ["", "## Year by year"] + [f"- {v['year']}: {v['headline']} ({v['short']})" for v in verdicts]
        if trend_h is not None:
            report += ["", f"Annual trend over {len(full_years)} complete years: {trend_h[0]:+.1f}% per year (Theil-Sen)."]
            if trend_viirs is not None:
                report += [f"VIIRS-era-only trend ({viirs_years[0]}+): {trend_viirs[0]:+.1f}% per year, independent of the MODIS scaling factor."]
        if not flat.empty:
            report += ["", "## Most active months"] + [
                f"- {calendar.month_abbr[int(r.month)]} {int(r.year)}: {r.n:,.0f}"
                for r in flat.nlargest(5, "n").itertuples()
            ]
        report += ["", "Source: NASA FIRMS active-fire detections (MODIS Terra/Aqua, VIIRS SNPP/NOAA-20/NOAA-21)."]
        report_txt = "\n".join(report)

        stem = slug(match, canon) + ("_aoi" if aoi_poly is not None else "")
        d1c, d2c, d3c = st.columns(3)
        d1c.download_button(
            "Hotspots (CSV)",
            to_csv(key, sub),
            file_name=f"{stem}_hotspots.csv",
            mime="text/csv",
            help="The harmonized (spliced) record for the current area and period — post-2012 MODIS rows are omitted when the VIIRS-only rule applies. Includes sensor, satellite and VIIRS-equivalent weight.",
        )
        d2c.download_button(
            "Monthly series with anomalies (CSV)",
            mt.to_csv(index=False).encode(),
            file_name=f"{stem}_monthly.csv",
            mime="text/csv",
        )
        d3c.download_button(
            "Field report (Markdown)",
            report_txt.encode(),
            file_name=f"{stem}_summary.md",
            mime="text/markdown",
        )
        if aoi_poly is not None:
            st.download_button(
                "Area of interest (GeoJSON)",
                json.dumps({"type": "Feature", "properties": {}, "geometry": mapping(aoi_poly)}).encode(),
                file_name=f"{stem}.geojson",
                mime="application/geo+json",
            )
        st.dataframe(mt, hide_index=True)

else:
    render_map(None, height=540)
    if match is None:
        loaded_keys = sorted(k for k in regions if has_data(k) and k not in ALIASES)
        if loaded_keys:
            st.markdown(
                '<div class="landing-caption">'
                "<span>Click any highlighted country on the map to open its archive. "
                "Inside any region, draw a polygon or enter coordinates to inspect a park, province, or watershed.</span>"
                "</div>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<p class="note" style="margin-top:.85rem">Search for any country in the sidebar, and place '
                "FIRMS archive CSV or Parquet files in <code>data/</code> to light them up on the map.</p>",
                unsafe_allow_html=True,
            )

st.markdown(
    '<div class="credits">Fire detections come from NASA FIRMS (MODIS and VIIRS). Country boundaries are from '
    "Natural Earth, and base maps are from Esri and OpenStreetMap contributors. "
    "Made for the 2026 NASA Space Apps Challenge.</div>",
    unsafe_allow_html=True,
)
