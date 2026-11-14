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
SENSOR_COLORS = {"MODIS": MODIS_COLOR, "VIIRS": VIIRS_COLOR, "UNKNOWN": OTHER_COLOR}

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
    )
    fig.update_xaxes(showgrid=False, automargin=True)
    fig.update_yaxes(gridcolor="rgba(128,128,128,.18)", zeroline=False, automargin=True)
    return fig


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
            fill=True, fill_color=col, fill_opacity=0.55, tooltip=f"{d} · {sn}",
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
.block-container {padding-top: 2.2rem; max-width: 1500px;}
footer {visibility: hidden;}
.page-head h1 {font-size: 1.85rem; font-weight: 650; letter-spacing: -0.02em; margin: 0; padding: 0;}
.page-head p {color: rgba(128,128,128,.95); margin: .2rem 0 1.3rem 0; font-size: .95rem;}
.kicker {font-size: .7rem; letter-spacing: .09em; text-transform: uppercase;
         color: rgba(128,128,128,.95); margin: .9rem 0 .25rem 0;}
[data-testid="stMetric"] {border: 1px solid rgba(128,128,128,.25); border-radius: 6px; padding: .75rem .95rem;}
[data-testid="stMetricLabel"] p {font-size: .78rem; color: rgba(128,128,128,.95);}
.note {font-size: .85rem; color: rgba(128,128,128,.95); line-height: 1.5;}
.callout {border-left: 4px solid; padding: .7rem 1rem; border-radius: 4px;
          background: rgba(128,128,128,.08); margin: .3rem 0 1rem 0; font-size: .95rem; line-height: 1.5;}
.callout.low, .callout.normal {border-color: #4f8fc0;}
.callout.elevated {border-color: #e0a030;}
.callout.high {border-color: #ef6c33;}
</style>
"""

# --------------------------------------------------------------------------
# Page
# --------------------------------------------------------------------------
st.markdown(PAGE_CSS, unsafe_allow_html=True)
st.markdown(
    '<div class="page-head"><h1>Fire Activity Explorer</h1>'
    "<p>A burning-activity calendar from MODIS (2000 on) and VIIRS (2012 on) hotspots, "
    "harmonized into one consistent record for any area of interest.</p></div>",
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
    return key.title() + ("   ·   data" if has_data(key) else "")


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
style = sb.radio("Base map", list(TILES), horizontal=True, label_visibility="collapsed")
sb.markdown('<div class="kicker">Processing</div>', unsafe_allow_html=True)
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
    sb.caption(f"Area of interest: ~{aoi_area:,.0f} km² · "
               f"{miny:.2f}° to {maxy:.2f}° N, {minx:.2f}° to {maxx:.2f}° E")

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

    region_text = match_title + (" (selected area)" if aoi_poly is not None else "")
    if dmax is not None:
        st.caption(f"{region_text} · data through {dmax:%d %b %Y}")

    if status is not None:
        ratio = f"{current / status['median']:.1f}× the median" if status["median"] > 0 else "no median to compare with"
        callout_txt = (
            f"<b>{calendar.month_name[lm]} 1–{last_day}, {ly}:</b> {fmt(current)} detections "
            f"(VIIRS-equivalent), {ratio} of {fmt(status['median'])} for the same days in the "
            f"{len(hist)} earlier years. That is the {ordinal(status['rank'])} highest of "
            f"{status['n']} years, <b>{status['label']}</b>."
        )
        st.markdown(f'<div class="callout {status["level"]}">{callout_txt}</div>', unsafe_allow_html=True)

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Hotspots detected", fmt(len(sub)),
              help="Individual detections kept after the quality filters and the MODIS/VIIRS splice.")
    k2.metric("Harmonized total", fmt(sub["weight"].sum()),
              help="VIIRS-equivalent detections: per-satellite normalized, MODIS scaled.")
    clim = med_by_month
    k3.metric("Peak month", calendar.month_name[int(clim.idxmax())] if clim.notna().any() else "n/a",
              help="Calendar month with the highest median across complete months.")
    label = f"{calendar.month_abbr[lm]} {ly}, days 1–{last_day}"
    if status is not None:
        k4.metric(label, fmt(current),
                  delta=f"{current - status['median']:+,.0f} vs median", delta_color="inverse",
                  help=f"Median of the same days in earlier years: {fmt(status['median'])}.")
    else:
        k4.metric(label, fmt(current))
    if trend_h is not None:
        k5.metric("Annual trend", f"{trend_h[0]:+.1f}% / yr",
                  help=f"Theil-Sen slope over {len(full_years)} complete years, relative to the median year.")
    else:
        k5.metric("Annual trend", "n/a", help="Needs at least five complete years in the selected range.")

    if info["has_both"]:
        ci = info["factor_ci"]
        ci_txt = f" (range {ci[0]:.2f}–{ci[1]:.2f})" if ci else ""
        st.caption(
            f"MODIS scaled ×{info['factor']:.2f}{ci_txt} to VIIRS-equivalent from {info['n_valid']} "
            f"overlap months, calibrated on the whole file. {fmt(info['n_final'])} of "
            f"{fmt(info['n_raw'])} detections kept."
        )

# ---- Map ------------------------------------------------------------------
draw_mode = bool(match) and aoi_mode == "Draw on map"
m = base_map(style)
if sub is not None:
    if layer == "Points":
        n_all, n_shown = add_points(m, sub)
        if n_shown < n_all:
            st.caption(f"Showing {n_shown:,} of {n_all:,} detections, sampled evenly per sensor. "
                       "Switch to Density grid to see the full counts.")
    elif layer == "Density grid":
        cell, maxc = add_density(m, sub)
        st.caption(f"Grid cell about {cell:g}° (~{int(cell * 111)} km). "
                   f"Densest cell: {maxc:,} VIIRS-equivalent detections.")
    else:
        if len(full_years) >= 8:
            recent, base = full_years[-5:], full_years[:-5]
            cell, lim = add_change(m, sub, recent, base)
            st.caption(
                f"Mean annual detections per cell, {recent[0]}–{recent[-1]} compared with "
                f"{base[0]}–{base[-1]}. Orange = more fire than the baseline, purple = less "
                f"(strongest change about {lim:,.1f} per year per cell)."
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

aoi_key = hashlib.md5(aoi_poly.wkt.encode()).hexdigest()[:8] if aoi_poly is not None else "none"
result = st_folium(
    m, height=560, use_container_width=True,
    returned_objects=["all_drawings"] if (draw_mode and data_ready) else [],
    key=f"map-{match}-{layer}-{style}-{aoi_key}",
)
if draw_mode and data_ready and result and result.get("all_drawings"):
    poly = drawing_to_polygon(result["all_drawings"][-1])
    if poly is not None:
        saved = st.session_state.get("aoi_draw")
        if not saved or saved["region"] != match or saved["wkt"] != poly.wkt:
            st.session_state["aoi_draw"] = {"region": match, "wkt": poly.wkt}
            st.rerun()

if match is None:
    avail = sorted(s.replace("_", " ").title() for s in stems)
    st.markdown(
        '<p class="note">Pick a region in the sidebar. Regions marked "data" have a hotspot file '
        "in the data folder"
        + (f" (found: {', '.join(avail)})." if avail else " (none found yet).")
        + " Then narrow to an area of interest by drawing on the map or typing coordinates.</p>",
        unsafe_allow_html=True,
    )

# ---- Tabs -----------------------------------------------------------------
if sub is not None:
    t_cal, t_trend, t_harm, t_data = st.tabs(["Calendar", "Trends", "Harmonization", "Data"])

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
