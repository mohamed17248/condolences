"""Build per-country parquet files for the Streamlit app from FIRMS year zips.

Reads members straight out of the zips (no full extraction), keeps only the
columns the app uses, dedupes like the app loader, and writes
`data/<slug>.parquet` (zstd) per country.

Usage:
    python build_world_parquets.py                 # everything
    python build_world_parquets.py --countries Egypt Brazil --limit-zips 2
"""

import argparse
import ast
import io
import re
import sys
import time
import unicodedata
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

APP_PY = Path(__file__).with_name("app.py")
DATA_DIR = Path(__file__).with_name("data")
GEOJSON = Path(__file__).with_name("ne_countries.geojson")

MODIS_DIR = Path(r"E:\Nasa_Resources\Modis")
VIIRS_DIR = Path(r"E:\Nasa_Resources\Viirs")

# Columns the app actually reads (mirrors WANTED in app.py)
KEEP_COLS = {
    "latitude", "longitude", "acq_date", "acq_time", "satellite", "instrument",
    "confidence", "type", "frp", "brightness", "bright_ti4",
}

DEDUP_SUBSET = ["latitude", "longitude", "acq_date", "acq_time", "satellite"]

# Fixed output schema so yearly shards with different source columns
# (MODIS has brightness, VIIRS has bright_ti4, some years lack frp/type)
# still merge into one parquet.
CANON_COLS = [
    "latitude", "longitude", "acq_date", "acq_time", "satellite", "instrument",
    "confidence", "type", "frp", "brightness", "bright_ti4",
]


def arrow_schema():
    import pyarrow as pa
    return pa.schema([
        ("latitude", pa.float32()), ("longitude", pa.float32()),
        ("acq_date", pa.timestamp("ns")), ("acq_time", pa.int32()),
        ("satellite", pa.string()), ("instrument", pa.string()),
        ("confidence", pa.string()), ("type", pa.string()),
        ("frp", pa.float32()), ("brightness", pa.float32()),
        ("bright_ti4", pa.float32()),
    ])

MEMBER_RE = re.compile(r"^(?:modis|viirs-snpp)_\d{4}_(?P<country>.+)\.csv$", re.I)


# ---------------------------------------------------------------- app tables --

def _literal(name: str):
    """Pull a literal dict out of app.py without executing the Streamlit app."""
    tree = ast.parse(APP_PY.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name) and t.id == name:
                    return ast.literal_eval(node.value)
    return {}


def norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()
    s = s.lower().replace("&", " and ")
    return re.sub(r"[^a-z0-9]+", " ", s).strip()


def slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", norm(name)).strip("_")


def build_region_map():
    """Return resolve(key) -> canonical app region name for a normalized
    FIRMS country token. Resolution order: Natural Earth names, then
    manual/override regions, then ALIASES (which point at geojson names)."""
    import json
    geo = json.loads(GEOJSON.read_text(encoding="utf-8"))
    geo_names = sorted({(f["properties"].get("NAME") or "") for f in geo["features"]} - {""})

    aliases = _literal("ALIASES")
    manual = _literal("MANUAL_REGIONS")
    overrides = _literal("OVERRIDES")

    geo_map = {norm(n): n for n in geo_names}
    extra_map = {norm(n): n for n in list(manual) + list(overrides)}

    def resolve_alias(alias, target):
        t = norm(target)
        if t in geo_map:
            return geo_map[t]
        if t in extra_map:
            return extra_map[t]
        return target

    alias_map = {norm(a): resolve_alias(a, t) for a, t in aliases.items()}

    def resolve(key: str):
        return geo_map.get(key) or extra_map.get(key) or alias_map.get(key)

    return resolve, geo_names


# -------------------------------------------------------------------- zips --

def open_zips():
    zips = []
    for d in (MODIS_DIR, VIIRS_DIR):
        for p in sorted(d.glob("*_all_countries.zip")):
            zips.append((p, zipfile.ZipFile(p)))
    return zips


def index_members(zips):
    """country token -> list of (zip_path, member_name) in chronological order."""
    by_country = {}
    for path, zf in zips:
        for info in zf.infolist():
            if info.is_dir():
                continue
            m = MEMBER_RE.match(Path(info.filename).name)
            if m:
                by_country.setdefault(m.group("country"), []).append((path, info))
    return by_country


# ------------------------------------------------------------------ convert --

def clean_frame(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = [c.strip().lower() for c in df.columns]
    df = df.reindex(columns=CANON_COLS)
    if df.empty:
        return df
    df["acq_date"] = pd.to_datetime(df["acq_date"], errors="coerce")
    df = df.dropna(subset=["acq_date", "latitude", "longitude"])
    for c in ("latitude", "longitude", "frp", "brightness", "bright_ti4"):
        if c in df:
            df[c] = pd.to_numeric(df[c], errors="coerce").astype("float32")
    if "acq_time" in df:
        df["acq_time"] = pd.to_numeric(df["acq_time"], errors="coerce").astype("Int32")
    for c in ("satellite", "instrument", "confidence", "type"):
        df[c] = df[c].astype("string")
    sub = [c for c in DEDUP_SUBSET if c in df]
    df = df.drop_duplicates(subset=sub)
    return df.reset_index(drop=True)


def build_country(token: str, members, zips_by_path, out_path: Path, shard_dir: Path):
    """Clean each yearly CSV separately and spill to shard parquets so peak
    memory stays at one year-file, then stream-merge shards into the output."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    shard_dir.mkdir(parents=True, exist_ok=True)
    shards = []
    for i, (path, info) in enumerate(members):
        zf = zips_by_path[path]
        with zf.open(info) as fh:
            raw = pd.read_csv(io.TextIOWrapper(fh, encoding="utf-8"))
        df = clean_frame(raw)
        del raw
        if df.empty:
            continue
        sp = shard_dir / f"part_{i:03d}.parquet"
        df.to_parquet(sp, index=False, compression="zstd")
        shards.append(sp)
        del df
    if not shards:
        return 0

    schema = arrow_schema()
    writer = None
    total = 0
    try:
        for sp in shards:
            part = pd.read_parquet(sp)
            total += len(part)
            table = pa.Table.from_pandas(part, preserve_index=False).cast(schema)
            if writer is None:
                writer = pq.ParquetWriter(out_path, schema, compression="zstd")
            writer.write_table(table)
            del part, table
        writer.close()
    except Exception:
        if writer is not None:
            try:
                writer.close()
            except Exception:
                pass
        out_path.unlink(missing_ok=True)
        raise
    finally:
        if writer is not None:
            try:
                writer.close()
            except Exception:
                pass
        for sp in shards:
            sp.unlink(missing_ok=True)
        try:
            shard_dir.rmdir()
        except OSError:
            pass
    return total


# ---------------------------------------------------------------------- main --

def main():
    global MODIS_DIR, VIIRS_DIR, DATA_DIR
    ap = argparse.ArgumentParser()
    ap.add_argument("--countries", nargs="*", default=None,
                    help="FIRMS country tokens to build (default: all)")
    ap.add_argument("--limit-zips", type=int, default=None,
                    help="Only read the first N zips (for testing)")
    ap.add_argument("--src-modis", default=str(MODIS_DIR))
    ap.add_argument("--src-viirs", default=str(VIIRS_DIR))
    ap.add_argument("--out", default=str(DATA_DIR))
    args = ap.parse_args()

    MODIS_DIR, VIIRS_DIR, DATA_DIR = Path(args.src_modis), Path(args.src_viirs), Path(args.out)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    print("Opening zips…")
    zips = open_zips()
    if args.limit_zips:
        zips = zips[: args.limit_zips]
    zips_by_path = {p: zf for p, zf in zips}
    print(f"  {len(zips)} zips open")

    print("Indexing members…")
    by_country = index_members(zips)
    print(f"  {len(by_country)} country tokens")

    resolve, geo_names = build_region_map()

    tokens = sorted(by_country)
    if args.countries:
        wanted = {norm(c).replace(" ", "_") for c in args.countries}
        wanted |= {c for c in args.countries}
        tokens = [t for t in tokens if t in wanted or norm(t) in {norm(c) for c in args.countries}]

    unmatched, done, skipped = [], [], []
    t0 = time.time()
    for i, token in enumerate(tokens, 1):
        key = norm(token.replace("_", " "))
        region = resolve(key)
        if region is None:
            unmatched.append(token)
            continue
        out_path = DATA_DIR / f"{slug(region)}.parquet"
        if out_path.exists():
            skipped.append(token)
            continue
        n = build_country(token, by_country[token], zips_by_path, out_path,
                          DATA_DIR / "_shards" / slug(region))
        done.append((token, region, n))
        print(f"[{i}/{len(tokens)}] {token} -> {region}: {n:,} rows "
              f"({time.time() - t0:.0f}s)", flush=True)

    for zf in (zf for _, zf in zips):
        zf.close()

    print("\n=== Summary ===")
    print(f"built: {len(done)}  skipped(existing): {len(skipped)}  unmatched: {len(unmatched)}")
    if unmatched:
        print("Unmatched FIRMS names (no app region — add an alias or region):")
        for t in unmatched:
            print("  ", t)
    total_mb = sum(p.stat().st_size for p in DATA_DIR.glob("*.parquet")) / 1e6
    print(f"data/ total: {total_mb:,.0f} MB")


if __name__ == "__main__":
    sys.exit(main())
