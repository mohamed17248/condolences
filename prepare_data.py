"""Convert raw FIRMS CSV exports into the parquet files the app loads fastest.

Usage:
    python prepare_data.py path/to/modis_2000_Egypt.csv [more.csv ...]
    python prepare_data.py --dir data/raw          # every .csv in the folder

Each input `name.csv` produces `name.parquet` next to it (or in --out DIR).
Parquet keeps only the columns the app uses, with compact dtypes, so a
multi-million-row country loads in seconds instead of minutes.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

WANTED = {
    "latitude", "longitude", "acq_date", "acq_time", "satellite", "instrument",
    "confidence", "brightness", "bright_ti4", "type", "frp",
}
CATEGORICAL = ("satellite", "instrument", "confidence")


def convert(csv_path: Path, out_dir: Path | None = None, force: bool = False) -> Path | None:
    out = (out_dir or csv_path.parent) / (csv_path.stem + ".parquet")
    if out.exists() and not force:
        print(f"skip  {csv_path.name} -> {out.name} (exists)")
        return None

    df = pd.read_csv(csv_path, usecols=lambda c: c.strip().lower() in WANTED)
    df.columns = [c.strip().lower() for c in df.columns]
    df["acq_date"] = pd.to_datetime(df["acq_date"], errors="coerce")
    df = df.dropna(subset=["acq_date", "latitude", "longitude"])
    for c in CATEGORICAL:
        if c in df:
            df[c] = df[c].astype("category")

    out_dir = out_dir or csv_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False, compression="zstd")
    print(f"wrote {out}  ({len(df):,} rows, {out.stat().st_size / 1e6:.1f} MB)")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="*", type=Path, help="FIRMS CSV files to convert")
    ap.add_argument("--dir", type=Path, help="convert every .csv directly inside this folder")
    ap.add_argument("--out", type=Path, help="output folder (default: next to each CSV)")
    ap.add_argument("--force", action="store_true", help="overwrite existing parquet files")
    args = ap.parse_args()

    paths = list(args.files)
    if args.dir:
        paths += sorted(args.dir.glob("*.csv"))
    if not paths:
        ap.error("give CSV files or --dir")

    done = 0
    for p in paths:
        if not p.exists():
            print(f"missing {p}", file=sys.stderr)
            continue
        if convert(p, args.out, args.force):
            done += 1
    print(f"{done} file(s) converted")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
