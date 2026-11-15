"""Re-encode large per-country parquets with categorical string columns.

The builder stores satellite/instrument/confidence/type as arrow strings,
which pandas reads back as object dtype — ~5 GB of Python strings on the
biggest countries (Russia, Angola, Brazil, DRC). Categoricals cut that to
~100 MB and stop the app's RAM spike on huge files.
"""

from pathlib import Path

import pandas as pd

DATA = Path(__file__).with_name("data")
STR_COLS = ["satellite", "instrument", "confidence", "type"]
MIN_MB = 80


def main():
    files = [p for p in DATA.glob("*.parquet") if p.stat().st_size > MIN_MB * 1e6]
    print(f"{len(files)} files over {MIN_MB} MB")
    for p in sorted(files, key=lambda x: -x.stat().st_size):
        df = pd.read_parquet(p)
        if str(df["satellite"].dtype) == "category":
            print(f"{p.name}: already categorical, skipped", flush=True)
            del df
            continue
        for c in STR_COLS:
            if c in df:
                df[c] = df[c].astype("category")
        tmp = p.with_suffix(".tmp.parquet")
        df.to_parquet(tmp, index=False, compression="zstd")
        tmp.replace(p)
        print(f"{p.name}: {len(df):,} rows, {p.stat().st_size/1e6:.0f} MB", flush=True)
        del df


if __name__ == "__main__":
    main()
