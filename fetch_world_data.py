"""Download the world fire-archive parquets into data/.

The parquets are distributed as release-asset zip volumes (too large for
git). Pass the base URL of the release, e.g.:

    python fetch_world_data.py "https://github.com/OWNER/REPO/releases/download/TAG"

Usage:
    python fetch_world_data.py "https://github.com/OWNER/REPO/releases/download/TAG"
    python fetch_world_data.py "https://github.com/OWNER/REPO/releases/download/TAG" --lite

`--lite` downloads only popular_countries.zip (~160 MB: Greece, Portugal,
Spain, Italy, France, Türkiye, South Africa, Kenya, Chile, Indonesia)
instead of the full ~4.5 GB world dataset.

Volumes are fetched one at a time and extracted straight into data/ —
no need to keep the zips around afterwards.
"""

import argparse
import sys
import urllib.request
import zipfile
from pathlib import Path

DATA = Path(__file__).with_name("data")
VOLUMES = [f"volume_{i:02d}.zip" for i in range(1, 5)]
LITE = ["popular_countries.zip"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("base_url", help="Release download base URL (no trailing slash needed)")
    ap.add_argument("--lite", action="store_true",
                    help="Download only the ~160 MB popular-countries subset")
    ap.add_argument("--keep", action="store_true", help="Keep downloaded zips")
    args = ap.parse_args()

    base = args.base_url.rstrip("/")
    DATA.mkdir(exist_ok=True)
    tmp = Path(__file__).with_name("_dl")
    tmp.mkdir(exist_ok=True)

    for name in (LITE if args.lite else VOLUMES):
        url = f"{base}/{name}"
        zp = tmp / name
        print(f"Downloading {name} …", flush=True)
        urllib.request.urlretrieve(url, zp)
        print(f"  {zp.stat().st_size / 1e9:.2f} GB — extracting", flush=True)
        with zipfile.ZipFile(zp) as z:
            z.extractall(DATA)
        if not args.keep:
            zp.unlink()
        print(f"  done ({len(list(DATA.glob('*.parquet')))} parquets in data/)", flush=True)

    print("All volumes installed.")


if __name__ == "__main__":
    sys.exit(main())
