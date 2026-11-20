"""Download the world fire-archive parquets into data/.

The parquets are distributed as release-asset zip volumes (too large for
git). Pass the base URL of the release, e.g.:

    python fetch_world_data.py "https://github.com/OWNER/REPO/releases/download/TAG"

Usage:
    python fetch_world_data.py "https://github.com/OWNER/REPO/releases/download/TAG"
    python fetch_world_data.py "https://github.com/OWNER/REPO/releases/download/TAG" --lite

`--lite` downloads only popular_countries.zip (~160 MB) instead of the
full ~4.5 GB world dataset.

Downloads use 8 parallel HTTP range connections per volume and resume
interrupted parts, so re-running the command continues where it left
off instead of starting over. Volumes are extracted straight into data/
— no need to keep the zips around afterwards.
"""

import argparse
import concurrent.futures
import re
import sys
import urllib.request
import zipfile
from pathlib import Path

DATA = Path(__file__).with_name("data")
VOLUMES = [f"volume_{i:02d}.zip" for i in range(1, 5)]
LITE = ["popular_countries.zip"]
PARTS = 8
CHUNK = 1 << 20  # 1 MiB


def _req(url, headers=None):
    return urllib.request.Request(url, headers=headers or {})


def _open(url, headers=None):
    return urllib.request.urlopen(_req(url, headers), timeout=60)


def probe_size(url):
    """Return (total_size, supports_ranges) for a release asset URL."""
    try:
        with _open(url, {"Range": "bytes=0-0"}) as r:
            cr = r.headers.get("Content-Range", "")
            m = re.search(r"/(\d+)$", cr)
            if m:
                return int(m.group(1)), True
            cl = r.headers.get("Content-Length")
            return (int(cl) if cl else None), False
    except Exception:
        return None, False


def fetch_part(url, part_path, start, end, have=0):
    """Download bytes [start+have, end] of url, appending to part_path."""
    if have >= end - start + 1:
        return end - start + 1
    rng = f"bytes={start + have}-{end}"
    got = have
    with _open(url, {"Range": rng}) as r, open(part_path, "ab") as f:
        while True:
            buf = r.read(CHUNK)
            if not buf:
                break
            f.write(buf)
            got += len(buf)
    return got


def download(url, dest: Path):
    """Download url to dest with parallel ranged parts + resume."""
    total, ranged = probe_size(url)
    tmp = dest.with_suffix(dest.suffix + ".parts")
    tmp.mkdir(exist_ok=True)

    if not ranged or not total or total < 64 << 20:
        # Small file or no range support: single stream with resume.
        have = dest.stat().st_size if dest.exists() else 0
        if have >= (total or 0) and total:
            print("  already complete")
            return
        print(f"  single-stream download ({total / 1e9:.2f} GB)" if total else "  downloading…")
        headers = {"Range": f"bytes={have}-"} if have else {}
        mode = "ab" if have else "wb"
        with _open(url, headers) as r, open(dest, mode) as f:
            while True:
                buf = r.read(CHUNK)
                if not buf:
                    break
                f.write(buf)
        return

    n = PARTS
    bounds = []
    step = (total + n - 1) // n
    for i in range(n):
        s = i * step
        e = min(total - 1, s + step - 1)
        if s <= e:
            bounds.append((i, s, e))

    def job(b):
        i, s, e = b
        pp = tmp / f"part{i:02d}"
        have = pp.stat().st_size if pp.exists() else 0
        want = e - s + 1
        if have < want:
            fetch_part(url, pp, s, e, have)
        if pp.stat().st_size != want:
            raise IOError(f"part {i} incomplete ({pp.stat().st_size}/{want} bytes) — re-run to resume")
        return i, want

    print(f"  {total / 1e9:.2f} GB via {len(bounds)} connections")
    with concurrent.futures.ThreadPoolExecutor(len(bounds)) as ex:
        for i, want in ex.map(job, bounds):
            print(f"    part {i + 1}/{len(bounds)} done", flush=True)

    # Concatenate parts into dest, then clean up.
    with open(dest, "wb") as out:
        for i, _, _ in bounds:
            with open(tmp / f"part{i:02d}", "rb") as p:
                while True:
                    buf = p.read(CHUNK)
                    if not buf:
                        break
                    out.write(buf)
    for p in tmp.iterdir():
        p.unlink()
    tmp.rmdir()


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
        download(url, zp)
        print(f"  {zp.stat().st_size / 1e9:.2f} GB — extracting", flush=True)
        with zipfile.ZipFile(zp) as z:
            z.extractall(DATA)
        if not args.keep:
            zp.unlink()
        print(f"  done ({len(list(DATA.glob('*.parquet')))} parquets in data/)", flush=True)

    print("All volumes installed.")


if __name__ == "__main__":
    sys.exit(main())
