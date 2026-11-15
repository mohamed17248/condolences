"""Unit tests for the pure helpers in firecore.py — no Streamlit needed.

Run:  python -m pytest tests/   (or: python tests/test_firecore.py)
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from firecore import (  # noqa: E402
    level_for,
    normalize_platform,
    ordinal,
    rel_trend,
    short_text,
    size_text,
    status_for,
    theil_sen,
)


def test_theil_sen_recovers_line():
    x = np.array([2000, 2005, 2010, 2015, 2020], float)
    y = 3.0 * x - 6000.0  # slope 3, intercept -6000
    slope, icpt = theil_sen(x, y)
    assert abs(slope - 3.0) < 1e-9
    assert abs(icpt - (-6000.0)) < 1e-6


def test_theil_sen_medial_outlier():
    # One wild outlier should not dominate the median slope.
    x = np.arange(10, dtype=float)
    y = 2.0 * x + 5.0
    y[4] = 1e6
    slope, _ = theil_sen(x, y)
    assert abs(slope - 2.0) < 1e-9


def test_normalize_platform_strings():
    s = pd.Series(["N", "T", "A", "1", "2", "npp", "junk", " aqua "])
    out = normalize_platform(s).tolist()
    assert out == ["SNPP", "Terra", "Aqua", "NOAA-20", "NOAA-21", "SNPP", "UNKNOWN", "Aqua"]


def test_normalize_platform_categorical():
    s = pd.Series(["N", "T", "1"]).astype("category")
    out = normalize_platform(s)
    assert out.tolist() == ["SNPP", "Terra", "NOAA-20"]
    # categorical and plain-string paths must agree
    plain = normalize_platform(pd.Series(["N", "T", "1"]))
    assert [str(v) for v in out] == [str(v) for v in plain]


def test_status_for_buckets():
    hist = pd.Series([10.0] * 10 + [100.0])
    assert status_for(100.0, hist)["level"] == "high"
    hist2 = pd.Series([10.0, 20, 30, 40, 50, 60, 70, 80, 90, 100])
    assert status_for(95.0, hist2)["level"] == "elevated"
    assert status_for(50.0, hist2)["level"] == "normal"
    assert status_for(5.0, hist2)["level"] == "low"
    zero = status_for(0.0, pd.Series([0.0, 0.0, 0.0]))
    assert zero["level"] == "normal"


def test_level_for_thresholds():
    assert level_for(1.6)[0] == "much_more"
    assert level_for(1.3)[0] == "more"
    assert level_for(1.0)[0] == "typical"
    assert level_for(0.7)[0] == "less"
    assert level_for(0.2)[0] == "much_less"


def test_rel_trend_min_years():
    s = pd.Series({2020: 10.0, 2021: 12.0, 2022: 11.0, 2023: 13.0})
    assert rel_trend(s, [2020, 2021, 2022, 2023]) is None  # <5 years
    s2 = pd.Series({y: 10.0 + y - 2015 for y in range(2015, 2025)})
    res = rel_trend(s2, list(range(2015, 2025)))
    assert res is not None and res[0] > 0


def test_ordinal():
    assert ordinal(1) == "1st" and ordinal(2) == "2nd" and ordinal(3) == "3rd"
    assert ordinal(4) == "4th" and ordinal(11) == "11th" and ordinal(21) == "21st"
    assert ordinal(12) == "12th" and ordinal(23) == "23rd"


def test_size_text():
    assert size_text(2.5) == "2.5× the usual year"
    assert size_text(1.0) == "right at the usual year"
    assert size_text(None) == "with no baseline year to compare"
    assert short_text(1.6) == "+60% vs usual"


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"{len(fns)} tests passed")
