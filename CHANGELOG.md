# Changelog — Fire Activity Explorer

What changed between the original `fire-app-repo` version (Egypt-only prototype)
and the current `fire-app-final` / `nasa-space-app-mohamed-abbas` version.

## Data & harmonization

- **Worldwide coverage** — pipeline (`build_world_parquets.py`) turns the yearly
  all-country MODIS/VIIRS ZIP archives into one cleaned Parquet per country
  (~190 files, ~4.5 GB) without extracting the full archives. VIIRS is
  Suomi-NPP only; JPSS is excluded.
- **Compact storage** — string columns stored as categoricals; the 16 largest
  countries were re-encoded (`reencode_parquets.py`), cutting ~5.8 GB to ~886 MB
  and cutting Russia's load time from ~12 s to ~3.4 s.
- **Shard-based builder** — large countries are processed in bounded chunks to
  avoid the multi-GB concatenation that crashed on Russia/Angola.
- **FRP ingest** — `frp` (fire radiative power) is loaded, constellation-
  averaged, and surfaced as intensity metrics/charts. Deliberately *not* scaled
  by the MODIS→VIIRS factor (intensity ≠ detection count).
- **Documented satellite start dates** — per-satellite weights use launch dates
  (Terra 2000-11, Aqua 2002-07, SNPP 2012-01, NOAA-20 2018, NOAA-21 2023) instead
  of first-detection-in-file, which lagged in quiet regions.
- **Calibration cutoff** — new "Calibrate MODIS→VIIRS through year" control
  (default 2019) excludes recent overlap months affected by Terra/Aqua orbit
  drift.
- **Honesty checks** — VIIRS-era-only Theil-Sen trend shown beside the
  full-record trend; AOI warning that the scaling factor is regional; NRT tail
  flagged when the file ends within ~4 months of today.

## App structure & UX

- **Section navigator** — `st.tabs` replaced with a URL-backed section switcher
  (`?tab=`) so only the selected section executes; expensive sections (Timelapse,
  Predict) no longer rerun on every interaction.
- **World map landing** — clickable country overlay on Natural Earth boundaries;
  click a country to open its archive. Sidebar search + "← All countries" button
  to return.
- **Custom AOIs** — draw polygon/rectangle/circle, bounding box, or point+radius;
  all analytics, maps, and downloads respect the AOI.
- **Map fixes** — basemap tile wrapping disabled (`no_wrap`), viewport bounded to
  the world extent (`max_bounds`), soft viscosity so edges pull back instead of
  walling.
- **Click handling rewrite** — country selection now uses the fresh
  `last_object_clicked` coordinates gated by tooltip + click count, fixing
  stale-coordinate picks that chose different countries depending on zoom.
  Minimal `returned_objects` so pan/zoom doesn't rerun the script.
- **Memory fixes for giant countries** — points sampled *before* string
  conversion, density/change grids memoized per data fingerprint, country
  overlay GeoJSON cached, archive stats cached with spinners.

## Analytics added

- "Every year at a glance" annual anomaly stripes — now archive-scoped so they
  stay meaningful for 1–5 year windows (was all-grey for short ranges).
- Year-by-year verdicts, rank, and month-by-month comparison.
- Burning-activity calendar heatmap + anomaly-vs-median view.
- Cumulative year-over-year burn pacing curve.
- Period comparison modes: same dates previous year / median year same months.
- Anomaly alerts — top days running ≥2× the same day-of-year median.
- Sensor continuity chart around the 2012 MODIS→VIIRS splice.
- Fire-source classification (conservative industrial/static detection) and a
  recent-activity footprint in "Predict & sources".
- Downloads: hotspots CSV, monthly series CSV, Markdown field report, AOI GeoJSON.
- Field notes: regime heuristics (crop-residue now labelled a hypothesis),
  overpass-timing diagnostics, archive extremes, orbital handoff.

## Fixes from code review

- Year-by-year picker no longer crashes (`KeyError`) when the date slider
  excludes verdict years.
- 24-hour chart caption corrected — it shows satellite overpass timing, not
  real burning behaviour.
- `source_class_summary` uses an integer cell key instead of per-row strings.
- `observed=True` groupbys, standalone day-weighted note, computed archive span,
  `max_entries` caps on Streamlit caches.

## Tooling & docs

- `firecore.py` — pure helpers (`theil_sen`, `normalize_platform`, `status_for`,
  `level_for`, …) extracted for testability.
- `tests/test_firecore.py` — 8 unit tests (pytest or standalone runner).
- `prepare_data.py` — FIRMS CSV → compact parquet converter.
- `fetch_world_data.py` — downloads the ~4.5 GB world dataset from GitHub
  Release assets into `data/`.
- `data/` gitignored for the world parquets; `egypt.parquet` sample stays tracked.
- README rewritten: FIRMS download steps, release fetch, method, limitations.
- `.streamlit/config.toml` — dark/sage theme.

## Known limitations (unchanged by design)

- Detections ≠ burned area; FRP is a proxy, not a measurement.
- MODIS misses sub-pixel fires; pre-2012 detail can't be recovered by scaling.
- Source classification can't separate field burning from wildfire without a
  land-cover join.
- Launch dates are approximate to ~a month.


## Fixes applied

Year tab crash. The verdicts use the whole archive, but the chart only had the slider’s window. The year tab now reads the archive’s own monthly tables, so picking 2005 with the slider on 2018–2025 works. I confirmed that a narrow window still gives all 26 verdicts.
Confidence filter. It no longer crashes on a file where every row has the same confidence value. I tested this case ;p.
Missing satellite values. They are now labelled “UNKNOWN” instead of being dropped.
Source classifier. It now uses an integer cell id instead of building two strings for every row.
Memory. The unusual-days check no longer copies every column.
Region selector. index=None is back on the region selectbox.
Rating threshold. It is now four complete years everywhere, in one constant.
Tab name. “Predict & sources” is now “Recent activity & sources”.
A bug in the alert-days table. Every date showed 2021. It now shows the real year, for example “05 Apr 2011”.


## new features added and fixes

Daily burning calendar (Patterns > Calendar & clock). One cell per day for a chosen year, weeks across and weekdays down. Blank cells mean "outside the loaded record", never zero fire. The colour scale is capped at the 98th percentile so one extreme day does not wash out the season.
"How to read this page" panel on the Overview, explaining the stripes, the verdict, the map, harmonization and the downloads.
Compare tab upgrades
The open country is pre-selected as Country A and the one open before it fills Country B (both can still be changed).
Maps are now interactive and follow the base map chosen in the sidebar.
Monthly burning calendars for both countries, with a selectable colour ramp, an optional shared colour scale, and a seasonal-pattern chart.
Timelapse "Simple (no map tiles)" view. Draws the grid and the country outline on a plain dark canvas, so the timelapse works on any network.
Percentile in the verdict. The current-month verdict now also says how it ranks, for example "higher than 62% of those years".

@queriesonhold@gmail.com for more info.
