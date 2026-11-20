# PROJECT The Burn Record

> See [CHANGELOG.md](CHANGELOG.md) for everything added, changed, and fixed
> relative to the original prototype.

A web app that harmonizes NASA FIRMS active-fire detections from MODIS (Terra, Aqua; 2000 on)
and VIIRS (SNPP, NOAA-20, NOAA-21; 2012 on) into one consistent record, and presents it as a
burning-activity calendar for any area of interest. Built for the NASA Space Apps Challenge,
"Harmonization of MODIS and VIIRS Hot Spots".

## Quick start

```
pip install -r requirements.txt
streamlit run app.py
```

A sample dataset (`data/egypt.parquet`, MODIS + VIIRS SNPP, 2000–present) is included,
so the app opens Egypt immediately. To add more countries, see "Getting data" below.

Required columns: `latitude`, `longitude`, `acq_date`. Used when present: `satellite`,
`instrument`, `confidence`, `type`, `acq_time`, `frp`, `brightness` (MODIS), `bright_ti4` (VIIRS).

## Getting data

**Option A — the prepared world dataset (fastest).** Country parquets for ~190 countries
(~4.5 GB) are published as release assets. After cloning:

```
python fetch_world_data.py "https://github.com/mohamed17248/condolences/releases/download/Data"
```

This downloads and extracts the `volume_*.zip` assets into `data/`.

**Option B — build your own from FIRMS.** Download country/year CSVs from
[firms.modaps.eosdis.nasa.gov/download](https://firms.modaps.eosdis.nasa.gov/download/)
(MODIS `modis_YYYY_<Country>.csv`, VIIRS `viirs-snpp_YYYY_<Country>.csv`; JPSS files are
not needed — the app defaults to SNPP only). Convert them once:

```
python prepare_data.py --dir raw_downloads --out data
```

Each `name.csv` becomes `name.parquet` — lowercase the country name with underscores
(`united_states_of_america.parquet`) so the app's country picker can find it. The app
also reads `.csv` directly, but parquet is far faster on multi-million-row countries.

**Option C — drop a file in `data/`.** Any FIRMS CSV or parquet named after a country
works with no conversion step.

## What it does

- Pick a region, then optionally narrow to an area of interest by drawing a polygon, rectangle or
  circle, or by typing a bounding box or a point and radius.
- Burning-activity calendar (year by month) with colour-blind-safe ramps, plus an anomaly view
  showing each month's deviation from that calendar month's median.
- Current month against the historical median for the same days, with rank and a plain-language
  status, so an unusual period is visible at a glance.
- Seasonal cycle, annual totals with a robust trend, most active months and years.
- Map layers: individual hotspots, density grid, and change against an earlier baseline.
- Downloads: hotspots (CSV), monthly series with anomalies (CSV), a Markdown summary, and the
  area of interest (GeoJSON).

## Harmonization method

1. **Filter.** Drop low-confidence detections (MODIS confidence below 30, VIIRS "low") and
   non-vegetation detections (FIRMS type other than 0, which removes volcanoes and gas flares).
   Both are optional.
2. **Per-satellite normalization.** Each detection is divided by the number of satellites of its
   sensor on orbit on that date, using documented launch dates (Terra Nov 2000, Aqua Jul 2002,
   SNPP Jan 2012, NOAA-20 2018, NOAA-21 2023) rather than the file's first detection, which can
   lag the real start in quiet regions. By default only VIIRS SNPP is used, the longest single-
   satellite VIIRS record.
3. **Scaling.** In the months where both sensors exist — up to a configurable cutoff year
   (default 2019, to limit Terra/Aqua orbit-drift bias) — compute VIIRS / MODIS monthly count
   ratios. The median is the scaling factor (a per-calendar-month version is available).
   MODIS counts are multiplied by it to give VIIRS-equivalent detections.
4. **Splice.** Scaled MODIS is used before the first full VIIRS month and VIIRS after it, so no
   month is counted twice.

### Validation shown in the app

- Bootstrap range of the factor over the overlap months.
- Correlation and median error between scaled MODIS and VIIRS in the overlap.
- Leave-one-year-out error: the factor is fitted without one year and used to predict that
  year's VIIRS counts.
- Trend with and without scaling on the same years, which shows how much of an apparent trend
  is caused by the sensor change alone.

## Limitations

- Counts are detections, not burned area or emissions, and one fire can produce many detections.
- The scaling factor is calibrated on the whole regional file, not each drawn area, to keep it
  stable for small areas. It can differ between land-cover types, so very small areas carry
  more uncertainty.
- MODIS has coarser pixels and misses small fires that VIIRS sees. Scaling matches monthly
  totals but cannot recover the missing detail before 2012.
- Terra and Aqua orbit drift in recent years is only partly handled — the calibration
  cutoff year excludes late overlap months, but the drift itself is not corrected.
- The most recent weeks of a fresh download may be near-real-time (NRT) rather than
  standard reprocessed quality; the app flags this when the file tail is recent.
- Satellite start dates use documented launch dates; JPSS VIIRS (NOAA-20/21) is excluded
  by default but can be included with per-satellite normalization.
- Months at the edges of the selected period are partial and are excluded from anomaly and
  climatology statistics.

## Data and credits

- Active-fire detections: NASA FIRMS (LANCE / Earthdata), MODIS Collection 6.1 and VIIRS 375 m.
- Country boundaries: Natural Earth.
- Base maps: OpenStreetMap contributors and Esri.
- Built with Streamlit, pandas, Plotly, folium, GeoPandas and Shapely.


## Contributors

- queriesonhold/egobreaking (Ahmed sultan)
- mohamed17248 (Mohamed abbas)
