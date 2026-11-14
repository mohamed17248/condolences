# Fire Activity Explorer

A web app that harmonizes NASA FIRMS active-fire detections from MODIS (Terra, Aqua; 2000 on)
and VIIRS (SNPP, NOAA-20, NOAA-21; 2012 on) into one consistent record, and presents it as a
burning-activity calendar for any area of interest. Built for the NASA Space Apps Challenge,
"Harmonization of MODIS and VIIRS Hot Spots".

## Quick start

```
pip install -r requirements.txt
streamlit run app.py
```

Required columns: `latitude`, `longitude`, `acq_date`. Used when present: `satellite`,
`instrument`, `confidence`, `type`, `acq_time`, `brightness` (MODIS), `bright_ti4` (VIIRS).

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
   sensor already in orbit on that date (Terra alone before Aqua, SNPP alone before NOAA-20),
   so counts do not jump when a satellite joins. Start dates are taken from the file itself.
   By default only VIIRS SNPP is used, the longest single-satellite VIIRS record.
3. **Scaling.** In the months where both sensors exist, compute VIIRS / MODIS monthly count
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
- Terra and Aqua orbit drift in recent years is not corrected.
- Satellite start dates come from the first detection in the file, which can be later than the
  real start in a quiet region.
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
