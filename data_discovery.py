import duckdb
import pandas as pd
import numpy as np

DB_PATH = "gcoos.duckdb"
CSV_PATH = "data/DO.csv"
START = "2004-11-01"
END = "2009-11-01"
WINDOW_MONTHS = 60
CELL_SIZES = [5000, 10000, 15000, 25000]
MIN_MONTHS_REQUIRED = 24
CELL_SIZE_M = 15000

con = duckdb.connect(DB_PATH)

print(con.execute("""
SELECT 'base_clean' AS table_name, COUNT(*) AS n FROM base_clean
UNION ALL
SELECT 'fixed_stations', COUNT(*) FROM fixed_stations
UNION ALL
SELECT 'clean_surface_all', COUNT(*) FROM clean_surface_all
UNION ALL
SELECT 'clean_surface_region', COUNT(*) FROM clean_surface_region
""").fetchdf().to_string(index=False))

print(con.execute(f"""
SELECT
  COUNT(*) AS n_rows,
  SUM(dissolved_oxygen IS NULL) AS do_null,
  SUM(latitude IS NULL OR longitude IS NULL) AS latlon_null,
  SUM(activity_depth_height IS NULL) AS depth_null,
  SUM(activity_start_date IS NULL) AS dt_null,
  MIN(activity_start_date) AS tmin,
  MAX(activity_start_date) AS tmax
FROM read_csv_auto('{CSV_PATH}')
""").fetchdf().to_string(index=False))

print(con.execute("""
SELECT
  APPROX_QUANTILE(n_months, 0.50) AS p50,
  APPROX_QUANTILE(n_months, 0.75) AS p75,
  APPROX_QUANTILE(n_months, 0.90) AS p90,
  APPROX_QUANTILE(n_months, 0.95) AS p95,
  APPROX_QUANTILE(n_months, 0.99) AS p99,
  MIN(n_months) AS min_months,
  MAX(n_months) AS max_months,
  COUNT(*) AS n_stations
FROM (
  SELECT station_id, COUNT(DISTINCT DATE_TRUNC('month', dt)) AS n_months
  FROM clean_surface_region
  GROUP BY 1
) t
""").fetchdf().to_string(index=False))

print(con.execute("""
SELECT
  MIN(n_stations) AS min_stations_per_month,
  APPROX_QUANTILE(n_stations, 0.25) AS p25,
  APPROX_QUANTILE(n_stations, 0.50) AS p50,
  APPROX_QUANTILE(n_stations, 0.75) AS p75,
  MAX(n_stations) AS max_stations_per_month
FROM (
  SELECT DATE_TRUNC('month', dt) AS month,
         COUNT(DISTINCT station_id) AS n_stations
  FROM clean_surface_region
  GROUP BY 1
) t
""").fetchdf().to_string(index=False))

per_month = con.execute("""
WITH cell_month AS (
  SELECT
    FLOOR(x_m / 10000) AS cx,
    FLOOR(y_m / 10000) AS cy,
    DATE_TRUNC('month', dt) AS month
  FROM clean_surface_region
  GROUP BY 1,2,3
)
SELECT month, COUNT(DISTINCT (cx,cy)) AS n_cells
FROM cell_month
GROUP BY 1
ORDER BY 1
""").fetchdf()

per_month["month"] = pd.to_datetime(per_month["month"])
vals = per_month["n_cells"].to_numpy()
months = per_month["month"].to_numpy()

rows = []
for i in range(len(per_month) - WINDOW_MONTHS + 1):
    seg = vals[i:i + WINDOW_MONTHS]
    rows.append({
        "start": months[i],
        "end": months[i + WINDOW_MONTHS - 1],
        "min_cells": int(seg.min()),
        "p25_cells": float(np.quantile(seg, 0.25)),
        "median_cells": float(np.median(seg)),
        "mean_cells": float(seg.mean())
    })

candidate = pd.DataFrame(rows).sort_values(["median_cells", "p25_cells", "min_cells"], ascending=False)
print(candidate.head(10).to_string(index=False))

for cs in CELL_SIZES:
    print(con.execute(f"""
    WITH cell_month AS (
      SELECT
        FLOOR(x_m / {cs}) AS cx,
        FLOOR(y_m / {cs}) AS cy,
        DATE_TRUNC('month', dt) AS month
      FROM clean_surface_region
      WHERE dt >= '{START}' AND dt < '{END}'
      GROUP BY 1,2,3
    ),
    dims AS (
      SELECT
        COUNT(DISTINCT (cx,cy)) AS n_cells,
        COUNT(DISTINCT month) AS n_months,
        COUNT(*) AS n_filled
      FROM cell_month
    ),
    per_month AS (
      SELECT month, COUNT(DISTINCT (cx,cy)) AS n
      FROM cell_month
      GROUP BY 1
    ),
    per_cell AS (
      SELECT (cx,cy) AS cell_id, COUNT(*) AS n
      FROM cell_month
      GROUP BY 1
    )
    SELECT
      {cs} AS cell_size_m,
      (SELECT n_cells FROM dims) AS n_cells,
      (SELECT n_months FROM dims) AS n_months,
      (SELECT n_filled FROM dims) AS n_filled,
      (SELECT n_filled / (n_cells*n_months)::DOUBLE FROM dims) AS density,
      (SELECT MIN(n) FROM per_month) AS min_cells_per_month,
      (SELECT APPROX_QUANTILE(n, 0.25) FROM per_month) AS p25_cells_per_month,
      (SELECT APPROX_QUANTILE(n, 0.50) FROM per_month) AS med_cells_per_month,
      (SELECT APPROX_QUANTILE(n, 0.50) FROM per_cell) AS med_months_per_cell,
      (SELECT APPROX_QUANTILE(n, 0.75) FROM per_cell) AS p75_months_per_cell,
      (SELECT APPROX_QUANTILE(n, 0.90) FROM per_cell) AS p90_months_per_cell
    """).fetchdf().to_string(index=False))

print(con.execute(f"""
WITH cell_month AS (
  SELECT
    FLOOR(x_m / {CELL_SIZE_M}) AS cx,
    FLOOR(y_m / {CELL_SIZE_M}) AS cy,
    DATE_TRUNC('month', dt) AS month
  FROM clean_surface_region
  GROUP BY 1,2,3
)
SELECT month, COUNT(*) AS n_cells
FROM cell_month
GROUP BY 1
ORDER BY n_cells DESC
LIMIT 12
""").fetchdf().to_string(index=False))

cell = con.execute(f"""
SELECT
  FLOOR(x_m / 15000) AS cx,
  FLOOR(y_m / 15000) AS cy,
  DATE_TRUNC('month', dt) AS month,
  AVG(do_mg_l) AS do_mean,
  COUNT(*) AS n_obs
FROM clean_surface_region
WHERE dt >= '{START}' AND dt < '{END}'
GROUP BY 1,2,3
ORDER BY 3,1,2
""").fetchdf()

cell["month"] = pd.to_datetime(cell["month"])
cell["cell_id"] = cell["cx"].astype(str) + "_" + cell["cy"].astype(str)
X = cell.pivot(index="month", columns="cell_id", values="do_mean").sort_index()

print("Matrix shape:", X.shape)
print("Overall missing %:", X.isna().mean().mean())
print("Min observed cells in a month:", int(X.notna().sum(axis=1).min()))
print("Median observed cells per month:", float(X.notna().sum(axis=1).median()))
print("Min observed months for a cell:", int(X.notna().sum(axis=0).min()))
print("Median observed months per cell:", float(X.notna().sum(axis=0).median()))

good_cells = X.notna().sum(axis=0) >= MIN_MONTHS_REQUIRED
Xg = X.loc[:, good_cells]

print("Kept cells:", Xg.shape[1], "out of", X.shape[1])
print("New missing %:", Xg.isna().mean().mean())
print("Min months per kept cell:", int(Xg.notna().sum(axis=0).min()))
print("Min cells observed per month:", int(Xg.notna().sum(axis=1).min()))

con.close()