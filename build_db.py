import duckdb
from pathlib import Path

CSV_PATH = "data/DO.csv"
DB_PATH = "gcoos.duckdb"

START = "2002-01-01" # starting time window, decided to focus on 2000s
END = "2013-01-01"

DO_MIN, DO_MAX = 0.0, 25.0
DEPTH_SENTINEL = -900
DEPTH_MAX = 2.0
FIXED_N_DEPTHS_MAX = 10

# dense data region from candidate box results in other file
REGION = {
    "lat_min": 25.75, "lat_max": 29.75,
    "lon_min": -85.25, "lon_max": -80.25
}

con = duckdb.connect(DB_PATH)

# create base_clean table of only rows with required fields and basic numeric filters
con.execute(f"""
CREATE OR REPLACE TABLE base_clean AS
SELECT
  monitoring_location_identifier AS station_id,
  CAST(latitude AS DOUBLE) AS lat,
  CAST(longitude AS DOUBLE) AS lon,
  CAST(x AS DOUBLE) AS x_m,
  CAST(y AS DOUBLE) AS y_m,
  activity_start_date AS dt,
  CAST(activity_depth_height AS DOUBLE) AS depth_m,
  CAST(dissolved_oxygen AS DOUBLE) AS do_mg_l
FROM read_csv_auto('{CSV_PATH}')
WHERE
  activity_start_date >= '{START}' AND activity_start_date < '{END}'
  AND dissolved_oxygen IS NOT NULL
  AND latitude IS NOT NULL
  AND longitude IS NOT NULL
  AND x IS NOT NULL
  AND y IS NOT NULL
  AND activity_depth_height IS NOT NULL
  AND activity_start_date IS NOT NULL
  AND CAST(activity_depth_height AS DOUBLE) > {DEPTH_SENTINEL}
  AND CAST(activity_depth_height AS DOUBLE) < {DEPTH_MAX}
  AND CAST(dissolved_oxygen AS DOUBLE) BETWEEN {DO_MIN} AND {DO_MAX};
""")

# compute station depth-variability on the cleaned surface subset and keep fixed stations
con.execute(f"""
CREATE OR REPLACE TABLE fixed_stations AS
SELECT station_id
FROM (
  SELECT station_id, COUNT(DISTINCT ROUND(depth_m, 2)) AS n_depths
  FROM base_clean
  GROUP BY 1
)
WHERE n_depths <= {FIXED_N_DEPTHS_MAX};
""")

# create clean_surface_all: whole Gulf within  overall data bounds
# create clean_surface_region: dense region subset
con.execute("""
CREATE OR REPLACE TABLE clean_surface_all AS
SELECT b.*
FROM base_clean b
JOIN fixed_stations f USING (station_id);
""")

con.execute(f"""
CREATE OR REPLACE TABLE clean_surface_region AS
SELECT *
FROM clean_surface_all
WHERE lat BETWEEN {REGION["lat_min"]} AND {REGION["lat_max"]}
  AND lon BETWEEN {REGION["lon_min"]} AND {REGION["lon_max"]};
""")

print("\nCounts:")
print(con.execute(f"SELECT COUNT(*) AS n FROM read_csv_auto('{CSV_PATH}')").fetchdf())
print(con.execute("SELECT COUNT(*) AS n FROM base_clean").fetchdf())
print(con.execute("SELECT COUNT(*) AS n FROM fixed_stations").fetchdf())
print(con.execute("SELECT COUNT(*) AS n FROM clean_surface_all").fetchdf())
print(con.execute("SELECT COUNT(*) AS n FROM clean_surface_region").fetchdf())