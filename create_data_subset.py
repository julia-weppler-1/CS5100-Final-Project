import duckdb
import pandas as pd

DB_PATH = "gcoos.duckdb"

CELL_SIZE_M = 15_000
START = "2004-11-01"
END   = "2009-11-01" 
OUT   = "cleaned_data.parquet"

con = duckdb.connect(DB_PATH)

con.execute(f"""
COPY (
  SELECT
    FLOOR(x_m / {CELL_SIZE_M}) AS cx,
    FLOOR(y_m / {CELL_SIZE_M}) AS cy,
    DATE_TRUNC('month', dt) AS month,
    AVG(do_mg_l) AS do_mean,
    COUNT(*) AS n_obs
  FROM clean_surface_region
  WHERE dt >= '{START}' AND dt < '{END}'
  GROUP BY 1,2,3
) TO '{OUT}' (FORMAT PARQUET);
""")
print("Wrote:", OUT)

print(con.execute(f"""
WITH cell_month AS (SELECT * FROM read_parquet('{OUT}')),
dims AS (
  SELECT COUNT(DISTINCT (cx,cy)) AS n_cells,
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
  (SELECT n_cells FROM dims) AS n_cells,
  (SELECT n_months FROM dims) AS n_months,
  (SELECT n_filled FROM dims) AS n_filled,
  (SELECT n_filled / (n_cells*n_months)::DOUBLE FROM dims) AS density,
  (SELECT MIN(n) FROM per_month) AS min_cells_per_month,
  (SELECT APPROX_QUANTILE(n, 0.50) FROM per_month) AS med_cells_per_month,
  (SELECT APPROX_QUANTILE(n, 0.50) FROM per_cell) AS med_months_per_cell
""").fetchdf())

cell = pd.read_parquet(OUT)
cell["cell_id"] = cell["cx"].astype(str) + "_" + cell["cy"].astype(str)
X = cell.pivot(index="month", columns="cell_id", values="do_mean").sort_index()
min_months_required = 24  

good_cells = X.notna().sum(axis=0) >= min_months_required
Xg = X.loc[:, good_cells]

Xg_out = "Xg_out.parquet"
Xg.to_parquet(Xg_out)
print("Saved filtered matrix:", Xg_out)
con.close()