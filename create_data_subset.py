import duckdb
import pandas as pd

DB_PATH = "gcoos.duckdb"
START = "2004-11-01"
END   = "2009-11-01"

CONFIGS = [
    {"cell_size_m": 15_000, "out": "cleaned_data_15km.parquet"},
    {"cell_size_m": 25_000, "out": "cleaned_data_25km.parquet"},
]

con = duckdb.connect(DB_PATH)

for cfg in CONFIGS:
    CELL_SIZE_M = cfg["cell_size_m"]
    OUT         = cfg["out"]

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

    cell = pd.read_parquet(OUT)
    cell["cell_id"] = cell["cx"].astype(str) + "_" + cell["cy"].astype(str)
    X = cell.pivot(index="month", columns="cell_id", values="do_mean").sort_index()

    good_cells = X.notna().sum(axis=0) >= 24
    Xg = X.loc[:, good_cells]
    print(f"  {CELL_SIZE_M//1000}km: {Xg.shape[1]} cells retained")

con.close()