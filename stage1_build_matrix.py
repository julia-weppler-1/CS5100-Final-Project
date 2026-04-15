import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
import warnings
import os

CELL_SIZES_TO_RUN = [15_000, 25_000]

PARQUET_PATHS = {
    15_000: "cleaned_data_15km.parquet",
    25_000: "cleaned_data_25km.parquet",
}

# cells observed in fewer than this many months are dropped
MIN_MONTHS_PER_CELL = 24

# first 48 months = training, last 12 months = test
N_TRAIN = 48

# iterative SVD imputation settings
ISVD_RANK    = 10
ISVD_MAXITER = 500


def load_raw_matrix(parquet_path: str, min_months: int) -> pd.DataFrame:
    cell = pd.read_parquet(parquet_path)
    cell["cell_id"] = cell["cx"].astype(str) + "_" + cell["cy"].astype(str)

    X = (
        cell
        .pivot(index="month", columns="cell_id", values="do_mean")
        .sort_index()
    )
    X.index = pd.to_datetime(X.index)

    keep = X.notna().sum(axis=0) >= min_months
    X    = X.loc[:, keep]

    print(f"  Matrix shape: {X.shape}  ")
    print(f"(missing {X.isna().mean().mean():.1%})")
    return X


def impute_iterative_svd(X_raw: np.ndarray, rank: int, max_iter: int) -> np.ndarray:
    missing_mask = np.isnan(X_raw)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        col_means = np.nanmean(X_raw, axis=0)
    col_means = np.where(np.isnan(col_means), 0.0, col_means)

    X_filled = X_raw.copy()
    X_filled[missing_mask] = np.take(col_means, np.where(missing_mask)[1])

    svd          = TruncatedSVD(n_components=rank, random_state=42)
    prev_imputed = X_filled[missing_mask].copy()

    for iteration in range(max_iter):
        X_low_rank = svd.fit_transform(X_filled)
        X_recon    = X_low_rank @ svd.components_

        X_filled[missing_mask] = X_recon[missing_mask]

        new_imputed = X_filled[missing_mask]
        delta       = np.sqrt(np.mean((new_imputed - prev_imputed) ** 2))
        rel_delta   = delta / (np.sqrt(np.mean(new_imputed ** 2)) + 1e-10)

        if rel_delta < 1e-4:
            print(f"    Converged at iteration {iteration + 1} "
                  f"(rel_delta = {rel_delta:.2e})")
            break

        prev_imputed = new_imputed.copy()
    else:
        print(f"    Reached max_iter={max_iter} "
              f"(rel_delta = {rel_delta:.2e})")

    return X_filled


def run_pipeline(cell_size_m: int):
    print(f"\n{'='*60}")
    print(f"  Cell size: {cell_size_m // 1000} km")
    print(f"{'='*60}")

    parquet_path = PARQUET_PATHS[cell_size_m]

    X_df        = load_raw_matrix(parquet_path, MIN_MONTHS_PER_CELL)
    cell_ids    = X_df.columns.tolist()
    month_index = X_df.index

    assert len(X_df) >= N_TRAIN + 1, "Not enough months for the requested split"

    X_train_raw = X_df.iloc[:N_TRAIN].to_numpy()
    X_test_raw  = X_df.iloc[N_TRAIN:].to_numpy()

    train_months = month_index[:N_TRAIN]
    test_months  = month_index[N_TRAIN:]

    print(f"  Train: {train_months[0].strftime('%Y-%m')} through {train_months[-1].strftime('%Y-%m')}  ({len(train_months)} months)")
    print(f"  Test : {test_months[0].strftime('%Y-%m')} through {test_months[-1].strftime('%Y-%m')}  ({len(test_months)} months)")
    print(f"  Train missing: {np.isnan(X_train_raw).mean():.1%}   Test missing: {np.isnan(X_test_raw).mean():.1%}")

    X_train_full = impute_iterative_svd(X_train_raw, ISVD_RANK, ISVD_MAXITER)
    X_train_full = np.clip(X_train_full, 0.0, 25.0)
    print(f"  Residual NaN after imputation: {np.isnan(X_train_full).sum()}")

    tag      = f"{cell_size_m // 1000}km"
    out_path = f"stage1_output_{tag}.npz"

    np.savez(
        out_path,
        X_train_full = X_train_full,
        X_train_raw  = X_train_raw,
        X_test_raw   = X_test_raw,
        cell_ids     = np.array(cell_ids, dtype=object),
        train_months = np.array(train_months.astype(str), dtype=object),
        test_months  = np.array(test_months.astype(str),  dtype=object),
        n_train      = np.array(N_TRAIN),
        cell_size_m  = np.array(cell_size_m),
    )

    print(f"Cells: {len(cell_ids)}   "
          f"Train months: {len(train_months)}   "
          f"Test months: {len(test_months)}")


if __name__ == "__main__":
    for cs in CELL_SIZES_TO_RUN:
        run_pipeline(cs)