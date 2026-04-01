import numpy as np
from sklearn.linear_model import Ridge

RIDGE_ALPHA      = 1.0   # regularisation strength for Ridge
BUDGET_FRACTIONS = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
N_RANDOM_TRIALS  = 50    # random subsets to sample per budget level
SEED             = 42

# Min fraction of selected sensors that must have reported in a given test month for that month to be included in eval
MIN_OBS_FRAC = 0.80


def load_stage1(cell_size_km: int) -> dict:
    path = f"stage1_output_{cell_size_km}km.npz"
    raw  = np.load(path, allow_pickle=True)
    data = {k: raw[k] for k in raw.files}
    data["n_train"]     = int(data["n_train"])
    data["cell_size_m"] = int(data["cell_size_m"])
    data["cell_ids"]    = list(data["cell_ids"])
    return data

def evaluate_sensor_subset(sensor_indices: np.ndarray, data: dict) -> dict:
    # training uses the imputed matrix 
    X_train = data["X_train_full"]   
    # test uses raw observed values
    X_test  = data["X_test_raw"]     

    T_test, N      = X_test.shape
    sensor_indices = np.asarray(sensor_indices)
    p              = len(sensor_indices)
    non_sensor_idx = np.setdiff1d(np.arange(N), sensor_indices)

    # sensor feature matrix for training 
    X_train_sensors = X_train[:, sensor_indices] 

    # train one Ridge model per non-sensor cell
    models = {}
    for cell_idx in non_sensor_idx:
        y_train = X_train[:, cell_idx]             
        model   = Ridge(alpha=RIDGE_ALPHA, fit_intercept=True)
        model.fit(X_train_sensors, y_train)
        models[cell_idx] = model


    sensor_present   = ~np.isnan(X_test[:, sensor_indices])  
    frac_present     = sensor_present.mean(axis=1)            
    valid_months     = np.where(frac_present >= MIN_OBS_FRAC)[0]

    errors_sq      = []
    per_month_rmse = []

    # sensor column means from training,  used to fill if missing sensor values in months that pass the 80% threshold
    sensor_col_means = X_train[:, sensor_indices].mean(axis=0)  

    for t in valid_months:
        # build sensor feature vector for this month, fill any missing sensor values with the training-period mean for that sensor
        x_sensors = X_test[t, sensor_indices].copy()
        missing   = np.isnan(x_sensors)
        x_sensors[missing] = sensor_col_means[missing]

        month_errors = []
        for cell_idx in non_sensor_idx:
            y_true = X_test[t, cell_idx]
            if np.isnan(y_true):
                continue   # no ground truth 

            y_hat = models[cell_idx].predict(x_sensors.reshape(1, -1))[0]
            month_errors.append((y_hat - y_true) ** 2)

        if month_errors:
            mse_t = float(np.mean(month_errors))
            errors_sq.append(mse_t)
            per_month_rmse.append(float(np.sqrt(mse_t)))

    if not errors_sq:
        return {
            "rmse"          : np.inf,
            "n_months_eval" : 0,
            "n_cells_eval"  : len(non_sensor_idx),
            "sensor_indices": sensor_indices,
            "per_month_rmse": np.array([]),
        }

    return {
        "rmse"          : float(np.sqrt(np.mean(errors_sq))),
        "n_months_eval" : len(errors_sq),
        "n_cells_eval"  : len(non_sensor_idx),
        "sensor_indices": sensor_indices,
        "per_month_rmse": np.array(per_month_rmse),
    }


def run_budget_sweep(data: dict, cell_size_km: int) -> list:

    rng = np.random.default_rng(SEED)
    N   = data["X_train_full"].shape[1]

    print(f"\n{'='*60}")
    print(f"  Budget sweep -- {cell_size_km}km cells  (N={N} cells)")
    print(f"{'='*60}")
    print(f"  {'Budget':>8}  {'p':>4}  {'Mean RMSE':>10}  "
          f"{'Std RMSE':>9}  {'Months eval':>12}")
    print("  " + "-"*52)

    results = []

    for frac in BUDGET_FRACTIONS:
        p     = max(1, int(round(frac * N)))
        rmses = []
        months_evals = []

        for _ in range(N_RANDOM_TRIALS):
            subset = rng.choice(N, size=p, replace=False)
            r      = evaluate_sensor_subset(subset, data)
            if np.isfinite(r["rmse"]):
                rmses.append(r["rmse"])
                months_evals.append(r["n_months_eval"])

        mean_rmse  = float(np.mean(rmses))        if rmses else np.nan
        std_rmse   = float(np.std(rmses))         if rmses else np.nan
        avg_months = float(np.mean(months_evals)) if months_evals else np.nan

        print(f"  {frac:.0%}:    {p:4d}  {mean_rmse:10.4f}  "
              f"{std_rmse:9.4f}  {avg_months:12.1f}")

        results.append({
            "cell_size_km"       : cell_size_km,
            "budget_frac"        : frac,
            "p"                  : p,
            "N"                  : N,
            "random_rmse_mean"   : mean_rmse,
            "random_rmse_std"    : std_rmse,
            "avg_months_evaluated": avg_months,
        })

    return results

if __name__ == "__main__":
    for km in [15, 25]:
        data = load_stage1(km)
        N    = data["X_train_full"].shape[1]
        print(f"  Cells: {N}   "
              f"Train shape: {data['X_train_full'].shape}   "
              f"Test shape: {data['X_test_raw'].shape}")
        run_budget_sweep(data, km)

    print("These random baselines are what the GA must beat.")