import time
import numpy as np
from typing import Tuple
from stage2_reconstruction import load_stage1, evaluate_sensor_subset

# budget fractions to optimise, must match Stage 2 sweep for comparison
BUDGET_FRACTIONS = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]

POP_SIZE       = 150    
MAX_GENERATIONS = 200   # upper limit on generations
PATIENCE       = 60     # stop early if best fitness doesn't improve for this many gens
ELITE_SIZE     = 5      # number of top individuals preserved unchanged each gen
CROSSOVER_RATE = 0.85   # probability of crossover vs. direct copy
MUTATION_RATE  = 0.15   # probability each individual undergoes swap mutation
TOURNAMENT_K   = 3      # tournament selection pool size

# fraction of initial population with high-variance cells
INFORMED_FRAC  = 0.0

N_RUNS = 5             

def sequence_to_indices(seq: np.ndarray) -> np.ndarray:
    return np.where(seq)[0]

# repair after crossover
def repair(seq: np.ndarray, p: int, rng: np.random.Generator) -> np.ndarray:
    seq = seq.copy()
    selected = np.where(seq == 1)[0]
    unselected = np.where(seq == 0)[0]
    n_selected = len(selected)

    if n_selected > p:
        # turn off the excess randomly
        turn_off = rng.choice(selected, size=n_selected - p, replace=False)
        seq[turn_off] = 0
    elif n_selected < p:
        # turn on more randomly
        turn_on = rng.choice(unselected, size=p - n_selected, replace=False)
        seq[turn_on] = 1

    return seq

def initialise_population(
    N: int,
    p: int,
    pop_size: int,
    informed_frac: float,
    train_variances: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    population = np.zeros((pop_size, N), dtype=np.int8)
    n_informed = int(pop_size * informed_frac)

    # rank cells by training variance
    ranked = np.argsort(train_variances)[::-1]

    for i in range(pop_size):
        if i < n_informed:
            # take the top-p * 1.5 cells by variance, then sample p from them
            pool_size = min(N, int(p * 1.5))
            pool = ranked[:pool_size]
            selected = rng.choice(pool, size=p, replace=False)
        else:
            selected = rng.choice(N, size=p, replace=False)

        population[i, selected] = 1

    return population

def tournament_select(
    fitness: np.ndarray,
    k: int,
    rng: np.random.Generator,
) -> int:
    candidates = rng.choice(len(fitness), size=k, replace=False)
    return int(candidates[np.argmax(fitness[candidates])])


def uniform_crossover(
    parent_a: np.ndarray,
    parent_b: np.ndarray,
    p: int,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    mask = rng.integers(0, 2, size=len(parent_a), dtype=np.int8)  # random 0/1
    child_a = np.where(mask, parent_a, parent_b)
    child_b = np.where(mask, parent_b, parent_a)

    child_a = repair(child_a, p, rng)
    child_b = repair(child_b, p, rng)

    return child_a, child_b

def swap_mutation(
    seq: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    seq = seq.copy()
    selected   = np.where(seq == 1)[0]
    unselected = np.where(seq == 0)[0]

    if len(selected) == 0 or len(unselected) == 0:
        return seq  

    turn_off = rng.choice(selected)
    turn_on  = rng.choice(unselected)

    seq[turn_off] = 0
    seq[turn_on]  = 1
    return seq

def evaluate_population(
    population: np.ndarray,
    data: dict,
    fitness_cache: dict,
) -> np.ndarray:
    pop_size = population.shape[0]
    fitness  = np.zeros(pop_size)

    for i in range(pop_size):
        key = population[i].tobytes()

        if key not in fitness_cache:
            indices = sequence_to_indices(population[i])
            result  = evaluate_sensor_subset(indices, data)
            fitness_cache[key] = -result["rmse"]   # maximize this to minimize RMSE

        fitness[i] = fitness_cache[key]

    return fitness

def run_ga(
    data: dict,
    p: int,
    cell_size_km: int,
    budget_frac: float,
    rng: np.random.Generator,
) -> dict:
    N = data["X_train_full"].shape[1]

    # training-period variance per cell to seed informed individuals
    train_variances = np.nanvar(data["X_train_full"], axis=0)

    print(f"\n  Budget {budget_frac:.0%}  (p={p} sensors / {N} cells)")
    print(f"  {'Gen':>5}  {'Best RMSE':>10}  {'Mean RMSE':>10}  {'Cache hits':>11}  {'Elapsed':>8}")
    print(f"  {'-'*60}")

    population    = initialise_population(
        N, p, POP_SIZE, INFORMED_FRAC, train_variances, rng
    )
    fitness_cache : dict = {}
    fitness        = evaluate_population(population, data, fitness_cache)

    best_idx      = int(np.argmax(fitness))
    best_seq    = population[best_idx].copy()
    best_fitness  = float(fitness[best_idx])

    history_best  = []
    history_mean  = []
    no_improve    = 0
    t0            = time.time()

    for gen in range(MAX_GENERATIONS):
        elite_idx  = np.argsort(fitness)[::-1][:ELITE_SIZE]
        new_pop    = [population[i].copy() for i in elite_idx]

        while len(new_pop) < POP_SIZE:
            # select two parents via tournament
            idx_a = tournament_select(fitness, TOURNAMENT_K, rng)
            idx_b = tournament_select(fitness, TOURNAMENT_K, rng)

            pa = population[idx_a]
            pb = population[idx_b]

            # crossover
            if rng.random() < CROSSOVER_RATE:
                child_a, child_b = uniform_crossover(pa, pb, p, rng)
            else:
                child_a, child_b = pa.copy(), pb.copy()

            # mutation
            if rng.random() < MUTATION_RATE:
                child_a = swap_mutation(child_a, rng)
            if rng.random() < MUTATION_RATE:
                child_b = swap_mutation(child_b, rng)

            new_pop.append(child_a)
            if len(new_pop) < POP_SIZE:
                new_pop.append(child_b)

        population = np.array(new_pop, dtype=np.int8)
        fitness    = evaluate_population(population, data, fitness_cache)

        gen_best_idx = int(np.argmax(fitness))
        gen_best_fit = float(fitness[gen_best_idx])

        history_best.append(-gen_best_fit)   # store as RMSE (positive)
        history_mean.append(-float(np.mean(fitness[np.isfinite(fitness)])))

        if gen_best_fit > best_fitness + 1e-6:
            best_fitness = gen_best_fit
            best_seq   = population[gen_best_idx].copy()
            no_improve   = 0
        else:
            no_improve  += 1

        if gen % 10 == 0 or no_improve == 0:
            elapsed    = time.time() - t0
            cache_hits = len(fitness_cache)
            print(f"  {gen:5d}  {-best_fitness:10.4f}  "
                  f"{history_mean[-1]:10.4f}  "
                  f"{cache_hits:11d}  "
                  f"{elapsed:7.1f}s")

        # Eearly stopping
        if no_improve >= PATIENCE:
            print(f"  Early stop at generation {gen} "
                  f"(no improvement for {PATIENCE} generations)")
            break

    elapsed = time.time() - t0
    best_rmse  = -best_fitness
    best_indices = sequence_to_indices(best_seq)

    best_result = evaluate_sensor_subset(best_indices, data)

    print(f"\n  Best RMSE: {best_rmse:.4f} mg/L  "
          f"[{elapsed:.1f}s total]")

    return {
        "cell_size_km"  : cell_size_km,
        "budget_frac"   : budget_frac,
        "p"             : p,
        "N"             : N,
        "best_rmse"     : best_rmse,
        "best_indices"  : best_indices,
        "best_seq"    : best_seq,
        "n_months_eval" : best_result["n_months_eval"],
        "per_month_rmse": best_result["per_month_rmse"],
        "history_best"  : np.array(history_best),
        "history_mean"  : np.array(history_mean),
        "generations_run": len(history_best),
        "elapsed_s"     : elapsed,
    }


def print_comparison_table(all_results: list, random_baselines: dict):
    print("\n" + "="*75)
    print("  RESULTS: GA vs Random Baseline")
    print("="*75)
    print(f"  {'Size':>6}  {'Budget':>7}  {'p':>4}  "
          f"{'Random RMSE':>12}  {'GA RMSE':>9}  {'Improvement':>12}  {'Gens':>5}")
    print("  " + "-"*68)

    for r in all_results:
        key = (r["cell_size_km"], r["budget_frac"])
        rand_rmse = random_baselines.get(key, float("nan"))
        improvement = (rand_rmse - r["best_rmse"]) / rand_rmse * 100 if not np.isnan(rand_rmse) else float("nan")
        print(f"  {r['cell_size_km']:>4}km  "
              f"{r['budget_frac']:>6.0%}  "
              f"{r['p']:>4d}  "
              f"{rand_rmse:>12.4f}  "
              f"{r['best_rmse']:>9.4f}  "
              f"{improvement:>11.1f}%  "
              f"{r['generations_run']:>5d}")


if __name__ == "__main__":
    # THESE NEED TO BE PASTED IN EACH RUN FROM PREV OUTPUTS
    RANDOM_BASELINES = {
        (15, 0.3): 1.6524,
        (15, 0.4): 1.6946,
        (15, 0.5): 1.6716,
        (15, 0.6): 1.6966,
        (15, 0.7): 1.5739,
        (15, 0.8): 1.5774,
        (25, 0.3): 1.4847,
        (25, 0.4): 1.5215,
        (25, 0.5): 1.6366,
        (25, 0.6): 1.5714,
        (25, 0.7): 1.5707,
        (25, 0.8): 1.5746,
    }

    grouped     : dict = {}
    all_results : list = []
 
    for km in [15, 25]:
        print(f"\n{'='*60}")
        print(f"  GA — {km}km cells  ({N_RUNS} runs per budget)")
        print(f"{'='*60}")
 
        data = load_stage1(km)
        N    = data["X_train_full"].shape[1]
 
        for frac in BUDGET_FRACTIONS:
            p = max(1, int(round(frac * N)))
            grouped[(km, frac)] = []
 
            for run_idx in range(N_RUNS):
                print(f"\n  --- Run {run_idx + 1}/{N_RUNS} ---")
                rng    = np.random.default_rng()  
                result = run_ga(data, p, km, frac, rng)
                result["run_idx"] = run_idx
                grouped[(km, frac)].append(result)
                all_results.append(result)
 
    # save all runs
    out_path = "ga_results_multiseed_uninformed.npz"
    save_dict  = {}
    config_idx = 0
    for km in [15, 25]:
        for frac in BUDGET_FRACTIONS:
            for run_idx, r in enumerate(grouped[(km, frac)]):
                prefix = f"cfg{config_idx}_run{run_idx}_"
                for k, v in r.items():
                    arr = v if isinstance(v, np.ndarray) else np.array(v)
                    save_dict[prefix + k] = arr
            config_idx += 1
    np.savez(out_path, **save_dict)
 
    # print mean +/- std summary
    print("\n" + "="*80)
    print("  STABILITY SUMMARY")
    print("="*80)
    print(f"  {'Size':>6}  {'Budget':>7}  {'p':>4}  {'Random':>8}  "
          f"{'Mean RMSE':>10}  {'Std RMSE':>9}  {'Improvement':>12}")
    print("  " + "-"*72)
    for km in [15, 25]:
        for frac in BUDGET_FRACTIONS:
            runs  = grouped[(km, frac)]
            rmses = [r["best_rmse"] for r in runs]
            mean_r = float(np.mean(rmses))
            std_r  = float(np.std(rmses))
            rand_r = RANDOM_BASELINES.get((km, frac), float("nan"))
            improv = (rand_r - mean_r) / rand_r * 100
            p      = runs[0]["p"]
            print(f"  {km:>4}km  {frac:>6.0%}  {p:>4d}  {rand_r:>8.4f}  "
                  f"{mean_r:>10.4f}  {std_r:>9.4f}  {improv:>11.1f}%")