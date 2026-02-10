import itertools
import pandas as pd
from pathlib import Path
import json
from tracellm_utils import calculate_f2_metrics, normalize_id_columns, to_binary_label

# =========================
# User Config (edit here)
# =========================
DATASET = "EasyClinic_UC_TC" 
MODE = "test"          # "val" or "test" (val should be run first)

SEEDS_PER_MODEL = 5    # number of different seeds
REPEATS_PER_SEED = 5   # runs per seed
RUNS = SEEDS_PER_MODEL * REPEATS_PER_SEED  

# This should be changed for each dataset. It corresponds to the folder names containing the predictions.
SEED_FOLDERS = {
    "mistral": [
        "mistralai/mistral-7b-instruct/diverse/EasyClinic_UC_TC_1764590890_1",
        "mistralai/mistral-7b-instruct/diverse/EasyClinic_UC_TC_1764590896_2",
        "mistralai/mistral-7b-instruct/diverse/EasyClinic_UC_TC_1764590935_3",
        "mistralai/mistral-7b-instruct/diverse/EasyClinic_UC_TC_1764590950_4",
        "mistralai/mistral-7b-instruct/diverse/EasyClinic_UC_TC_1764590995_5",
    ],
    "gemma": [
        "google/gemma-3-12b-it/diverse/EasyClinic_UC_TC_1764591339_1",
        "google/gemma-3-12b-it/diverse/EasyClinic_UC_TC_1764591466_2",
        "google/gemma-3-12b-it/diverse/EasyClinic_UC_TC_1764592018_3",
        "google/gemma-3-12b-it/diverse/EasyClinic_UC_TC_1764592423_4",
        "google/gemma-3-12b-it/diverse/EasyClinic_UC_TC_1764596679_5",
    ],
    "qwen": [
        "qwen/qwen-2.5-7b-instruct/diverse/EasyClinic_UC_TC_1764598459_1",
        "qwen/qwen-2.5-7b-instruct/diverse/EasyClinic_UC_TC_1764598463_2",
        "qwen/qwen-2.5-7b-instruct/diverse/EasyClinic_UC_TC_1764598492_3",
        "qwen/qwen-2.5-7b-instruct/diverse/EasyClinic_UC_TC_1764598729_4",
        "qwen/qwen-2.5-7b-instruct/diverse/EasyClinic_UC_TC_1764598911_5",
    ],
    "deepseek": [
        "deepseek/deepseek-r1-0528-qwen3-8b/diverse/EasyClinic_UC_TC_1764599135_1",
        "deepseek/deepseek-r1-0528-qwen3-8b/diverse/EasyClinic_UC_TC_1764599629_2",
        "deepseek/deepseek-r1-0528-qwen3-8b/diverse/EasyClinic_UC_TC_1764599679_3",
        "deepseek/deepseek-r1-0528-qwen3-8b/diverse/EasyClinic_UC_TC_1764600669_4",
        "deepseek/deepseek-r1-0528-qwen3-8b/diverse/EasyClinic_UC_TC_1764600743_5",
    ],
    "phi": [
        "microsoft/phi-4-reasoning-plus/diverse/EasyClinic_UC_TC_1764603722_1",
        "microsoft/phi-4-reasoning-plus/diverse/EasyClinic_UC_TC_1764603727_2",
        "microsoft/phi-4-reasoning-plus/diverse/EasyClinic_UC_TC_1764603730_3",
        "microsoft/phi-4-reasoning-plus/diverse/EasyClinic_UC_TC_1764603801_4",
        "microsoft/phi-4-reasoning-plus/diverse/EasyClinic_UC_TC_1764603810_5",
    ],
}

# =========================
# Paths
# =========================
INPUT_ROOT = Path("Results/TLC")
MERGED_OUT_DIR = Path(f"Datasets/{DATASET}_Light_LLMs")
ENSEMBLE_DIR = Path(f"Results/TLC/_ensemble/LightLLM_MV/{DATASET}")
MERGED_OUT_DIR.mkdir(parents=True, exist_ok=True)
ENSEMBLE_DIR.mkdir(parents=True, exist_ok=True)
VAL_ART_PATH = ENSEMBLE_DIR / "Light_LLMs_val_artifacts.json"

# =========================
# Helpers
# =========================
# Note: Using shared utilities from tracellm_utils.py to avoid code duplication
# - normalize_id_columns: Standardizes source_id/target_id column names
# - to_binary_label (aliased as to01): Converts labels to binary 0/1
# - calculate_f2_metrics (aliased as metrics): Calculates recall, precision, F2

# Alias for backward compatibility
to01 = to_binary_label
metrics = calculate_f2_metrics

def load_run_df_for_model(model_key, run_idx):
    # Map global run_idx (1..25) → (seed_index, local_run 1..5)
    seed_index = (run_idx - 1) // REPEATS_PER_SEED   # 0..4
    local_run = (run_idx - 1) % REPEATS_PER_SEED + 1 # 1..5

    try:
        folder = SEED_FOLDERS[model_key][seed_index]
    except (KeyError, IndexError):
        raise ValueError(f"No seed folder for model '{model_key}', seed_index {seed_index}")

    fp = INPUT_ROOT / folder / "role_1_shot_2" / f"2_shot_P1_run_{local_run}.csv"
    if not fp.exists():
        raise FileNotFoundError(
            f"Missing file for model '{model_key}', global run {run_idx} "
            f"(seed_index={seed_index}, local_run={local_run}): {fp}"
        )

    df = pd.read_csv(fp)
    df = normalize_id_columns(df)
    required = ["source_id","source_content","target_id","target_content","label","predicted_label","rationale"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{fp} is missing columns: {missing}")
    return df[required].copy()

def merge_all_models_for_run(run_idx, model_keys):
    base_df = None
    for mk in model_keys:
        df = load_run_df_for_model(mk, run_idx)
        if base_df is None:
            base_df = df.copy()
        else:
            base_df = base_df.merge(
                df,
                on=["source_id","source_content","target_id","target_content","label"],
                suffixes=("","_dup"),
                how="inner"
            )
            dup_cols = [c for c in base_df.columns if c.endswith("_dup")]
            if dup_cols:
                base_df.drop(columns=dup_cols, inplace=True)
        base_df.rename(
            columns={"predicted_label": f"{mk}_predicted_label","rationale": f"{mk}_rationale"},
            inplace=True
        )
    cols = ["source_id","source_content","target_id","target_content","label"]
    for mk in model_keys:
        cols += [f"{mk}_predicted_label", f"{mk}_rationale"]
    base_df = base_df[cols]
    base_df["label"] = pd.to_numeric(base_df["label"], errors="coerce").fillna(0).astype(int).clip(0,1)
    for mk in model_keys:
        c = f"{mk}_predicted_label"
        base_df[c] = pd.to_numeric(base_df[c], errors="coerce").fillna(0).astype(int).clip(0,1)
    return base_df

def save_merged_per_run(run_idx, df):
    out_path = MERGED_OUT_DIR / f"{DATASET}_Light_LLMs_{MODE}_run_{run_idx}.csv"
    df.to_csv(out_path, index=False)
    return out_path

def majority_vote_with_tiebreak(preds_dict, model_rank_order):
    bin_preds = {}
    for m, v in preds_dict.items():
        if v is None:
            continue
        bin_preds[m] = to01(v)
    counts0 = sum(1 for v in bin_preds.values() if v == 0)
    counts1 = sum(1 for v in bin_preds.values() if v == 1)
    if counts0 > counts1: return 0
    if counts1 > counts0: return 1
    for m in model_rank_order:
        if m in bin_preds:
            return bin_preds[m]
    return 0

def evaluate_combination(df_merged, combo, model_rank_order):
    y_true = df_merged["label"].apply(to01)
    y_pred = []
    for _, row in df_merged.iterrows():
        preds = {m: row[f"{m}_predicted_label"] for m in combo}
        y_pred.append(majority_vote_with_tiebreak(preds, model_rank_order))
    y_pred = pd.Series(y_pred)
    return metrics(y_true, y_pred)

def per_model_f2_rank_for_run(df_merged, model_keys):
    scores = {}
    y_true = df_merged["label"].apply(to01)
    for m in model_keys:
        y_pred = df_merged[f"{m}_predicted_label"].apply(to01)
        r, p, f2 = metrics(y_true, y_pred)
        scores[m] = f2
    ranked = [m for m, _ in sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))]
    return ranked, scores

def write_combinations_performance(perf_rows, outfile, append_best=True):
    df = pd.DataFrame(perf_rows)
    avg = df.groupby("combination", as_index=False)[["recall","precision","f2"]].mean()
    avg["run"] = "avg"
    df_out = pd.concat([df, avg], ignore_index=True)
    if append_best and not avg.empty:
        best = avg.loc[avg["f2"].idxmax()]
        df_out["best_combination_overall"] = ""
        df_out.loc[df_out["combination"] == best["combination"], "best_combination_overall"] = "YES"
    df_out.to_csv(outfile, index=False)
    return df_out

def read_best_combinations_from_val():
    perf_file = ENSEMBLE_DIR / f"Light_LLMs_ensemble_combinations_performance_val.csv"
    if not perf_file.exists():
        raise FileNotFoundError(f"Expected val performance file not found: {perf_file}")
    df = pd.read_csv(perf_file)
    best = df[(df.get("run") == "avg") & (df.get("best_combination_overall") == "YES")]
    if not best.empty:
        return best["combination"].tolist()
    g = df.groupby("combination", as_index=False)[["recall","precision","f2"]].mean()
    max_f2 = g["f2"].max()
    return g[g["f2"] == max_f2]["combination"].tolist()
   


def save_val_artifacts(best_combos, per_model_avg_f2):
    data = {
        "best_combinations": best_combos,
        "per_model_avg_f2": per_model_avg_f2,
        "model_rank": [m for m, _ in sorted(per_model_avg_f2.items(), key=lambda kv: (-kv[1], kv[0]))]
    }
    with open(VAL_ART_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def load_val_artifacts():
    with open(VAL_ART_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    data.setdefault("best_combinations", [])
    data.setdefault("per_model_avg_f2", {})
    if "model_rank" not in data or not data["model_rank"]:
        if data["per_model_avg_f2"]:
            data["model_rank"] = [m for m, _ in sorted(data["per_model_avg_f2"].items(), key=lambda kv: (-kv[1], kv[0]))]
        else:
            data["model_rank"] = sorted(SEED_FOLDERS.keys())
    return data

def compute_per_model_avg_f2_over_val(model_keys):
    f2_sum = {m: 0.0 for m in model_keys}
    n_runs = 0
    for i in range(1, RUNS+1):
        merged_fp = MERGED_OUT_DIR / f"{DATASET}_Light_LLMs_val_run_{i}.csv"
        if not merged_fp.exists(): continue
        df = pd.read_csv(merged_fp)
        df["label"] = pd.to_numeric(df["label"], errors="coerce").fillna(0).astype(int).clip(0,1)
        y_true = df["label"]
        for m in model_keys:
            c = f"{m}_predicted_label"
            if c not in df.columns: continue
            y_pred = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int).clip(0,1)
            r, p, f2 = metrics(y_true, y_pred)
            f2_sum[m] += f2
        n_runs += 1
    if n_runs == 0:
        return {m: 0.0 for m in model_keys}
    return {m: f2_sum[m] / n_runs for m in model_keys}

# =========================
# Step 1 & 3: Merge per run
# =========================
model_keys_all = list(SEED_FOLDERS.keys())
for run in range(1, RUNS+1):
    merged = merge_all_models_for_run(run, model_keys_all)
    out_path = save_merged_per_run(run, merged)
    print(f"[INFO] Saved merged file for run {run}: {out_path}")

# =========================
# Step 2 (val): Evaluate combos (>=3) + save artifacts
# =========================
if MODE == "val":
    perf_rows = []
    for run in range(1, RUNS+1):
        merged_fp = MERGED_OUT_DIR / f"{DATASET}_Light_LLMs_val_run_{run}.csv"
        df = pd.read_csv(merged_fp)
        df["label"] = pd.to_numeric(df["label"], errors="coerce").fillna(0).astype(int).clip(0,1)
        for mk in model_keys_all:
            c = f"{mk}_predicted_label"
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int).clip(0,1)
        # Fixed global rank from avg per-model F2 across all val runs (deployment-faithful)
        # (Compute once at end via compute_per_model_avg_f2_over_val, then reuse for test)
        # For evaluation within runs we can use this same global rank to mirror deployment:
        # build it lazily on first run's first combo computation if needed
        # Here we'll compute predictions using per-run independent of rank, but to mirror deployment use a
        # global rank after computing perf_rows; simpler: compute local rank per run as fallback
        per_run_rank, _ = per_model_f2_rank_for_run(df, model_keys_all)
        model_rank_order = per_run_rank  # or replace with global if you prefer strict deployment mirror

        for k in range(3, len(model_keys_all)+1):
            for combo in itertools.combinations(model_keys_all, k):
                combo = tuple(sorted(combo))
                r, p, f2 = evaluate_combination(df, combo, model_rank_order)
                perf_rows.append({"run": run,"combination": "+".join(combo),"recall": r,"precision": p,"f2": f2})

    perf_file = ENSEMBLE_DIR / f"Light_LLMs_ensemble_combinations_performance_{MODE}.csv"
    df_perf = write_combinations_performance(perf_rows, perf_file, append_best=True)
    print(f"[INFO] Wrote combinations performance ({MODE}) to: {perf_file}")

    # Best combos from the avg rows just written
    best_avg = df_perf[(df_perf["run"] == "avg") & (df_perf.get("best_combination_overall","") == "YES")]
    best_combos = best_avg["combination"].tolist() if not best_avg.empty else read_best_combinations_from_val()

    # Persist artifacts for test (best combos + global model rank from val)
    per_model_avg_f2 = compute_per_model_avg_f2_over_val(model_keys_all)
    save_val_artifacts(best_combos, per_model_avg_f2)
    print(f"[INFO] Saved validation artifacts to: {VAL_ART_PATH}")
    if best_combos:
        print("[INFO] Best combo(s) by avg F2:")
        for cmb in best_combos: print("   -", cmb)

# =========================
# Step 4 (test): Evaluate ONLY best combo(s) + ALL-models baseline
# =========================
if MODE == "test":
    if VAL_ART_PATH.exists():
        art = load_val_artifacts()
        best_combos = art["best_combinations"]
        global_rank = art["model_rank"]
    else:
        best_combos = read_best_combinations_from_val()
        global_rank = sorted(model_keys_all)

    if not best_combos:
        raise RuntimeError("No best combinations found from validation. Run MODE='val' first.")

    print(f"[INFO] Loaded {len(best_combos)} best combination(s) from validation.")
    print(f"[INFO] Using global model ranking (tie-break): {global_rank}")

    # ---------- (A) Best combination(s): predictions + performance ----------
    all_pred_rows = []
    perf_rows_best = []

    for run in range(1, RUNS + 1):
        merged_fp = MERGED_OUT_DIR / f"{DATASET}_Light_LLMs_test_run_{run}.csv"
        if not merged_fp.exists():
            raise FileNotFoundError(f"Missing merged test file for run {run}: {merged_fp}")
        df = pd.read_csv(merged_fp)
        df["label"] = pd.to_numeric(df["label"], errors="coerce").fillna(0).astype(int).clip(0, 1)
        for mk in model_keys_all:
            c = f"{mk}_predicted_label"
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int).clip(0, 1)

        y_true = df["label"]

        for combo_str in best_combos:
            combo = tuple(combo_str.split("+"))
            ens_preds = []
            for _, row in df.iterrows():
                preds = {m: row.get(f"{m}_predicted_label") for m in combo}
                pred_bin = majority_vote_with_tiebreak(preds, global_rank)
                ens_preds.append(pred_bin)
                all_pred_rows.append({
                    "run": run,
                    "source_id": row["source_id"],
                    "target_id": row["target_id"],
                    "label": row["label"],
                    "combination": combo_str,
                    "ensemble_predicted_label": pred_bin
                })

            r, p, f2 = metrics(y_true, pd.Series(ens_preds))
            perf_rows_best.append({
                "run": run,
                "combination": combo_str,
                "recall": r,
                "precision": p,
                "f2": f2
            })

    best_pred_file = ENSEMBLE_DIR / f"_Light_LLMs_ensemble_combinations_best_test.csv"
    pd.DataFrame(all_pred_rows).to_csv(best_pred_file, index=False)
    print(f"[INFO] Wrote best-combination predictions (test) to: {best_pred_file}")

    best_perf_file = ENSEMBLE_DIR / f"_Light_LLMs_ensemble_combinations_best_performance_test.csv"
    _ = write_combinations_performance(perf_rows_best, best_perf_file, append_best=False)
    print(f"[INFO] Wrote best-combination performance (test) to: {best_perf_file}")

    # ---------- (B) ALL-models baseline: performance only ----------
    perf_rows_all_models = []
    combo_all = tuple(sorted(model_keys_all))
    combo_all_str = "+".join(combo_all)

    for run in range(1, RUNS + 1):
        merged_fp = MERGED_OUT_DIR / f"{DATASET}_Light_LLMs_test_run_{run}.csv"
        if not merged_fp.exists():
            raise FileNotFoundError(f"Missing merged test file for run {run}: {merged_fp}")
        df = pd.read_csv(merged_fp)
        df["label"] = pd.to_numeric(df["label"], errors="coerce").fillna(0).astype(int).clip(0, 1)
        for mk in model_keys_all:
            c = f"{mk}_predicted_label"
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int).clip(0, 1)

        # Evaluate the all-models ensemble on this run
        r, p, f2 = evaluate_combination(df, combo_all, global_rank)
        perf_rows_all_models.append({
            "run": run,
            "combination": combo_all_str,
            "recall": r,
            "precision": p,
            "f2": f2
        })

    all_models_perf_file = ENSEMBLE_DIR / f"_Light_LLMs_ensemble_all_models_performance_test.csv"
    _ = write_combinations_performance(perf_rows_all_models, all_models_perf_file, append_best=False)
    print(f"[INFO] Wrote ALL-models ensemble performance (test) to: {all_models_perf_file}")