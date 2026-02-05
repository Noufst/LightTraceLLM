import itertools
import os
import subprocess
import sys
import argparse
import pandas as pd
from pathlib import Path

# =========================
# Static / Default Config
# =========================

# High-End Model Configuration (for escalation)
HIGH_END_MODEL_NAME = "openai/gpt-4o-mini"  # high-end escalation model

RUN_1_PY = "Run.py"

# Helpers for ID normalization
POSSIBLE_SRC_ID_COLS = ["source_id", "source_ID", "UC"]
POSSIBLE_TGT_ID_COLS = ["target_id", "target_ID", "TC", "ID"]


def get_system_role_for_dataset(dataset):
    ds = str(dataset).lower()

    # EasyClinic UC–TC
    if "easyclinic_uc_tc" in ds:
        return (
            "You are an expert in software traceability. You are given two artifacts from a healthcare system. "
            "(1) is a use case and (2) is a test case. Does (2) directly test (1)?\n\n"
            "Respond strictly in JSON format with the following structure:\n"
            "{\n"
            "  'decision': 'yes' or 'no',\n"
            "  'rationale': '<brief explanation>'\n"
            "}"
        )

    # EasyClinic UC–ID
    if "easyclinic_uc_id" in ds:
        return (
            "You are an expert in software traceability. You are given two artifacts from a healthcare system. "
            "(1) is a use case and (2) is an interaction diagram. Does (2) directly realize (1)?\n\n"
            "Respond strictly in JSON format with the following structure:\n"
            "{\n"
            "  'decision': 'yes' or 'no',\n"
            "  'rationale': '<brief explanation>'\n"
            "}"
        )

    # CCHIT (regulatory / certification requirements)
    if "cchit" in ds:
        return (
            "You are an expert in software traceability. You are given two artifacts from a healthcare system. "
            "(1) is a requirement and (2) is a regulation. Does (1) directly satisfy (2) ?\n\n"
            "Respond strictly in JSON format with the following structure:\n"
            "{\n"
            "  'decision': 'yes' or 'no',\n"
            "  'rationale': '<brief explanation>'\n"
            "}"
        )

    # CM1 NASA
    if "cm1" in ds:
        return (
            "You are an expert in software traceability. You are given two artifacts from an aerospace system. "
            "(1) is a high-level requirement and (2) is a design element. Does (2) directly fulfill (1)?\n\n"
            "Respond strictly in JSON format with the following structure:\n"
            "{\n"
            "  'decision': 'yes' or 'no',\n"
            "  'rationale': '<brief explanation>'\n"
            "}"
        )

def normalize_id_columns(df):
    df = df.copy()
    # Normalize source id
    for c in POSSIBLE_SRC_ID_COLS:
        if c in df.columns:
            if c != "source_id":
                df.rename(columns={c: "source_id"}, inplace=True)
            break
    else:
        raise ValueError(f"Missing source id column: one of {POSSIBLE_SRC_ID_COLS}")
    # Normalize target id
    for c in POSSIBLE_TGT_ID_COLS:
        if c in df.columns:
            if c != "target_id":
                df.rename(columns={c: "target_id"}, inplace=True)
            break
    else:
        raise ValueError(f"Missing target id column: one of {POSSIBLE_TGT_ID_COLS}")
    return df


def metrics(y_true, y_pred):
    tp = ((y_true == 1) & (y_pred == 1)).sum()
    fp = ((y_true == 0) & (y_pred == 1)).sum()
    fn = ((y_true == 1) & (y_pred == 0)).sum()
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    beta2 = 4.0
    denom = (beta2 * precision) + recall
    f2 = (1 + beta2) * precision * recall / denom if denom else 0.0
    return recall, precision, f2


def load_merged_light_run(run_idx):
    fp = MERGED_DIR / f"{DATASET}_Light_LLMs_{MODE}_run_{run_idx}.csv"
    if not fp.exists():
        raise FileNotFoundError(f"Missing merged file for run {run_idx}: {fp}")
    df = pd.read_csv(fp)
    df = normalize_id_columns(df)
    df["label"] = pd.to_numeric(df["label"], errors="coerce").fillna(0).astype(int).clip(0, 1)
    for c in df.columns:
        if c.endswith("_predicted_label"):
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int).clip(0, 1)
    return df


def detect_global_disagreement(df, model_keys):
    preds = df[[f"{m}_predicted_label" for m in model_keys]].values
    return (preds.min(axis=1) != preds.max(axis=1))


def run_high_end_model(input_pairs_csv, output_csv, run_seed, mode):
    """Launch Run_2.py and stream its output live to the console."""
    args = []
    for k, v in BASE_RUN1_ARGS.items():
        if k == "--mode":
            v = mode
        if k == "--random_seed":
            v = str(run_seed)  # or keep "1" if you want fixed seed always
        if k == "--input_pairs_csv":
            v = str(input_pairs_csv)
        if k == "--output_csv":
            v = str(output_csv)
        if v is None:
            # boolean flags like --balanced
            if k in ("--balanced",):
                args.append(k)
            continue
        args.extend([k, v])

    cmd = [sys.executable, "-u", RUN_1_PY] + args  # -u = unbuffered
    print("[INFO] Launching high-end model call:", flush=True)
    print(" ", " ".join(map(str, cmd)), flush=True)

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=env,
    )
    # stream line-by-line
    for line in proc.stdout:
        print(f"[HIGH-END] {line.rstrip()}", flush=True)
    proc.wait()
    if proc.returncode != 0:
        raise RuntimeError(f"Run_2.py failed with code {proc.returncode}")


def load_high_end_output(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"High-end output missing: {path}")
    df = pd.read_csv(path)
    df = normalize_id_columns(df)
    required = [
        "source_id",
        "source_content",
        "target_id",
        "target_content",
        "label",
        "predicted_label",
        "rationale",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{path} missing columns: {missing}")
    df["label"] = pd.to_numeric(df["label"], errors="coerce").fillna(0).astype(int).clip(0, 1)
    df["predicted_label"] = pd.to_numeric(df["predicted_label"], errors="coerce").fillna(0).astype(int).clip(0, 1)
    df.rename(
        columns={
            "predicted_label": "highend_predicted_label",
            "rationale": "highend_rationale",
        },
        inplace=True,
    )
    return df


def unanimous_label_for_combo(row, combo):
    vals = [row[f"{m}_predicted_label"] for m in combo]
    mn, mx = min(vals), max(vals)
    if mn == mx:
        return mn  # unanimous 0 or 1
    return None    # disagreement within combo


def evaluate_combo_using_cache(df, combo, high_map):
    preds = []
    for _, row in df.iterrows():
        u = unanimous_label_for_combo(row, combo)
        if u is not None:
            preds.append(u)
        else:
            key = (row["source_id"], row["target_id"], row["label"])
            preds.append(int(high_map.get(key, 0)))  # safe fallback
    return pd.Series(preds)


if __name__ == "__main__":
    # =========================
    # CLI Args
    # =========================
    parser = argparse.ArgumentParser(description="LightLLM_HE disagreement-escalation ensemble.")
    parser.add_argument("--dataset", required=True, help="Dataset name, e.g. CM1_NASA, EasyClinic_UC_TC, EasyClinic_UC_ID, CCHIT")
    parser.add_argument("--mode", required=True, choices=["val", "test"], help="Mode: val or test")
    parser.add_argument("--runs", type=int, required=True, help="Total runs (e.g., 25)")
    parser.add_argument("--min_combo", type=int, default=3, help="Minimum combo size (default: 3)")

    args = parser.parse_args()

    # Assign CLI args to globals used above
    DATASET = args.dataset
    MODE = args.mode
    RUNS = args.runs
    MIN_COMBO = args.min_combo

    print(f"[INFO] Dataset: {DATASET}")
    print(f"[INFO] Mode: {MODE}")
    print(f"[INFO] RUNS: {RUNS}")
    print(f"[INFO] MIN_COMBO: {MIN_COMBO}")

    # =========================
    # Derived Paths / Config
    # =========================
    MERGED_DIR = Path(f"Datasets/{DATASET}_Light_LLMs")

    ENSEMBLE_DIR = Path(f"Results/TLC/_ensemble/LightLLM_HE/{DATASET}")
    ENSEMBLE_DIR.mkdir(parents=True, exist_ok=True)

    HIGH_END_MODEL_ROOT = Path(
        f"Results/TLC/{HIGH_END_MODEL_NAME}/diverse/{DATASET}_high_end_model_escalation/{MODE}"
    )
    HIGH_END_MODEL_SUBDIR = "role_1_shot_2"
    HIGH_END_MODEL_ROOT.joinpath(HIGH_END_MODEL_SUBDIR).mkdir(parents=True, exist_ok=True)
    HIGH_END_MODEL_FILENAME_TEMPLATE = "2_shot_P1_run_{i}.csv"

    # Default Run_2.py args for the high-end call
    BASE_RUN1_ARGS = {
        "--mode": None,  # filled from MODE
        "--num_repeats": "1",
        "--inference_model_name": HIGH_END_MODEL_NAME,
        "--inference_model_temperature": "0",
        "--embedding_model_name": "all-mpnet-base-v2",
        "--task": "TLC",
        "--n_shots": "2",
        "--balanced": None,                 # boolean flag
        "--selection_strategy": "diverse",
        "--diversity_strategy": "sum",
        "--random_seed": "1",               # fixed seed (overwritten per run if needed)
        "--dataset": DATASET,
        "--system_roles": get_system_role_for_dataset(DATASET),
        "--input_pairs_csv": None,          # filled per run
        "--output_csv": None,               # filled per run
    }

    # =========================
    # Main Script
    # =========================
    perf_rows = []
    all_combos = None  # build after seeing model_keys once
    combo_all = None   # ALL-models baseline

    for run in range(1, RUNS + 1):
        print(f"\n[INFO] ===== Run {run} ({MODE}) =====", flush=True)
        df = load_merged_light_run(run)

        # Determine which Light LLMs are present
        model_keys = sorted(
            [c.split("_predicted_label")[0] for c in df.columns if c.endswith("_predicted_label")]
        )
        print(f"[INFO] Light-LLMs: {model_keys}", flush=True)

        # Build combos (once)
        if all_combos is None:
            all_combos = []
            for k in range(MIN_COMBO, len(model_keys) + 1):
                for combo in itertools.combinations(model_keys, k):
                    all_combos.append(tuple(sorted(combo)))
            combo_all = tuple(sorted(model_keys))
            print(f"[INFO] #Combos (>= {MIN_COMBO}): {len(all_combos)}", flush=True)
            print(f"[INFO] ALL-models combo: {combo_all}", flush=True)

        # ---- 1) Global disagree superset across ALL models (one-time escalation per run)
        global_disagree_mask = detect_global_disagreement(df, model_keys)
        df_disagree_global = df.loc[
            global_disagree_mask,
            ["source_id", "source_content", "target_id", "target_content", "label"],
        ].copy()
        print(f"[INFO] Global disagree count = {len(df_disagree_global)} / {len(df)}", flush=True)

        # ---- 2) Call high-end model ONCE per run (if anything to escalate)
        high_map = {}
        if len(df_disagree_global) > 0:
            tmp_dir = ENSEMBLE_DIR / "escalation_tmp"
            tmp_dir.mkdir(parents=True, exist_ok=True)
            in_csv = tmp_dir / f"escalate_{DATASET}_{MODE}_run_{run}.csv"
            out_csv = HIGH_END_MODEL_ROOT / HIGH_END_MODEL_SUBDIR / HIGH_END_MODEL_FILENAME_TEMPLATE.format(i=run)
            df_disagree_global.to_csv(in_csv, index=False)
            run_high_end_model(in_csv, out_csv, run_seed=run, mode=MODE)
            high_df = load_high_end_output(out_csv)
            high_map = {
                (r["source_id"], r["target_id"], r["label"]): r["highend_predicted_label"]
                for _, r in high_df.iterrows()
            }
        else:
            print("[INFO] No disagreements to escalate in this run.", flush=True)

        # ---------- Baseline: ALL models with disagreement-escalation ----------
        y_pred_all = evaluate_combo_using_cache(df, combo_all, high_map)
        r_all, p_all, f2_all = metrics(df["label"], y_pred_all)
        perf_rows.append({
            "run": run,
            "combination": "+".join(combo_all),
            "recall": r_all,
            "precision": p_all,
            "f2": f2_all,
            "n_rows": int(len(df)),
            "n_global_escalated": int(global_disagree_mask.sum()),
            "tag": "ALL_MODELS_BASELINE"
        })
        print(f"[INFO] ALL-models baseline F2={f2_all:.4f}", flush=True)

        # ---- 3) Validation: evaluate ALL combos (size ≥ MIN_COMBO)
        if MODE == "val":
            print(f"[INFO] Evaluating {len(all_combos)} combos (>= {MIN_COMBO}) ...", flush=True)
            for combo in all_combos:
                y_pred = evaluate_combo_using_cache(df, combo, high_map)
                r, p, f2 = metrics(df["label"], y_pred)
                perf_rows.append({
                    "run": run,
                    "combination": "+".join(combo),
                    "recall": r,
                    "precision": p,
                    "f2": f2,
                    "n_rows": int(len(df)),
                    "n_global_escalated": int(global_disagree_mask.sum())
                })

    # =========================
    # Save / Post-process Results
    # =========================
    perf_df = pd.DataFrame(perf_rows)

    if MODE == "val":
        # Per-combo averages across runs
        avg = perf_df.groupby("combination", as_index=False)[["recall", "precision", "f2"]].mean()
        avg["run"] = "avg"
        out_df = pd.concat([perf_df, avg], ignore_index=True)

        # Mark best by avg F2 (include all combos; baseline row is included but has a tag)
        best_pool = avg.sort_values("f2", ascending=False)
        best_row = best_pool.iloc[0] if not best_pool.empty else None
        if best_row is not None:
            out_df["best_combination_overall"] = ""
            out_df.loc[out_df["combination"] == best_row["combination"], "best_combination_overall"] = "YES"
            print(f"[INFO] Best combo by avg F2: {best_row['combination']} (F2={best_row['f2']:.4f})", flush=True)

        # Write full combinations table
        combos_out = ENSEMBLE_DIR / f"DisagreeEscalation_combinations_performance_val.csv"
        out_df.to_csv(combos_out, index=False)
        print(f"[INFO] Wrote validation combinations → {combos_out}", flush=True)

        # Write explicit ALL-models baseline file (per-run + avg)
        baseline_df = out_df[out_df.get("tag", "") == "ALL_MODELS_BASELINE"].drop(columns=["tag"])
        if not baseline_df.empty:
            avg_base = baseline_df[["recall", "precision", "f2"]].mean().to_dict()
            baseline_df = pd.concat([
                baseline_df,
                pd.DataFrame([{
                    "run": "avg",
                    "combination": "+".join(combo_all),
                    "recall": avg_base["recall"],
                    "precision": avg_base["precision"],
                    "f2": avg_base["f2"],
                    "n_rows": baseline_df["n_rows"].iloc[0],
                    "n_global_escalated": baseline_df["n_global_escalated"].mean()
                }])
            ], ignore_index=True)
            base_out = ENSEMBLE_DIR / f"DisagreeEscalation_all_models_performance_val.csv"
            baseline_df.to_csv(base_out, index=False)
            print(f"[INFO] Wrote ALL-models baseline (val) → {base_out}", flush=True)

    else:  # MODE == "test"
        # Load best combo from validation file
        combos_val = ENSEMBLE_DIR / f"DisagreeEscalation_combinations_performance_val.csv"
        if not combos_val.exists():
            raise FileNotFoundError(f"Missing validation combos file: {combos_val}. Run MODE='val' first.")

        df_val = pd.read_csv(combos_val)
        # pick combo with best avg F2
        df_avg = df_val[df_val["run"] == "avg"].copy()
        if df_avg.empty:
            df_avg = df_val.groupby("combination", as_index=False)[["recall", "precision", "f2"]].mean()
        best_row = df_avg.loc[df_avg["f2"].idxmax()]
        best_combo_str = best_row["combination"]
        best_combo = tuple(best_combo_str.split("+"))
        print(f"[INFO] Loaded best combo from validation: {best_combo_str}", flush=True)

        # Compute and write ALL-models baseline for test
        base_test = perf_df[perf_df.get("tag", "") == "ALL_MODELS_BASELINE"].drop(columns=["tag"])
        if not base_test.empty:
            avg_base = base_test[["recall", "precision", "f2"]].mean().to_dict()
            base_test = pd.concat([
                base_test,
                pd.DataFrame([{
                    "run": "avg",
                    "combination": "+".join(combo_all),
                    "recall": avg_base["recall"],
                    "precision": avg_base["precision"],
                    "f2": avg_base["f2"],
                    "n_rows": base_test["n_rows"].iloc[0],
                    "n_global_escalated": base_test["n_global_escalated"].mean()
                }])
            ], ignore_index=True)
            base_out = ENSEMBLE_DIR / f"DisagreeEscalation_all_models_performance_test.csv"
            base_test.to_csv(base_out, index=False)
            print(f"[INFO] Wrote ALL-models baseline (test) → {base_out}", flush=True)

        # Evaluate only the best combo on each test run
        best_rows = []
        for run in range(1, RUNS + 1):
            print(f"\n[INFO] (test) Evaluating best combo on run {run}", flush=True)
            df = load_merged_light_run(run)
            model_keys = sorted(
                [c.split("_predicted_label")[0] for c in df.columns if c.endswith("_predicted_label")]
            )

            # global disagree / high-end cache for this run
            global_disagree_mask = detect_global_disagreement(df, model_keys)
            df_disagree_global = df.loc[
                global_disagree_mask,
                ["source_id", "source_content", "target_id", "target_content", "label"],
            ].copy()

            high_map = {}
            if len(df_disagree_global) > 0:
                tmp_dir = ENSEMBLE_DIR / "escalation_tmp"
                tmp_dir.mkdir(parents=True, exist_ok=True)
                in_csv = tmp_dir / f"escalate_{DATASET}_{MODE}_run_{run}.csv"
                out_csv = HIGH_END_MODEL_ROOT / HIGH_END_MODEL_SUBDIR / HIGH_END_MODEL_FILENAME_TEMPLATE.format(i=run)
                # ensure files exist (re-create if missing)
                if not out_csv.exists():
                    df_disagree_global.to_csv(in_csv, index=False)
                    run_high_end_model(in_csv, out_csv, run_seed=run, mode=MODE)
                high_df = load_high_end_output(out_csv)
                high_map = {
                    (r["source_id"], r["target_id"], r["label"]): r["highend_predicted_label"]
                    for _, r in high_df.iterrows()
                }

            # evaluate best combo
            y_pred_best = evaluate_combo_using_cache(df, best_combo, high_map)
            r, p, f2 = metrics(df["label"], y_pred_best)
            best_rows.append({
                "run": run,
                "combination": best_combo_str,
                "recall": r,
                "precision": p,
                "f2": f2,
                "n_rows": int(len(df)),
                "n_global_escalated": int(global_disagree_mask.sum())
            })
            print(f"[INFO] Best-combo run {run}: F2={f2:.4f}", flush=True)

            # save per-run predictions for the best combo
            out_pred = ENSEMBLE_DIR / f"DisagreeEscalation_best_{DATASET}_{MODE}_run_{run}.csv"
            out_df = df[["source_id", "target_id", "label"]].copy()
            out_df["predicted_label"] = y_pred_best.values
            out_df.to_csv(out_pred, index=False)

        best_df = pd.DataFrame(best_rows)
        avg_best = best_df[["recall", "precision", "f2"]].mean().to_dict()
        best_df = pd.concat([
            best_df,
            pd.DataFrame([{
                "run": "avg",
                "combination": best_combo_str,
                "recall": avg_best["recall"],
                "precision": avg_best["precision"],
                "f2": avg_best["f2"],
                "n_rows": best_df["n_rows"].iloc[0],
                "n_global_escalated": best_df["n_global_escalated"].mean()
            }])
        ], ignore_index=True)
        best_out = ENSEMBLE_DIR / f"DisagreeEscalation_best_performance_test.csv"
        best_df.to_csv(best_out, index=False)
        print(f"[INFO] Wrote BEST-combo performance (test) → {best_out}", flush=True)