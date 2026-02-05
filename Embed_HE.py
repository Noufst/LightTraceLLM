"""
Task 3 — Similarity → High-End Escalation 
"""

import os
import sys
import subprocess
from pathlib import Path
import argparse

import numpy as np
import pandas as pd

# --------------------------
# Static Configuration
# --------------------------
RUN_1_PY = "Run.py"
HIGH_END_MODEL_NAME = "openai/gpt-4o-mini"

# ID column candidates
SRC_ID_CAND = ["source_id", "source_ID", "UC"]
TGT_ID_CAND = ["target_id", "target_ID", "TC", "ID"]


# --------------------------
# Utilities
# --------------------------
def norm_id(x):
    return str(x).strip()


def rename_ids(df):
    df = df.copy()
    for c in SRC_ID_CAND:
        if c in df.columns:
            if c != "source_id":
                df.rename(columns={c: "source_id"}, inplace=True)
            break
    else:
        raise ValueError("No valid source_id")

    for c in TGT_ID_CAND:
        if c in df.columns:
            if c != "target_id":
                df.rename(columns={c: "target_id"}, inplace=True)
            break
    else:
        raise ValueError("No valid target_id")

    return df


def load_split_from_parquet(path, split):
    df = pd.read_parquet(path)
    df = df[df["split"] == split].reset_index(drop=True)
    df = rename_ids(df)
    df["label"] = pd.to_numeric(df["label"], errors="coerce").fillna(0).astype(int).clip(0, 1)
    df["source_id"] = df["source_id"].apply(norm_id)
    df["target_id"] = df["target_id"].apply(norm_id)
    df["src_emb"] = df["src_emb"].apply(lambda x: np.asarray(x, dtype=np.float32))
    df["tgt_emb"] = df["tgt_emb"].apply(lambda x: np.asarray(x, dtype=np.float32))
    df["sim"] = [float(np.dot(a, b)) for a, b in zip(df["src_emb"], df["tgt_emb"])]
    return df


def metrics(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    tp = ((y_true == 1) & (y_pred == 1)).sum()
    fp = ((y_true == 0) & (y_pred == 1)).sum()
    fn = ((y_true == 1) & (y_pred == 0)).sum()
    precision = tp / (tp + fp) if (tp + fp) else 0
    recall = tp / (tp + fn) if (tp + fn) else 0
    beta2 = 4.0
    denom = beta2 * precision + recall
    f2 = (1 + beta2) * precision * recall / denom if denom else 0
    return recall, precision, f2


def system_role(dataset):
    ds = dataset.lower()

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


def call_high_end(input_csv, output_csv, mode, dataset):
    args = [
        sys.executable, "-u", RUN_1_PY,
        "--mode", mode,
        "--num_repeats", "1",
        "--inference_model_name", HIGH_END_MODEL_NAME,
        "--inference_model_temperature", "0",
        "--embedding_model_name", "all-mpnet-base-v2",
        "--task", "TLC",
        "--n_shots", "2",
        "--balanced",
        "--selection_strategy", "diverse",
        "--diversity_strategy", "sum",
        "--random_seed", "1",
        "--dataset", dataset,
        "--system_roles", system_role(dataset),
        "--input_pairs_csv", str(input_csv),
        "--output_csv", str(output_csv),
    ]
    print("[INFO] Calling high-end:", " ".join(args))
    p = subprocess.Popen(
        args,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    for line in p.stdout:
        print("[HIGH-END]", line.rstrip())
    p.wait()
    if p.returncode != 0:
        raise RuntimeError("High-end failed")


def load_high_end_output(path):
    df = pd.read_csv(path)
    df = rename_ids(df)
    df["predicted_label"] = pd.to_numeric(df["predicted_label"], errors="coerce").fillna(0).astype(int).clip(0, 1)
    return df[["source_id", "target_id", "predicted_label"]].rename(
        columns={"predicted_label": "highend_predicted_label"}
    )


# --------------------------
# MAIN
# --------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Embed-HE Similarity → High-End Escalation (offline).")
    parser.add_argument("--dataset", required=True, help="Dataset name, e.g., CM1_NASA, EasyClinic_UC_TC, EasyClinic_UC_ID, CCHIT")
    parser.add_argument("--mode", required=True, choices=["val", "test"], help="Mode: val or test")
    parser.add_argument("--runs", type=int, required=True, help="Number of runs (e.g., 25)")

    args = parser.parse_args()

    DATASET = args.dataset
    MODE = args.mode
    RUNS = args.runs

    print(f"[INFO] Dataset = {DATASET}")
    print(f"[INFO] Mode    = {MODE}")
    print(f"[INFO] RUNS    = {RUNS}")

    # Paths
    PARQUET_PATH = Path(f"Datasets/{DATASET}/{DATASET}_embeddings.parquet")

    OUT_DIR = Path(f"Results/TLC/_ensemble/Embed-HE/{DATASET}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    HIGH_OUT_DIR = Path(
        f"Results/TLC/{HIGH_END_MODEL_NAME}/diverse/{DATASET}_high_end_similarity_escalation/{MODE}/role_1_shot_2"
    )
    HIGH_OUT_DIR.mkdir(parents=True, exist_ok=True)

    HIGH_END_MODEL_FILENAME_TEMPLATE = f"2_shot_P1_run_{{i}}_{MODE}.csv"

    split = "val" if MODE == "val" else "test"
    df = load_split_from_parquet(PARQUET_PATH, split)
    print(f"[INFO] Loaded {split}: {len(df)} rows")
    print(f"[INFO] sim stats: min={df.sim.min():.4f} max={df.sim.max():.4f} mean={df.sim.mean():.4f}")

    results = []

    # ==============================================================
    # VAL MODE — now with DATA-DRIVEN coarse thresholds
    # ==============================================================
    if MODE == "val":

        # ---- DATA-DRIVEN THRESHOLDS ----
        pos_sims = df[df["label"] == 1]["sim"]
        q10 = float(pos_sims.quantile(0.10))
        q90 = float(pos_sims.quantile(0.90))

        # 5 coarse thresholds between 10–90% positive region
        THRESHOLDS = np.linspace(q10, q90, num=5)
        print(f"[INFO] Data-driven coarse thresholds = {THRESHOLDS}")

        min_th = float(min(THRESHOLDS))

        # =======================================================
        # MAIN VALIDATION LOOP
        # =======================================================
        for run in range(1, RUNS + 1):
            print(f"\n[INFO] === VAL run {run} ===")

            # Superset escalation only for rows >= min_th
            mask_sup = df["sim"] >= min_th
            df_esc_sup = df.loc[
                mask_sup,
                ["source_id", "source_content", "target_id", "target_content", "label"],
            ].copy()
            print(f"[INFO] Superset escalated for min_th={min_th:.4f}: {len(df_esc_sup)}")

            high_map = {}
            if len(df_esc_sup) > 0:
                tmp = OUT_DIR / "escalation_tmp"
                tmp.mkdir(parents=True, exist_ok=True)
                in_csv = tmp / f"escalate_{DATASET}_{split}_run{run}_min{min_th:.4f}.csv"
                out_csv = HIGH_OUT_DIR / HIGH_END_MODEL_FILENAME_TEMPLATE.format(i=run)
                df_esc_sup.to_csv(in_csv, index=False)

                call_high_end(in_csv, out_csv, MODE, DATASET)
                hd = load_high_end_output(out_csv)
                high_map = {(r.source_id, r.target_id): r.highend_predicted_label for r in hd.itertuples()}

            y_true = df["label"].values

            for th in THRESHOLDS:
                mask = df["sim"] >= th
                n_esc_th = int(mask.sum())
                print(f"[INFO] Threshold {th:.4f}: n_escalated={n_esc_th}")

                preds = []
                fallback = 0

                for r in df.itertuples():
                    if r.sim < th:
                        preds.append(0)
                    else:
                        v = high_map.get((r.source_id, r.target_id))
                        if v is None:
                            fallback += 1
                            preds.append(0)
                        else:
                            preds.append(v)

                rec, prec, f2 = metrics(y_true, preds)

                results.append({
                    "mode": "val",
                    "run": run,
                    "threshold": float(th),
                    "recall": rec,
                    "precision": prec,
                    "f2": f2,
                    "fallback_misses": fallback,
                    "n_escalated": n_esc_th
                })

        res_df = pd.DataFrame(results)

        # Averages
        avg = res_df.groupby("threshold", as_index=False)[["recall", "precision", "f2", "n_escalated"]].mean()
        avg["run"] = "avg"

        best_row = avg.loc[avg["f2"].idxmax()]
        best_th = float(best_row["threshold"])

        # Add threshold metadata (only stored in avg rows)
        avg["q10_sim"] = q10
        avg["q90_sim"] = q90
        avg["coarse_thresholds"] = ",".join([f"{t:.4f}" for t in THRESHOLDS])

        # Build final output
        res_df["q10_sim"] = None
        res_df["q90_sim"] = None
        res_df["coarse_thresholds"] = None

        out_df = pd.concat([res_df, avg], ignore_index=True)
        out_df["best_threshold"] = ""
        out_df.loc[out_df["threshold"] == best_th, "best_threshold"] = "YES"

        # Save
        out_file = OUT_DIR / "SimilarityEscalation_val.csv"
        out_df.to_csv(out_file, index=False)

        print(f"[INFO] Saved → {out_file}")
        print(f"[INFO] Best threshold = {best_th:.4f}")
        print(f"[INFO] q10={q10:.4f}, q90={q90:.4f}")
        print(f"[INFO] coarse_thresholds={THRESHOLDS}")

    # ==============================================================
    # TEST MODE
    # ==============================================================
    else:
        val_file = OUT_DIR / "SimilarityEscalation_val.csv"
        if not val_file.exists():
            raise FileNotFoundError("Run val first.")

        vdf = pd.read_csv(val_file)
        vavg = vdf[vdf["run"] == "avg"]
        if vavg.empty:
            vavg = vdf.groupby("threshold", as_index=False)[["recall", "precision", "f2"]].mean()

        best_th = float(vavg.loc[vavg["f2"].idxmax(), "threshold"])
        print(f"[INFO] Using best_th from val: {best_th:.4f}")

        for run in range(1, RUNS + 1):
            print(f"\n[INFO] === TEST run {run} ===")

            mask = df["sim"] >= best_th
            n_esc_th = int(mask.sum())
            print(f"[INFO] n_escalated={n_esc_th} for th={best_th:.4f}")

            df_esc = df.loc[
                mask,
                ["source_id", "source_content", "target_id", "target_content", "label"],
            ]

            high_map = {}
            if len(df_esc) > 0:
                tmp = OUT_DIR / "escalation_tmp"
                tmp.mkdir(parents=True, exist_ok=True)
                in_csv = tmp / f"escalate_{DATASET}_{split}_run{run}_th{best_th:.4f}.csv"
                out_csv = HIGH_OUT_DIR / HIGH_END_MODEL_FILENAME_TEMPLATE.format(i=run)
                df_esc.to_csv(in_csv, index=False)

                call_high_end(in_csv, out_csv, MODE, DATASET)
                hd = load_high_end_output(out_csv)
                high_map = {(r.source_id, r.target_id): r.highend_predicted_label for r in hd.itertuples()}

            y_true = df["label"].values
            preds = []
            fallback = 0

            for r in df.itertuples():
                if r.sim < best_th:
                    preds.append(0)
                else:
                    v = high_map.get((r.source_id, r.target_id))
                    if v is None:
                        fallback += 1
                        preds.append(0)
                    else:
                        preds.append(v)

            rec, prec, f2 = metrics(y_true, preds)

            results.append({
                "mode": "test",
                "run": run,
                "threshold": best_th,
                "recall": rec,
                "precision": prec,
                "f2": f2,
                "fallback_misses": fallback,
                "n_escalated": n_esc_th
            })

        out_df = pd.DataFrame(results)
        avg = out_df[["recall", "precision", "f2", "n_escalated"]].mean().to_dict()

        out_df = pd.concat([
            out_df,
            pd.DataFrame([{
                "mode": "test",
                "run": "avg",
                "threshold": best_th,
                "recall": avg["recall"],
                "precision": avg["precision"],
                "f2": avg["f2"],
                "fallback_misses": out_df["fallback_misses"].mean(),
                "n_escalated": avg["n_escalated"],
            }])
        ])

        out = OUT_DIR / "SimilarityEscalation_test.csv"
        out_df.to_csv(out, index=False)
        print(f"[INFO] Saved → {out}")