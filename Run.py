import os
import json
import time
import argparse
import random
from pathlib import Path

import pandas as pd

from openrouter_client import OpenRouterClient
from Evaluation import evaluate_results
import Utils

# ==============================================================
# Helpers
# ==============================================================

def set_seeds(seed):
    try:
        import numpy as np
        random.seed(seed)
        np.random.seed(seed)
    except Exception:
        pass

def decision_to01(d):
    s = str(d).strip().lower()
    if s in ("yes", "true", "1"):
        return 1
    return 0

def ensure_int01(series):
    return pd.to_numeric(series, errors="coerce").fillna(0).astype(int).clip(0,1)

def safe_read_csv(path):
    try:
        return pd.read_csv(path)
    except UnicodeDecodeError:
        import chardet
        raw = Path(path).read_bytes()
        enc = chardet.detect(raw)["encoding"]
        return pd.read_csv(path, encoding=enc)

# ==============================================================
# Data Loading (Only Diverse Strategy)
# ==============================================================

def load_training_data(config):
    ds = config["dataset"]
    emb = config["embedding_model_name"]
    p = f"Embeddings/{ds}/{emb}_train_embeddings.csv"
    print(f"[INFO] Loading training embeddings: {p}")
    return safe_read_csv(p)

def generate_few_shot_examples(config, training_data, n_shots):
    if n_shots <= 0:
        return None

    print(f"[INFO] Generating {n_shots} diverse balanced examples (sum strategy)")
    return Utils.get_diverse_samples(
        training_data,
        strategy="sum",
        num_samples=n_shots
    )

# ==============================================================
# Inference
# ==============================================================

def run_inference_on_df(df, llm_client, system_role, examples, temperature):
    responses = []
    total = len(df)

    for idx, row in df.iterrows():
        prompt = f"(1) {row['source_content']}\n\n(2) {row['target_content']}"

        resp = llm_client.generate_response(
            model=llm_client.model_name,
            system_role=system_role,
            prompt=prompt,
            examples=examples,
            temperature=float(temperature),
        )

        responses.append(resp)
        print(
            f"[INFO] {len(responses)}/{total} "
            f"decision={resp.get('decision')} cost=${resp.get('usd_cost'):.6f}"
        )

    return responses

def save_predictions(config, df, responses, results_directory, n_shots, role_index, repeat):
    # Decide where to save the CSV
    if config.get("output_csv"):
        out_file = Path(config["output_csv"])
        out_file.parent.mkdir(parents=True, exist_ok=True)
    else:
        dir_path = Path(results_directory)
        dir_path.mkdir(parents=True, exist_ok=True)
        out_file = dir_path / f"{n_shots}_shot_P{role_index}_run_{repeat}.csv"

    out = df.copy()
    out["predicted_label"] = [decision_to01(r.get("decision")) for r in responses]
    out["rationale"] = [r.get("rationale") for r in responses]

    if "label" in out.columns:
        out["label"] = ensure_int01(out["label"])

    out.to_csv(out_file, index=False)
    print(f"[INFO] Saved predictions → {out_file}")
    return out_file

# ==============================================================
# Main
# ==============================================================

def main(config):
    print(f"[INFO] Running run_2.py in mode={config['mode']}")
    set_seeds(config["random_state"])

    ds = config["dataset"]
    task = config["task"]

    # -------- Subset mode detection --------
    subset_mode = bool(config.get("input_pairs_csv")) or bool(config.get("output_csv"))
    print(f"[DEBUG] subset_mode={subset_mode}")
    print(f"[DEBUG] input_pairs_csv={config.get('input_pairs_csv')}")
    print(f"[DEBUG] output_csv={config.get('output_csv')}")

    timestamp = str(int(time.time()))
    experiment_id = f"{ds}_{timestamp}_{config['random_state']}"

    if not subset_mode:
        base_dir = Path(f"Results/{task}/{config['model_name']}/diverse/{experiment_id}")
        base_dir.mkdir(parents=True, exist_ok=True)
        with open(base_dir / "experiment_settings.json", "w") as f:
            json.dump(config, f, indent=2)
        print(f"[INFO] Full experiment mode → base_dir={base_dir}")
    else:
        base_dir = None
        print("[INFO] Subset mode → experiment folder disabled")

    # -------- Load data --------
    val_file  = f"Datasets/{ds}/{ds}_val_{task}.csv"
    test_file = f"Datasets/{ds}/{ds}_test_{task}.csv"

    if config.get("input_pairs_csv"):
        df = safe_read_csv(config["input_pairs_csv"])
        print(f"[INFO] Loaded input_pairs_csv={config['input_pairs_csv']} (n={len(df)})")
    else:
        if config["mode"] == "val":
            df = safe_read_csv(val_file)
        elif config["mode"] == "test":
            df = safe_read_csv(test_file)
        else:
            df = safe_read_csv(val_file).sample(n=1, random_state=config["random_state"])
        print(f"[INFO] Loaded default split (n={len(df)})")

    # -------- Load training data (diverse only) --------
    training_data = load_training_data(config)

    # -------- Init OpenRouter client --------
    llm_client = OpenRouterClient(
        mode=config["mode"],
        log_file=f"Log/{ds}_log_{config['random_state']}.md",
    )
    llm_client.model_name = config["model_name"]

    wrote_any = False

    # -------- Roles × Shots × Repeats --------
    for role_index, system_role in enumerate(config["system_roles"], start=1):
        for n_shots in config["n_shots_list"]:

            if not subset_mode:
                role_dir = Path(f"Results/{task}/{config['model_name']}/diverse/{experiment_id}") / f"role_{role_index}_shot_{n_shots}"
                role_dir.mkdir(parents=True, exist_ok=True)
            else:
                role_dir = None

            few_shot_examples = generate_few_shot_examples(config, training_data, n_shots)

            for repeat in range(1, config["num_repeats"] + 1):
                print(f"[INFO] Role={role_index} | Shots={n_shots} | Run={repeat}")

                responses = run_inference_on_df(
                    df,
                    llm_client,
                    system_role,
                    few_shot_examples,
                    config["model_temperature"],
                )

                out_path = save_predictions(
                    config, df, responses,
                    str(role_dir) if role_dir else "",
                    n_shots, role_index, repeat
                )
                wrote_any = True

            # Evaluate only in full experiment mode
            if not subset_mode and config["mode"] != "dev":
                try:
                    evaluate_results(str(role_dir))
                except Exception as e:
                    print(f"[WARN] Evaluation failed: {e}")

    # ==================================================
    # COST SUMMARY
    # ==================================================
    print("\n========== COST SUMMARY ==========")
    print(f"Input tokens       : {llm_client.total_input_tokens}")
    print(f"Output tokens      : {llm_client.total_output_tokens}")
    print(f"Total tokens       : {llm_client.total_tokens}")
    print(f"Approx. total cost : ${llm_client.total_cost_usd:.6f}")
    print("==================================")

    summary = {
        "input_tokens": int(llm_client.total_input_tokens),
        "output_tokens": int(llm_client.total_output_tokens),
        "total_tokens": int(llm_client.total_tokens),
        "total_cost_usd": float(llm_client.total_cost_usd),
        "model_name": config["model_name"],
        "dataset": ds,
        "mode": config["mode"],
        "experiment_id": experiment_id,
        "timestamp": timestamp,
    }

    # -------- CASE 1: Full experiment mode --------
    if not subset_mode:
        cost_path = base_dir / "cost_summary.json"
        with open(cost_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"[INFO] Cost summary saved → {cost_path}")

    # -------- CASE 2: Subset mode — save next to output_csv --------
    else:
        if config.get("output_csv"):
            out_dir = Path(config["output_csv"]).parent
            cost_path = out_dir / "cost_summary_escalation.json"
            print(f"[INFO] Subset mode: writing cost summary → {cost_path}")
            with open(cost_path, "w") as f:
                json.dump(summary, f, indent=2)
        else:
            print("[WARN] Subset mode: output_csv is missing, cannot write cost_summary_escalation.json")

    if not wrote_any:
        print("[WARN] No predictions produced.")
    print("[INFO] run_2.py completed.")

# ==============================================================
# CLI
# ==============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Experiment (Diverse Strategy Only)")

    parser.add_argument("--inference_model_name", type=str, required=True)
    parser.add_argument("--inference_model_temperature", type=float, required=True)
    parser.add_argument("--task", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--system_roles", nargs="+", type=str, required=True)

    parser.add_argument("--n_shots", nargs="+", type=int, required=True)
    parser.add_argument("--selection_strategy", type=str, default="diverse")
    parser.add_argument("--diversity_strategy", type=str, default="sum")
    parser.add_argument("--balanced", action="store_true")

    parser.add_argument("--mode", type=str, default="test", choices=["dev", "val", "test"])
    parser.add_argument("--num_repeats", type=int, default=1)
    parser.add_argument("--random_seed", type=int, required=True)

    parser.add_argument("--embedding_model_name", type=str, default="all-mpnet-base-v2")
    parser.add_argument("--input_pairs_csv", type=str)
    parser.add_argument("--output_csv", type=str)

    args = parser.parse_args()

    config = {
        "mode": args.mode,
        "num_repeats": args.num_repeats,
        "model_name": args.inference_model_name,
        "model_temperature": args.inference_model_temperature,
        "embedding_model_name": args.embedding_model_name,
        "task": args.task,

        "n_shots_list": args.n_shots,
        "selection_strategy": args.selection_strategy,
        "diversity_strategy": args.diversity_strategy,
        "balanced": args.balanced,

        "random_state": args.random_seed,
        "dataset": args.dataset,
        "system_roles": args.system_roles,

        "input_pairs_csv": args.input_pairs_csv,
        "output_csv": args.output_csv,
    }

    print("[INFO] CONFIG:")
    print(json.dumps(config, indent=2))

    main(config)