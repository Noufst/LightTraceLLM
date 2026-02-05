# Toward Cost-Efficient Automated Requirements Traceability with Large Language Models — Replication Package Guide

## Overview

This file provides step-by-step instructions for configuring and running LLM-based cost-efficient traceability experiments using the provided Python script. The script enables the setup and execution of experiments with flexible configuration options via command-line arguments.

This repository contains four runnable files:

- **TraceLLM**: Run.py (Used also as ICL of light LLMs)
- **LightLLM_MV**: LightLLM_MV.py
- **LightLLM_HE**: LightLLM_HE.py
- **Embed_HE**: Embed_HE.py

## Requirements

1. Python: Ensure that Python is installed on your system (preferably version 3.7 or higher).
2. Required Libraries: Please check requirements.txt
3. Additional Files: 
	- Utils.py: Contains utility functions like get_random_samples.
	- openrouter_client.py: Interface for interacting with the language models.
	- Evaluation.py: Contains the evaluate_results function for evaluating model responses.
4. Datasets: Ensure you have the required datasets in the correct file paths.

## **Setup Instructions**
1. **Directory Structure**:
    Ensure your project has the following structure:

    ```
    project_directory/
    ├── Datasets/
    │   └── CM1_NASA/
    │       ├── CM1_NASA_train_TLC.csv
    │       ├── CM1_NASA_val_TLC.csv
    │       ├── CM1_NASA_test_TLC.csv
    │       ├── CM1_NASA_embeddings.parquet
    │   └── EasyClinic_UC_TC/
    │       ├── EasyClinic_UC_TC_train_TLC.csv
    │       ├── EasyClinic_UC_TC_val_TLC.csv
    │       ├── EasyClinic_UC_TC_test_TLC.csv
    │       ├── EasyClinic_UC_TC.parquet
    │   └── EasyClinic_UC_ID/
    │       ├── EasyClinic_UC_ID_train_TLC.csv
    │       ├── EasyClinic_UC_ID_val_TLC.csv
    │       ├── EasyClinic_UC_ID_test_TLC.csv
    │       ├── EasyClinic_UC_ID.parquet
    │   └── CCHIT/
    │       ├── CCHIT_train_TLC.csv
    │       ├── CCHIT_val_TLC.csv
    │       ├── CCHIT_test_TLC.csv
    │       ├── CCHIT.parquet
    ├── Results/
    ├── Utils.py
    ├── LLMClient.py
    ├── LightLLM_MV.py
    ├── LightLLM_HE.py
    ├── Embed_HE.py
    ├── Evaluation.py
    └── Run.py
    ```

2. **Environment Setup**:
	Install required packages if they aren’t already:
     ```bash
     pip install <package>
     ```

3. **API Key Setup**:
	Set the API key(s) required by Run.py (example):

`export OPENROUTER_API_KEY="..."`


## TraceLLM / ICL (Run.py)

Generate base predictions for each Light LLM.

Example:

```python Run.py \
  --inference_model_name mistralai/mistral-7b-instruct \
  --inference_model_temperature 0.0 \
  --task TLC \
  --dataset EasyClinic_UC_TC \
  --system_roles "You are a helpful evaluator." \
  --n_shots 2 \
  --mode val \
  --num_repeats 25 \
  --random_seed 1
```

Repeat for other models and seeds.

Outputs are written under Results/TLC/ and consumed by the ensemble methods.


## 3. LightLLM_MV.py 

Input: per-run ICL CSVs from multiple Light LLMs.

Edit the top of LightLLM_MV.py:
- DATASET
- MODE ("val" then "test")
- SEED_FOLDERS

Run:

`python LightLLM_MV.py`

Outputs:
- Merged runs: `Datasets/<DATASET>_Light_LLMs/`
- Ensemble results: `Results/TLC/_ensemble/LightLLM_MV/<DATASET>/`


## 4. LightLLM_HE.py

Requires merged outputs from LightLLM_MV.py.

Validation:

```python LightLLM_HE.py \
  --dataset EasyClinic_UC_TC \
  --mode val \
  --runs 25 \
  --min_combo 3
```

Test:

```python LightLLM_HE.py \
  --dataset EasyClinic_UC_TC \
  --mode test \
  --runs 25 \
  --min_combo 3
```

Outputs are written to:

`Results/TLC/_ensemble/LightLLM_HE/<DATASET>/`

Run.py is invoked automatically for high-end escalation.


## 5. Embed_HE.py

Requires:

`Datasets/<DATASET>/<DATASET>_embeddings.parquet`

Validation:

```python Embed_HE.py \
  --dataset EasyClinic_UC_TC \
  --mode val \
  --runs 25
```

Test:

```python Embed_HE.py \
  --dataset EasyClinic_UC_TC \
  --mode test \
  --runs 25
```

Outputs are written to:

`Results/TLC/_ensemble/Embed-HE/<DATASET>/`