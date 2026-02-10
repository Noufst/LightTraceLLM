# Cost-Efficient Automated Requirements Traceability with LLMs

> **Research Implementation**: A replication package for cost-efficient requirements traceability using Large Language Models.

## Overview

This research implements and compares four approaches to automated trace link detection:
- **TraceLLM**: Baseline using in-context learning with high-end LLMs
  → [`Run.py`](Run.py)
- **LightLLM-MV**: Majority voting ensemble of lightweight LLMs
  → [`LightLLM_MV.py`](LightLLM_MV.py)
- **LightLLM-HE**: Lightweight LLMs disagreement-based high-end LLM escalation  → [`LightLLM_HE.py`](LightLLM_HE.py)
- **Embed-HE**: Embedding similarity-based high-end LLM escalation
  → [`Embed_HE.py`](Embed_HE.py)

## Quick Start

### Prerequisites
- Python 3.9+ ([download](https://www.python.org/downloads/))
- OpenRouter API key ([get one here](https://openrouter.ai/keys))
- ~500MB disk space for datasets

### Installation

1. **Clone the repository**:
```bash
git clone <GitHub link>
cd TraceLLM_Efficient
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **Download datasets** (see [Datasets](#datasets) section below):
```bash
# Option A: Automated download (recommended)
python download_datasets.py

# Option B: Manual download from Figshare
# Download from: https://figshare.com/s/b212efcec17eaa0c70dd 
# Extract to project root
```

4. **Set up API key**:
```bash
# Copy environment template
cp .env.example .env

# Edit .env and add your API key
# OPENROUTER_API_KEY=your_actual_key_here
```

5. **Run an experiment** (see [Run Expirements](#run-expirements) section below)


## Project Structure

```
/Cost-Efficient-LLM-Based-Requirements-Traceability
├── Run.py              # Main ICL code and baseline implementation
├── LightLLM_MV.py     # Majority voting ensemble of lightweight LLMs
├── LightLLM_HE.py     # Lightweight LLMs disagreement-based high-end LLM escalation
├── Embed_HE.py        # Embedding similarity-based high-end LLM escalation
├── Utils.py           # Utility functions
├── tracellm_utils.py  # Shared utilities (metrics, prompts)
├── Evaluation.py      # Performance metrics
├── Statistical_Test.py # Statistical analysis
├── openrouter_client.py # LLM API client
├── config.yaml        # Configuration settings
├── .env.example       # Environment template
├── requirements.txt   # Python dependencies
├── Datasets/          # Input datasets (NOT in git - too large)
└── Results/           # Experimental outputs (NOT in git)
```

## Datasets

**⚠️ Important**: Dataset and embedding files are NOT included in this repository due to their large size (~500MB). They are excluded via `.gitignore`.

### Download Datasets

**Option 1: Automated Download (Recommended)**
```bash
python download_datasets.py
```

**Option 2: Manual Download from Figshare**

📊 **Dataset Repository:** [Figshare](https://figshare.com/s/b212efcec17eaa0c70dd)


1. Download `TraceLLM_Datasets.zip` from the Figshare link above
2. Extract the archive to the project root directory
3. Verify the structure matches below

**Option 3: Use Your Own Datasets**
- Place your datasets in `Datasets/[dataset_name]/` following the structure below
- Generate embeddings using your preferred embedding model (e.g., all-mpnet-base-v2)
- Update paths in scripts if needed

### Expected Directory Structure

After downloading datasets, your directory should look like:
```
/Cost-Efficient-LLM-Based-Requirements-Traceability
├── Datasets/
│   ├── CCHIT/
│   │   ├── CCHIT_train_TLC.csv
│   │   ├── CCHIT_val_TLC.csv
│   │   ├── CCHIT_test_TLC.csv
│   │   └── CCHIT_embeddings.parquet
│   ├── CM1_NASA/
│   │   ├── CM1_NASA_train_TLC.csv
│   │   ├── CM1_NASA_val_TLC.csv
│   │   ├── CM1_NASA_test_TLC.csv
│   │   └── CM1_NASA_embeddings.parquet
│   ├── EasyClinic_UC_TC/
│   │   └── [similar structure]
│   └── EasyClinic_UC_ID/
│       └── [similar structure]
└── Embeddings/
    ├── CCHIT/
    │   ├── all-mpnet-base-v2_test_embeddings.csv
    │   └── all-mpnet-base-v2_train_embeddings.csv
    └── [other datasets...]
```

### Dataset Information

The datasets contain trace links between software artifacts:
- **CCHIT**: Requirements to regulations
- **CM1_NASA**: High-level requirements to design elements
- **EasyClinic_UC_TC**: Use cases to test cases
- **EasyClinic_UC_ID**: Use cases to interaction diagrams

Each dataset includes:
- Train/validation/test splits for reproducibility
- Pre-computed embeddings (all-mpnet-base-v2 model)
- Ground truth trace links (binary labels)

## Run Expirements

### TraceLLM / ICL (Run.py)

Generate predictions for TraceLLM and base predictions for each Light LLM.

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


### LightLLM_MV.py 

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


### LightLLM_HE.py

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


### Embed_HE.py

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