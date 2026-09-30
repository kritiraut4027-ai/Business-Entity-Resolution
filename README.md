# High-Throughput Business Entity Resolution System
### Large-Scale Multilingual Record Linkage Pipeline — Amazon ML Challenge

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![ML Framework](https://img.shields.io/badge/Model-LightGBM-green.svg)](https://lightgbm.readthedocs.io/)
[![Acceleration](https://img.shields.io/badge/Similarity-RapidFuzz%20C%2B%2B-orange.svg)](https://github.com/rapidfuzz/RapidFuzz)
[![Validation F0.5](https://img.shields.io/badge/Macro%20F0.5-0.8659-brightgreen.svg)]()
[![Singleton Accuracy](https://img.shields.io/badge/Singleton%20Acc-100%25-success.svg)]()
[![Zero External Lookups](https://img.shields.io/badge/External%20APIs-None-informational.svg)]()

An end-to-end, high-precision Business Entity Resolution pipeline engineered to link noisy, heterogeneous business entity records across three disparate data sources (~26 million total records) without external knowledge bases, web lookups, or third-party APIs.

---

## 1. Problem Overview & Challenges

Enterprise commercial platforms ingest vendor profiles, public registries, and internal scrapers that lack universal primary keys. The objective is to identify all corresponding entities in **Source 2** and **Source 3** for each reference entity in **Source 1**.

### Real-World Data Challenges
1. **Severe Combinatorial Scale**: Linking 1.73M reference entities against ~10M target records produces a Cartesian product space exceeding **$22.4 \times 10^{12}$ candidate pairs**.
2. **Missing & Truncated Fields**: Addresses are frequently missing (`NaN`), truncated, or lack suite numbers.
3. **Cross-Script Transliteration**: Source 1 contains English names, whereas Sources 2 & 3 frequently contain Indic scripts (Hindi, Devanagari, Telugu) or accented French strings.
4. **Doing Business As (DBA) & Trade Names**: A company's legal registered name (e.g., `Solkeloquo LLC`) often bears zero lexical similarity to its operating storefront name (e.g., `Uptown Pub`).
5. **Multi-Tenant Hubs & Franchises**: Shared commercial addresses house distinct businesses, while franchise chains share identical brand names across different physical addresses.

---

## 2. System Architecture

```
                  +----------------------------------------------+
                  |  Data Ingestion & Unicode Cleaner            |
                  |  (Diacritic-safe category Mn mark filtering) |
                  +----------------------------------------------+
                                         |
                                         v
                  +----------------------------------------------+
                  |  Multi-Strategy Prioritized Blocking         |
                  |  (NUM_LONG -> NTOK -> NUM_WORD -> LOC_PAIR)   |
                  |  Reduces 22.4T space to <200 cands / entity  |
                  |  Candidate Recall: 78.08% (>99.999% pruned)  |
                  +----------------------------------------------+
                                         |
                                         v
                  +----------------------------------------------+
                  |  19 C++ Accelerated Similarity Features      |
                  |  (Rapidfuzz string, token, number, & geo)   |
                  +----------------------------------------------+
                                         |
                                         v
                  +----------------------------------------------+
                  |  Supervised LightGBM Binary Classifier       |
                  |  (150 trees, balanced weights, <5MB artifact) |
                  +----------------------------------------------+
                                         |
                                         v
                  +----------------------------------------------+
                  |  Extreme-Precision Calibration (Cutoff 0.95) |
                  |  Macro F0.5: 0.8659 | Singleton Acc: 100.0%  |
                  +----------------------------------------------+
                                         |
                                         v
                  +----------------------------------------------+
                  |  Fault-Tolerant SQLite Streaming Pipeline    |
                  |  (Partition audits, WAL mode, atomic commits)|
                  +----------------------------------------------+
                         /                                \
                        v                                  v
         output/candidate_pairs.tsv             output/matching_results.tsv
```

---

## 3. Core Technical Components

### 3.1 Normalization & Text Standardization (`src/normalization.py`)
- **Diacritic-Safe Accent Stripping**: Uses Unicode category `Mn` mark filtering to normalize European accents (`Marina École` $\to$ `marina ecole`) without stripping or damaging Indic Unicode scripts (Hindi/Devanagari).
- **DBA & Domain Extraction**: Parses website domains into core business tokens (`maurewilliamscolombier.com` $\to$ `maurewilliamscolombier`) to bridge legal vs. trade name discrepancies.
- **Multinational Legal Suffix Removal**: Handles multi-jurisdiction entities across US (`Inc`, `LLC`, `Corp`), India (`Pvt Ltd`, `Private Limited`, `LLP`), and France (`SARL`, `SAS`, `SA`, `EURL`).
- **Address & Locality Standardization**: Normalizes street types (`st` $\to$ `street`, `ave` $\to$ `avenue`, `twp` $\to$ `township`), state abbreviations (`ny` $\to$ `new york`, `dl` $\to$ `delhi`, `tn` $\to$ `tamil nadu`), and strips leading zeros from plot/house numbers (`0684` $\to$ `684`).

### 3.2 Prioritized Inverted Multi-Index Blocking (`src/blocking.py`)
To avoid missing candidates while strictly bounding candidate generation, the blocking engine indexes multiple deterministic keys with strict prioritization:

```python
def get_blocking_key_priority(key):
    # 0: NUM_LONG  -> PIN codes, phone numbers, unique IDs (>= 5 digits)
    # 1: NTOK      -> Exact significant name tokens (>= 3 chars)
    # 2: NUM_WORD  -> Normalized street number + locality word (e.g. 684_ghaziabad)
    # 3: LOC_PAIR  -> Locality word bigrams (bridges English vs Indic script DBAs)
    # 4: NPREF4    -> 4-character core business name prefix
    # 5: NPREF3    -> 3-character fallback prefix
```

- **Candidate Recall**: **78.08%** across all true ground truth links.
- **Selectivity**: Prunes **>99.999%** of non-matching pairs, producing an average of only **52.77 candidates per entity** (capped at `max_candidates=200`).

### 3.3 Feature Engineering (`src/features.py`)
Computes 19 pairwise similarity features using C++ accelerated `rapidfuzz` on precomputed record tuples:
- **Name Signals (10 features)**: Levenshtein ratio, Core ratio, Token sort ratio, Token set ratio, Partial ratio, Jaro-Winkler, Core Jaro-Winkler, Jaccard token overlap, Common tokens count, Character length delta.
- **Address Signals (8 features)**: Missing address indicator, Levenshtein ratio, Token set ratio, Partial ratio, Token Jaccard overlap, Common address tokens count, Shared number match, Number Jaccard overlap.
- **Geographic Signals (1 feature)**: Exact normalized country match.

### 3.4 Model Selection & Threshold Optimization (`src/train.py`, `src/evaluation.py`)
- **Classifier**: LightGBM (`n_estimators=150`, `learning_rate=0.08`, `num_leaves=31`, `class_weight='balanced'`).
- **Metric Sensitivity**: Evaluated using **Macro-averaged $F_{0.5}$** across Source 1 entities:
  $$F_{0.5} = \frac{1.25 \times \text{Precision} \times \text{Recall}}{0.25 \times \text{Precision} + \text{Recall}}$$
- **The Singleton Challenge**: Singletons (entities with 0 true matches) evaluate to $1.0$ if predicted empty, but immediately drop to $0.0$ if any false positive is predicted. Because precision is weighted $4\times$ as heavily as recall, threshold calibration is vital:
  - At threshold `0.50`: Singleton accuracy is only `82.61%` (Macro $F_{0.5} = 0.7951$).
  - At threshold `0.95`: Singleton accuracy reaches **100.00%** (Macro $F_{0.5} = \mathbf{0.8659}$), eliminating costly false mergers.

---

## 4. Resumable Large-Scale Inference Architecture (`src/run_test_inference_phase1.py`)

Executing pairwise inference across 1.73M entities against ~10M target records requires fault tolerance:
1. **Partition Audit**: Inspects the test set and SQLite database upfront, reporting completed vs. pending entities per country in <5 seconds.
2. **Auto-Skipping**: Fully processed partitions (e.g. France: 259,452/259,452, US: 663,106/663,106) are skipped immediately without reading multi-gigabyte candidate tables.
3. **Atomic Streaming Inserts**: Processes entities in configurable chunks (default: 5,000 entities), writing predictions into SQLite with `INSERT OR REPLACE INTO preds` and transaction commits per batch.
4. **Zero Duplicate Risk**: `s1_id` is indexed as the primary key. If interrupted, re-running the same command automatically resumes from the exact entity where it stopped.

---

## 5. Experiment History & Ablation Studies

Full details are documented in [`docs/experiments.md`](docs/experiments.md).

| Experiment | Description | Candidate Recall | Precision | Recall | Macro $F_{0.5}$ | Singleton Acc | Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **EXP-001** | Baseline Prefix/Token Blocking | 58.16% | 0.7410 | 0.5620 | 0.6942 | 76.5% | Superseded |
| **EXP-002** | Prioritized Multi-Strategy Blocking | **78.08%** | — | — | — | — | Adopted |
| **EXP-003** | Phase 1 Production LightGBM (Cutoff 0.95) | **78.08%** | **0.8982** | **0.7554** | **0.8659** | **100.0%** | **PRODUCTION** |
| **EXP-004** | Phase 2 Hard-Negative Mining Ablation | 78.08% | 0.8841 | 0.6120 | 0.8063 | 100.0% | **REJECTED** |

### Why Phase 2 Hard-Negative Mining Was Rejected
- **Hypothesis**: Mining "hard negatives" (co-located independent businesses with identical addresses but different names, and franchise chains with identical names but different addresses) would improve discrimination.
- **Empirical Result**: Validation Macro $F_{0.5}$ dropped from **0.8659 down to 0.8063** (recall collapsed from 0.7554 to 0.6120).
- **Root Cause**: In real-world data, legitimate matches frequently contain noisy address abbreviations, omitted suite numbers, or minor typographical variations. By over-weighting hard negatives during training, the model learned an overly aggressive rejection boundary that penalized genuine matches.
- **Production Choice**: The Phase 1 model (`models/lgbm_model.pkl`) was retained as production default. `models/lgbm_model_phase2.pkl` is preserved strictly as an experimental benchmark.

---

## 6. How to Reproduce & Run Locally

### A. Environment Setup
```bash
git clone https://github.com/your-username/business-entity-resolution.git
cd business-entity-resolution
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### B. Run Unit Tests
```bash
python tests/test_normalization.py
```

### C. Train Primary Production Model
Trains the primary model, sweeps thresholds, and saves `models/lgbm_model.pkl`:
```bash
python src/train.py
```

To run the experimental Phase 2 hard-negative ablation:
```bash
python src/train.py --phase2
```

### D. Run / Resume Large-Scale Inference
The inference pipeline automatically detects data directories, audits SQLite checkpoints, skips completed partitions, and resumes pending partitions:
```bash
python src/run_test_inference_phase1.py
```

Optional CLI overrides:
```bash
# Process a specific country partition
python src/run_test_inference_phase1.py --country India

# Adjust candidate pool depth or decision threshold
python src/run_test_inference_phase1.py --max-candidates 200 --threshold 0.95

# Directly export TSVs from SQLite without re-running inference
python src/run_test_inference_phase1.py --export-only
```

### E. Validate Output Files
Verifies schema, uniqueness, singleton formatting, and candidate subset constraints:
```bash
python utils/validate_submission.py --matching output/matching_results_phase1.tsv --candidate output/candidate_pairs_phase1.tsv
```

---

## 7. Repository Structure

```
├── docs/
│   └── experiments.md          # Comprehensive ablation log & failure analysis
├── models/
│   ├── lgbm_model.pkl          # Primary production LightGBM model (Macro F0.5: 0.8659)
│   ├── lgbm_model_phase2.pkl   # Experimental hard-negative ablation model
│   └── model_config.json       # Threshold sweep and validation metrics
├── src/
│   ├── blocking.py             # Deterministic prioritized multi-strategy blocking index
│   ├── data_loader.py          # Memory-mapped streaming TSV chunk loaders
│   ├── evaluation.py           # Official Macro F0.5 and singleton evaluation logic
│   ├── features.py             # 19 RapidFuzz C++ pairwise similarity features
│   ├── normalization.py        # Diacritic-safe Unicode, legal suffix, & address normalizer
│   ├── predict.py              # Baseline inference script with sample limits
│   ├── predict_full.py         # Full test inference pipeline
│   ├── run_test_inference_phase1.py # Resumable, partition-aware production inference
│   └── train.py                # Model training, threshold calibration, & ablation suite
├── tests/
│   ├── diagnose_missed.py      # Diagnostic script for false negative candidate pairs
│   ├── inspect_db.py           # SQLite checkpoint coverage inspection utility
│   ├── test_blocking_recall.py # Candidate generation recall benchmark
│   └── test_normalization.py   # Unit test suite for Unicode and text cleaning
├── utils/
│   └── validate_submission.py  # Official competition submission format validator
├── .gitignore                  # Clean exclusion of datasets, DBs, TSVs, and caches
├── README.md                   # System documentation & architectural reference
└── requirements.txt            # Minimal, pinned Python dependencies
```

---

## 8. Summary of Key Engineering Decisions

1. **Deterministic Prioritized Blocking over Brute Force**: Prioritizing high-selectivity numeric keys before broad 3-character prefixes reduced candidate volume from 22.4 trillion pairs to ~52 per entity while boosting recall by +19.92%.
2. **Threshold Tuning for Metric Alignment**: Sweeping the classification cutoff to $0.95$ leveraged the mathematical properties of $F_{0.5}$ and eliminated singleton false alarms (achieving 100% singleton accuracy).
3. **Resumable Database Architecture over In-Memory Buffers**: Streaming through SQLite with write-ahead logging (WAL) enables fault-tolerant resumption of multi-hour batch runs across 1.73 million records.
4. **Honest ML Engineering**: Explicitly documenting why Phase 2 hard-negative mining degraded performance proves an empirical, data-driven approach to production machine learning.
