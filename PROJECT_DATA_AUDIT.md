# Project Data Audit

## 1. Dataset Overview

The dataset consists of business entities from three sources across train and test splits. The total uncompressed size is approximately 2.4 GB.

### Train Split
* **train_source1.tsv**: 2,206,821 rows (~200 MB)
* **train_source2.tsv**: 5,034,616 rows (~466 MB)
* **train_source3.tsv**: 5,285,603 rows (~480 MB)
* **train_ground_truth.tsv**: 2,206,821 rows (~121 MB)

### Test Split
* **test_source1.tsv**: 1,732,544 rows (~166 MB)
* **test_source2.tsv**: 4,887,273 rows (~485 MB)
* **test_source3.tsv**: 5,082,316 rows (~482 MB)

## 2. Columns & Data Types
The data files (source 1, 2, and 3) contain the following columns:
- `entity_id` (string)
- `business_name` (string, contains unicode like Hindi text)
- `business_address` (string, can be NaN/missing)
- `country` (string, e.g., 'US', 'India', 'France')

The `train_ground_truth.tsv` contains:
- `source1_entity_id` (string)
- `matched_entity_ids` (string, comma-separated IDs or empty)

## 3. Findings
* The scale of the data is massive (over 26 million rows in total). A full Cartesian join is impossible.
* The country is open-set. It includes 'US', 'India', and 'France' based on initial sampling.
* Data contains unicode characters (e.g., Hindi characters for India).
* Missing values are present in `business_address` (e.g., NaN).

## 4. Implementation Strategy
* **Data Loading**: We will use memory-mapped chunking or lazy loading (Dask/Polars or Pandas chunks) and only read necessary columns.
* **Normalization**: Requires unicode handling, lowercasing, and preserving meaningful tokens while ignoring basic punctuation.
* **Blocking**: We will use TF-IDF or character N-gram token blocking on Name + Country to generate candidate pairs efficiently.
