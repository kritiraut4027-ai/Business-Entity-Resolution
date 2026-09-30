import os
import sys
import time
import argparse
import sqlite3
import pickle
import gc
import functools
import numpy as np
import pandas as pd
from rapidfuzz import distance, fuzz

print = functools.partial(print, flush=True)

sys.path.insert(0, os.path.dirname(__file__))

from normalization import normalize_business_name, normalize_address, normalize_country
from blocking import BlockingIndex
from features import compute_pair_features

DEFAULT_CANDIDATE_LIMIT = 200
DEFAULT_THRESHOLD = 0.95
DEFAULT_BATCH_SIZE = 5000
DEFAULT_BUCKET_SIZE = 800


def find_default_data_dir():
    candidates = [
        r"c:\Users\USER\Downloads\ML Challenge",
        r"G:\My Drive\ML Challenge",
        r"./data",
        r"../data",
    ]
    for path in candidates:
        if os.path.exists(path) and os.path.exists(os.path.join(path, "test_source1.tsv")):
            return path
    return candidates[0]


def init_database(db_path):
    """
    Initializes the SQLite prediction store with primary key indexing and write-ahead logging (WAL).
    """
    os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("PRAGMA journal_mode = WAL")
    cursor.execute("PRAGMA synchronous = NORMAL")
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS preds (
            s1_id TEXT PRIMARY KEY,
            candidates TEXT,
            matches TEXT
        )
    """)
    conn.commit()
    conn.close()


def get_existing_ids(db_path):
    """
    Loads all previously processed source1 IDs from the SQLite database.
    """
    if not os.path.exists(db_path):
        return set()
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT s1_id FROM preds")
    existing = set(r[0] for r in cursor.fetchall())
    conn.close()
    return existing


def audit_partitions(data_dir, existing_ids):
    """
    Fast audit of all country partitions in test_source1.tsv against the checkpoint DB.
    """
    s1_path = os.path.join(data_dir, "test_source1.tsv")
    if not os.path.exists(s1_path):
        raise FileNotFoundError(f"Missing {s1_path}")

    stats = {}
    for chunk in pd.read_csv(s1_path, sep="\t", usecols=["entity_id", "country"], chunksize=250000):
        for country, group in chunk.groupby("country"):
            c_name = str(country).strip()
            if c_name not in stats:
                stats[c_name] = {"total": 0, "completed": 0}
            stats[c_name]["total"] += len(group)
            stats[c_name]["completed"] += len(set(group["entity_id"]).intersection(existing_ids))
    return stats


def run_country_inference(country, data_dir, model, threshold, db_path, max_candidates, batch_size, existing_ids, limit=None):
    """
    Executes resumable entity resolution inference for a single country partition.
    Memory-optimized: stores lightweight (name, address) tuples for candidate pool.
    """
    print(f"\n============================================================")
    print(f"  PARTITION INFERENCE: {country.upper()}")
    print(f"============================================================")
    t0 = time.time()

    # 1. Load candidate pool for this country only into BlockingIndex and memory store
    cand_records = {}  # eid -> (name, addr)
    indexer = BlockingIndex(max_bucket_size=DEFAULT_BUCKET_SIZE)

    for s_file in ["test_source2.tsv", "test_source3.tsv"]:
        path = os.path.join(data_dir, s_file)
        print(f"Loading {s_file} for country {country}...")
        t_load = time.time()
        for chunk in pd.read_csv(path, sep="\t", chunksize=250000, dtype=str):
            sub = chunk[chunk["country"].astype(str).str.strip().str.lower() == country.lower()]
            eids = sub["entity_id"].tolist()
            b_names = [str(x) if pd.notna(x) else "" for x in sub["business_name"]]
            b_addrs = [str(x) if pd.notna(x) else "" for x in sub["business_address"]]
            for eid, b_name, b_addr in zip(eids, b_names, b_addrs):
                cand_records[eid] = (b_name, b_addr)
                indexer.add_record(eid, b_name, b_addr, country)
        print(f"  Loaded {len(cand_records):,} candidate records in {time.time() - t_load:.1f}s")

    print(f"Candidate index ready: {len(cand_records):,} total target pool records.")

    # 2. Open DB connection for streaming inserts
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    s1_path = os.path.join(data_dir, "test_source1.tsv")
    s1_buffer = []
    processed_count = 0
    total_matches = 0
    total_cands = 0
    t_start = time.time()

    def process_s1_batch(batch):
        nonlocal total_matches, total_cands, processed_count
        if not batch:
            return

        pair_features = []
        pair_meta = []  # (batch_idx, cid)
        entity_cands = []
        entity_matches = [[] for _ in range(len(batch))]

        for b_idx, (s1_id, s1_name, s1_addr, s1_country) in enumerate(batch):
            # Prioritized deterministic blocking (up to max_candidates)
            cands = indexer.get_candidates_for_record(s1_name, s1_addr, s1_country, max_candidates=max_candidates)
            valid_cands = [c for c in cands if c in cand_records]
            entity_cands.append(valid_cands)
            total_cands += len(valid_cands)

            s1_name_str = str(s1_name or "")
            s1_addr_str = str(s1_addr or "")

            for cid in valid_cands:
                c_name, c_addr = cand_records[cid]

                # Fast Heuristic Pre-filter: Jaro-Winkler >= 0.70 or address token set >= 70
                jw = distance.JaroWinkler.similarity(s1_name_str, c_name)
                if jw >= 0.70 or bool(s1_addr_str and c_addr and fuzz.token_set_ratio(s1_addr_str, c_addr) >= 70):
                    feats = compute_pair_features(s1_name, s1_addr, s1_country, c_name, c_addr, country)
                    pair_features.append(feats)
                    pair_meta.append((b_idx, cid))

        # Batched LightGBM scoring
        if pair_features:
            probs = model.predict_proba(np.array(pair_features))[:, 1]
            for i, p in enumerate(probs):
                if p >= threshold:
                    b_idx, cid = pair_meta[i]
                    entity_matches[b_idx].append(cid)

        # Batch database write
        db_rows = []
        for (s1_id, _, _, _), c_list, m_list in zip(batch, entity_cands, entity_matches):
            db_rows.append((s1_id, ",".join(c_list), ",".join(m_list)))
            total_matches += len(m_list)
            processed_count += 1
            existing_ids.add(s1_id)

        cursor.executemany("INSERT OR REPLACE INTO preds (s1_id, candidates, matches) VALUES (?, ?, ?)", db_rows)
        conn.commit()

    # 3. Stream test_source1 for this country partition
    for chunk in pd.read_csv(s1_path, sep="\t", chunksize=50000, dtype=str):
        c_chunk = chunk[chunk["country"].astype(str).str.strip().str.lower() == country.lower()]
        eids = c_chunk["entity_id"].tolist()
        b_names = [str(x) if pd.notna(x) else "" for x in c_chunk["business_name"]]
        b_addrs = [str(x) if pd.notna(x) else "" for x in c_chunk["business_address"]]
        b_countries = c_chunk["country"].tolist()
        for s1_id, s1_name, s1_addr, s1_country in zip(eids, b_names, b_addrs, b_countries):
            if s1_id in existing_ids:
                continue

            s1_buffer.append((s1_id, s1_name, s1_addr, s1_country))

            if limit is not None and (processed_count + len(s1_buffer)) >= limit:
                break

            if len(s1_buffer) >= batch_size:
                process_s1_batch(s1_buffer)
                s1_buffer = []
                elapsed = time.time() - t_start
                rate = processed_count / max(1.0, elapsed)
                print(f"[{country}] Processed {processed_count:,} entities | Cands: {total_cands:,} | Matches: {total_matches:,} | Rate: {rate:.1f} ent/s")

        if limit is not None and (processed_count + len(s1_buffer)) >= limit:
            break

    if s1_buffer:
        process_s1_batch(s1_buffer)
        s1_buffer = []
        elapsed = time.time() - t_start
        rate = processed_count / max(1.0, elapsed)
        print(f"[{country}] Processed {processed_count:,} entities | Cands: {total_cands:,} | Matches: {total_matches:,} | Rate: {rate:.1f} ent/s")

    conn.close()

    del cand_records
    del indexer
    gc.collect()

    total_time = time.time() - t0
    print(f"Partition {country} finished: {processed_count:,} new entities processed in {total_time / 60:.2f} min.")
    return processed_count


def export_submission_tsvs(data_dir, output_dir, db_path):
    """
    Exports final matching_results_phase1.tsv and candidate_pairs_phase1.tsv in the exact sequence of test_source1.tsv.
    """
    print("\n============================================================")
    print("  EXPORTING FINAL SUBMISSION TSV FILES")
    print("============================================================")
    os.makedirs(output_dir, exist_ok=True)
    matching_tsv = os.path.join(output_dir, "matching_results_phase1.tsv")
    candidate_tsv = os.path.join(output_dir, "candidate_pairs_phase1.tsv")

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    s1_path = os.path.join(data_dir, "test_source1.tsv")
    written_count = 0
    empty_matches = 0
    total_matches = 0
    total_cands = 0

    t_export = time.time()
    with open(matching_tsv, "w", encoding="utf-8") as f_m, open(candidate_tsv, "w", encoding="utf-8") as f_c:
        f_m.write("source1_entity_id\tmatched_entity_ids\n")
        f_c.write("source1_entity_id\tcandidate_entity_ids\n")

        for chunk in pd.read_csv(s1_path, sep="\t", chunksize=50000, usecols=["entity_id"]):
            eids = chunk["entity_id"].tolist()
            results = {}
            for i in range(0, len(eids), 500):
                sub_eids = eids[i:i + 500]
                placeholders = ",".join("?" for _ in sub_eids)
                cursor.execute(f"SELECT s1_id, candidates, matches FROM preds WHERE s1_id IN ({placeholders})", sub_eids)
                for row in cursor.fetchall():
                    results[row[0]] = (row[1], row[2])

            for eid in eids:
                c_val, m_val = results.get(eid, ("", ""))
                f_c.write(f"{eid}\t{c_val}\n")
                f_m.write(f"{eid}\t{m_val}\n")

                written_count += 1
                if m_val:
                    total_matches += len(m_val.split(","))
                else:
                    empty_matches += 1

                if c_val:
                    total_cands += len(c_val.split(","))

    conn.close()

    print(f"Exported {written_count:,} rows in {time.time() - t_export:.1f}s.")
    print(f"  Candidate file: {candidate_tsv} ({total_cands:,} total candidate links)")
    print(f"  Matching file:  {matching_tsv} ({total_matches:,} total matches, {empty_matches:,} singletons)")

    # Run official validator
    validator_path = os.path.join(os.path.dirname(__file__), "..", "utils", "validate_submission.py")
    if os.path.exists(validator_path):
        print("\nRunning official submission validator...")
        sys.path.insert(0, os.path.dirname(validator_path))
        from validate_submission import validate
        passed = validate(matching_tsv, candidate_tsv)
        print(f"Submission Validator Status: {'PASS' if passed else 'FAIL'}")


def main():
    parser = argparse.ArgumentParser(
        description="Resumable, Large-Scale Business Entity Resolution Inference Pipeline"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Directory containing test_source1.tsv, test_source2.tsv, and test_source3.tsv (auto-detected if omitted)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output",
        help="Directory where TSV outputs are exported (default: output)",
    )
    parser.add_argument(
        "--output-db",
        type=str,
        default="output/temp_preds_phase1.db",
        help="SQLite checkpoint database path (default: output/temp_preds_phase1.db)",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="models/lgbm_model.pkl",
        help="Trained LightGBM model file path (default: models/lgbm_model.pkl)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        help=f"Classification probability threshold (default: {DEFAULT_THRESHOLD})",
    )
    parser.add_argument(
        "--max-candidates",
        type=int,
        default=DEFAULT_CANDIDATE_LIMIT,
        help=f"Maximum candidate pairs per entity (default: {DEFAULT_CANDIDATE_LIMIT})",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"Number of entities per batch commit (default: {DEFAULT_BATCH_SIZE})",
    )
    parser.add_argument(
        "--country",
        type=str,
        default="all",
        choices=["all", "France", "US", "India"],
        help="Process a specific country partition or 'all' (default: all)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        default=True,
        help="Resume from existing checkpoint (default: True)",
    )
    parser.add_argument(
        "--export-only",
        action="store_true",
        help="Skip inference and directly export TSV submission files from SQLite DB",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional entity limit for testing and quick validation passes",
    )

    args = parser.parse_args()

    data_dir = args.data_dir or find_default_data_dir()
    if not os.path.exists(data_dir):
        print(f"[ERROR] Data directory not found: {data_dir}")
        sys.exit(1)

    print(f"=== ENTITY RESOLUTION INFERENCE PIPELINE ===")
    print(f"Data Directory:    {data_dir}")
    print(f"Model Path:        {args.model_path}")
    print(f"Checkpoint DB:     {args.output_db}")
    print(f"Decision Cutoff:   {args.threshold}")
    print(f"Max Candidates:    {args.max_candidates}")

    # Handle export-only mode
    if args.export_only:
        export_submission_tsvs(data_dir, args.output_dir, args.output_db)
        return

    # Ensure model exists
    if not os.path.exists(args.model_path):
        print(f"[ERROR] Model file not found: {args.model_path}")
        print("Please train the primary model first using: python src/train.py")
        sys.exit(1)

    print(f"\nLoading production model from {args.model_path}...")
    with open(args.model_path, "rb") as f:
        model = pickle.load(f)
    _ = model.predict_proba(np.zeros((1, 19)))  # Warm up OpenMP thread pool

    # Initialize SQLite Checkpoint Database
    init_database(args.output_db)
    existing_ids = get_existing_ids(args.output_db) if args.resume else set()
    print(f"Existing checkpoint entries: {len(existing_ids):,} entities.")

    # Audit partition completion
    partition_stats = audit_partitions(data_dir, existing_ids)
    print("\n--- Partition Checkpoint Status ---")
    for c_name, st in partition_stats.items():
        pct = (st["completed"] / st["total"]) * 100 if st["total"] > 0 else 0.0
        status_msg = "COMPLETED (SKIPPED)" if st["completed"] == st["total"] else f"{st['total'] - st['completed']:,} entities remaining"
        print(f"  {c_name:<8}: {st['completed']:,} / {st['total']:,} ({pct:6.2f}%) -> {status_msg}")

    # Determine countries to process
    countries = ["France", "US", "India"] if args.country == "all" else [args.country]

    for country in countries:
        st = partition_stats.get(country, {"total": 0, "completed": 0})
        if st["completed"] == st["total"] and st["total"] > 0 and args.resume:
            print(f"\n[CHECKPOINT] Country '{country}' is 100% complete ({st['completed']:,} / {st['total']:,}). Skipping partition entirely.")
            continue

        run_country_inference(
            country=country,
            data_dir=data_dir,
            model=model,
            threshold=args.threshold,
            db_path=args.output_db,
            max_candidates=args.max_candidates,
            batch_size=args.batch_size,
            existing_ids=existing_ids,
            limit=args.limit,
        )

    # Audit after all runs
    final_existing = get_existing_ids(args.output_db)
    final_stats = audit_partitions(data_dir, final_existing)
    all_done = all(st["completed"] == st["total"] and st["total"] > 0 for st in final_stats.values())

    if all_done:
        print("\nAll partitions 100% completed! Exporting final TSV submission files...")
        export_submission_tsvs(data_dir, args.output_dir, args.output_db)
    else:
        print(f"\nCheckpoint updated: {len(final_existing):,} total entities in {args.output_db}.")
        print("To resume remaining partitions, re-run with: python src/run_test_inference_phase1.py")


if __name__ == "__main__":
    main()
