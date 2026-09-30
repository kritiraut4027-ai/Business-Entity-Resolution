import os
import sys
import json
import time
import sqlite3
import pickle
import gc
import numpy as np
import pandas as pd
from rapidfuzz import distance, fuzz

sys.path.insert(0, os.path.dirname(__file__))

from normalization import normalize_business_name, normalize_address, normalize_country
from blocking import BlockingIndex
from features import compute_pair_features

def run_country_batch(country, data_dir, model, threshold, db_path, existing_ids=None):
    print(f"\n==========================================")
    print(f"PROCESSING COUNTRY: {country}")
    print(f"==========================================")
    t0 = time.time()
    
    # 1. Load S2 and S3 for this country only
    cand_records = {} # eid -> (name, addr, country)
    indexer = BlockingIndex(max_bucket_size=800)
    
    for s_file in ["test_source2.tsv", "test_source3.tsv"]:
        path = os.path.join(data_dir, s_file)
        print(f"Streaming {s_file} for country {country}...")
        for chunk in pd.read_csv(path, sep="\t", chunksize=250000):
            filtered = chunk[chunk["country"].astype(str).str.strip().str.lower() == country.lower()]
            for _, r in filtered.iterrows():
                eid = r["entity_id"]
                cand_records[eid] = (r["business_name"], r["business_address"], r["country"])
                indexer.add_record(eid, r["business_name"], r["business_address"], r["country"])
                
    print(f"Indexed {len(cand_records)} candidate records for {country} in {time.time() - t0:.1f}s")
    
    # 2. Stream test_source1 for this country
    s1_path = os.path.join(data_dir, "test_source1.tsv")
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    if existing_ids is None:
        cursor.execute("SELECT s1_id FROM preds")
        existing_ids = set(r[0] for r in cursor.fetchall())
    
    db_batch = []
    s1_count = 0
    total_cands = 0
    total_matches = 0
    
    t_start_s1 = time.time()
    for chunk in pd.read_csv(s1_path, sep="\t", chunksize=50000):
        c_chunk = chunk[chunk["country"].astype(str).str.strip().str.lower() == country.lower()]
        if len(c_chunk) == 0:
            continue
            
        for _, s1 in c_chunk.iterrows():
            s1_id = s1["entity_id"]
            if s1_id in existing_ids:
                continue
            s1_name = s1["business_name"]
            s1_addr = s1["business_address"]
            s1_country = s1["country"]
            
            cands = indexer.get_candidates_for_record(s1_name, s1_addr, s1_country, max_candidates=50)
            valid_cands = [c for c in cands if c in cand_records]
            total_cands += len(valid_cands)
            
            # Predict
            matched_ids = []
            if valid_cands:
                pair_features = []
                eval_cands = []
                for cid in valid_cands:
                    c_name, c_addr, c_country = cand_records[cid]
                    
                    # Fast Pre-filter: if exact match on clean name
                    if s1_name and c_name and str(s1_name).strip().lower() == str(c_name).strip().lower():
                        matched_ids.append(cid)
                        continue
                        
                    # Calculate fast Jaro-Winkler
                    jw = distance.JaroWinkler.similarity(str(s1_name or ''), str(c_name or ''))
                    if jw >= 0.70 or (s1_addr and c_addr and fuzz.token_set_ratio(str(s1_addr), str(c_addr)) >= 70):
                        feats = compute_pair_features(s1_name, s1_addr, s1_country, c_name, c_addr, c_country)
                        pair_features.append(feats)
                        eval_cands.append(cid)
                        
                if pair_features:
                    probs = model.predict_proba(np.array(pair_features))[:, 1]
                    for cid, p in zip(eval_cands, probs):
                        if p >= threshold:
                            matched_ids.append(cid)
                            
            total_matches += len(matched_ids)
            db_batch.append((s1_id, ",".join(valid_cands), ",".join(matched_ids)))
            s1_count += 1
            
            if len(db_batch) >= 10000:
                cursor.executemany("INSERT OR REPLACE INTO preds VALUES (?, ?, ?)", db_batch)
                conn.commit()
                db_batch = []
                
        elapsed = time.time() - t_start_s1
        print(f"[{country}] Processed {s1_count} S1 entities... (Candidates: {total_cands}, Matches: {total_matches}) [{s1_count / max(1, elapsed):.1f} ent/s]")
        
    if db_batch:
        cursor.executemany("INSERT OR REPLACE INTO preds VALUES (?, ?, ?)", db_batch)
        conn.commit()
        
    conn.close()
    
    # Free memory
    del cand_records
    del indexer
    gc.collect()
    
    print(f"Finished {country}: {s1_count} entities, {total_matches} matches in {time.time() - t0:.1f}s")
    return s1_count, total_cands, total_matches

def run_full_inference(
    data_dir=r"G:\My Drive\ML Challenge",
    output_dir="output",
    model_path="models/lgbm_model.pkl",
    config_path="models/model_config.json"
):
    start_total = time.time()
    os.makedirs(output_dir, exist_ok=True)
    db_path = os.path.join(output_dir, "temp_preds.db")
    # Preserving existing database if resuming
        
    # Init SQLite DB (do NOT remove if resuming)
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("CREATE TABLE IF NOT EXISTS preds (s1_id TEXT PRIMARY KEY, candidates TEXT, matches TEXT)")
    conn.commit()
    
    # Check existing counts per country to enable resume
    s1_path = os.path.join(data_dir, "test_source1.tsv")
    df_s1_countries = pd.read_csv(s1_path, sep="\t", usecols=["entity_id", "country"])
    
    cursor.execute("SELECT s1_id FROM preds")
    existing_ids = set(r[0] for r in cursor.fetchall())
    conn.close()
    
    print(f"Existing entities found in intermediate database: {len(existing_ids):,} / {len(df_s1_countries):,}")
    
    # Load model
    print(f"Loading model from {model_path}...")
    with open(model_path, "rb") as f:
        model = pickle.load(f)
        
    threshold = 0.95
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            cfg = json.load(f)
            threshold = cfg.get("optimal_threshold", 0.95)
    print(f"Optimal Threshold: {threshold}")
    
    countries = ["France", "US", "India"]
    
    for c in countries:
        c_entities = set(df_s1_countries[df_s1_countries["country"].str.lower() == c.lower()]["entity_id"])
        c_completed = len(c_entities.intersection(existing_ids))
        if c_completed == len(c_entities) and len(c_entities) > 0:
            print(f"\n[SKIP] Country {c} is ALREADY 100% COMPLETED ({c_completed:,} / {len(c_entities):,}). Preserving results.")
            continue
        print(f"\n[RESUME] Country {c} needs processing ({c_completed:,} / {len(c_entities):,} done). Starting batch...")
        run_country_batch(c, data_dir, model, threshold, db_path, existing_ids=existing_ids)

    # Export to final TSVs in exact test_source1 order
    print("\n==========================================")
    print("EXPORTING IN EXACT SEQUENCE OF TEST_SOURCE1.TSV")
    print("==========================================")
    
    matching_final = os.path.join(output_dir, "matching_results.tsv")
    candidate_final = os.path.join(output_dir, "candidate_pairs.tsv")
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    s1_path = os.path.join(data_dir, "test_source1.tsv")
    written_count = 0
    empty_matches_count = 0
    total_matches_count = 0
    total_cands_count = 0
    
    with open(matching_final, "w", encoding="utf-8") as f_m, open(candidate_final, "w", encoding="utf-8") as f_c:
        f_m.write("source1_entity_id\tmatched_entity_ids\n")
        f_c.write("source1_entity_id\tcandidate_entity_ids\n")
        
        # Read test_source1 in chunks to batch query SQLite
        for chunk in pd.read_csv(s1_path, sep="\t", chunksize=50000, usecols=["entity_id"]):
            eids = chunk["entity_id"].tolist()
            results_dict = {}
            for i in range(0, len(eids), 500):
                sub_eids = eids[i:i+500]
                placeholders = ",".join("?" for _ in sub_eids)
                cursor.execute(f"SELECT s1_id, candidates, matches FROM preds WHERE s1_id IN ({placeholders})", sub_eids)
                for row in cursor.fetchall():
                    results_dict[row[0]] = (row[1], row[2])
            
            for eid in eids:
                c_val, m_val = results_dict.get(eid, ("", ""))
                f_c.write(f"{eid}\t{c_val}\n")
                f_m.write(f"{eid}\t{m_val}\n")
                
                written_count += 1
                if m_val:
                    total_matches_count += len(m_val.split(","))
                else:
                    empty_matches_count += 1
                    
                if c_val:
                    total_cands_count += len(c_val.split(","))
                    
        print(f"Exported {written_count} rows...")
        
    conn.close()
    
    # Keep temp DB for safety
    print(f"Intermediate database preserved at {db_path}")
        
    total_time = time.time() - start_total
    print("\n==========================================")
    print("FULL INFERENCE COMPLETED SUCCESSFULLY")
    print("==========================================")
    print(f"Total Source 1 Entities: {written_count:,}")
    print(f"Total Candidate Pairs: {total_cands_count:,}")
    print(f"Total Final Matches: {total_matches_count:,}")
    print(f"Singletons (Empty Matches): {empty_matches_count:,}")
    print(f"Total Runtime: {total_time / 60:.2f} minutes")
    print(f"Candidate File: {candidate_final}")
    print(f"Matching File: {matching_final}")
    
    # Run official validator
    print("\nTriggering official submission validator...")
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "utils"))
    from validate_submission import validate
    valid_pass = validate(matching_final, candidate_final)
    print(f"Final Validator Status: {'PASS' if valid_pass else 'FAIL'}")

if __name__ == "__main__":
    run_full_inference()
