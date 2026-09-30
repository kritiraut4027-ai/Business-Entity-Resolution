import os
import sys
import json
import time
import pickle
import gc
import numpy as np
import pandas as pd
from rapidfuzz import distance, fuzz

sys.path.insert(0, os.path.dirname(__file__))

from normalization import normalize_business_name, normalize_address, normalize_country
from blocking import BlockingIndex
from features import compute_pair_features

def run_country_inference(country, data_dir, model, threshold, temp_dir):
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
            # Filter chunk by country
            filtered = chunk[chunk["country"].astype(str).str.strip().str.lower() == country.lower()]
            for _, r in filtered.iterrows():
                eid = r["entity_id"]
                cand_records[eid] = (r["business_name"], r["business_address"], r["country"])
                indexer.add_record(eid, r["business_name"], r["business_address"], r["country"])
                
    print(f"Indexed {len(cand_records)} candidate records for {country} in {time.time() - t0:.1f}s")
    
    # 2. Stream test_source1 for this country
    s1_path = os.path.join(data_dir, "test_source1.tsv")
    match_out_path = os.path.join(temp_dir, f"matching_{country}.tsv")
    cand_out_path = os.path.join(temp_dir, f"candidate_{country}.tsv")
    
    f_match = open(match_out_path, "w", encoding="utf-8")
    f_cand = open(cand_out_path, "w", encoding="utf-8")
    
    f_match.write("source1_entity_id\tmatched_entity_ids\n")
    f_cand.write("source1_entity_id\tcandidate_entity_ids\n")
    
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
            s1_name = s1["business_name"]
            s1_addr = s1["business_address"]
            s1_country = s1["country"]
            
            cands = indexer.get_candidates_for_record(s1_name, s1_addr, s1_country, max_candidates=50)
            valid_cands = [c for c in cands if c in cand_records]
            
            # Write candidates
            f_cand.write(f"{s1_id}\t{','.join(valid_cands)}\n")
            total_cands += len(valid_cands)
            
            # Predict
            matched_ids = []
            if valid_cands:
                pair_features = []
                eval_cands = []
                for cid in valid_cands:
                    c_name, c_addr, c_country = cand_records[cid]
                    
                    # Fast Pre-filter: if exact match on clean name, fast path
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
                            
            f_match.write(f"{s1_id}\t{','.join(matched_ids)}\n")
            total_matches += len(matched_ids)
            s1_count += 1
            
        elapsed = time.time() - t_start_s1
        print(f"[{country}] Processed {s1_count} S1 entities... (Candidates: {total_cands}, Matches: {total_matches}) [{s1_count / max(1, elapsed):.1f} ent/s]")
        
    f_match.close()
    f_cand.close()
    
    # Clean memory
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
    temp_dir = os.path.join(output_dir, "temp_partitions")
    os.makedirs(temp_dir, exist_ok=True)
    
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
    partition_stats = {}
    
    for c in countries:
        count, cands, matches = run_country_inference(c, data_dir, model, threshold, temp_dir)
        partition_stats[c] = {"entities": count, "candidates": cands, "matches": matches}

    # Merge partitions in exact order of test_source1.tsv
    print("\nMerging partitions into final submission files...")
    matching_final = os.path.join(output_dir, "matching_results.tsv")
    candidate_final = os.path.join(output_dir, "candidate_pairs.tsv")
    
    # Read all partition lines into memory-efficient maps
    match_map = {}
    cand_map = {}
    for c in countries:
        with open(os.path.join(temp_dir, f"matching_{c}.tsv"), "r", encoding="utf-8") as f:
            next(f) # header
            for line in f:
                parts = line.strip().split("\t")
                match_map[parts[0]] = parts[1] if len(parts) > 1 else ""
                
        with open(os.path.join(temp_dir, f"candidate_{c}.tsv"), "r", encoding="utf-8") as f:
            next(f)
            for line in f:
                parts = line.strip().split("\t")
                cand_map[parts[0]] = parts[1] if len(parts) > 1 else ""

    print(f"Writing final ordered outputs for 1,732,544 entities...")
    s1_path = os.path.join(data_dir, "test_source1.tsv")
    with open(matching_final, "w", encoding="utf-8") as f_m, open(candidate_final, "w", encoding="utf-8") as f_c:
        f_m.write("source1_entity_id\tmatched_entity_ids\n")
        f_c.write("source1_entity_id\tcandidate_entity_ids\n")
        
        with open(s1_path, "r", encoding="utf-8") as s1_f:
            next(s1_f) # header
            for line in s1_f:
                s1_id = line.split("\t")[0]
                m_val = match_map.get(s1_id, "")
                c_val = cand_map.get(s1_id, "")
                f_m.write(f"{s1_id}\t{m_val}\n")
                f_c.write(f"{s1_id}\t{c_val}\n")

    total_time = time.time() - start_total
    print("\n==========================================")
    print("FULL INFERENCE COMPLETED SUCCESSFULLY")
    print("==========================================")
    print(f"Total Runtime: {total_time / 60:.2f} minutes")
    print(f"Candidate File: {candidate_final}")
    print(f"Matching File: {matching_final}")

if __name__ == "__main__":
    run_full_inference()
