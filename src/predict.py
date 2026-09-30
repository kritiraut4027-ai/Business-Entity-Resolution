import os
import sys
import json
import pickle
import argparse
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))

from normalization import normalize_business_name, normalize_address, normalize_country
from blocking import BlockingIndex
from features import compute_pair_features

def run_inference(
    data_dir=r"G:\My Drive\ML Challenge",
    output_dir="output",
    model_path="models/lgbm_model.pkl",
    config_path="models/model_config.json",
    limit=None
):
    print("=== STARTING INFERENCE & SUBMISSION GENERATION ===")
    os.makedirs(output_dir, exist_ok=True)
    
    # Load model and threshold
    print(f"Loading trained matching model from {model_path}...")
    with open(model_path, "rb") as f:
        model = pickle.load(f)
        
    threshold = 0.95
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            cfg = json.load(f)
            threshold = cfg.get("optimal_threshold", 0.95)
    print(f"Applying decision threshold: {threshold}")
    
    # 1. Index Test Sources 2 & 3
    print("Building Inverted Blocking Index for test sources...")
    indexer = BlockingIndex(max_bucket_size=1000)
    cand_records = {} # store minimal info needed for feature computation: (name, addr, country)
    
    for s_file in ["test_source2.tsv", "test_source3.tsv"]:
        path = os.path.join(data_dir, s_file)
        if not os.path.exists(path):
            print(f"Warning: {path} not found. Skipping.")
            continue
        print(f"Indexing {s_file}...")
        for chunk in pd.read_csv(path, sep="\t", chunksize=250000):
            for _, r in chunk.iterrows():
                eid = r["entity_id"]
                cand_records[eid] = (r["business_name"], r["business_address"], r["country"])
                indexer.add_record(eid, r["business_name"], r["business_address"], r["country"])
            if limit and len(cand_records) >= limit * 20:
                break
                
    print(f"Total candidate records indexed: {len(cand_records)}")
    
    # 2. Open output TSVs for streaming writes
    matching_path = os.path.join(output_dir, "matching_results.tsv")
    candidate_path = os.path.join(output_dir, "candidate_pairs.tsv")
    
    match_f = open(matching_path, "w", encoding="utf-8")
    cand_f = open(candidate_path, "w", encoding="utf-8")
    
    match_f.write("source1_entity_id\tmatched_entity_ids\n")
    cand_f.write("source1_entity_id\tcandidate_entity_ids\n")
    
    # 3. Stream Test Source 1 entities and predict
    print("Processing test_source1.tsv and writing predictions...")
    s1_path = os.path.join(data_dir, "test_source1.tsv")
    
    processed_count = 0
    total_matches_count = 0
    total_candidates_count = 0
    
    for chunk in pd.read_csv(s1_path, sep="\t", chunksize=20000):
        for _, s1 in chunk.iterrows():
            s1_id = s1["entity_id"]
            cands = indexer.get_candidates_for_record(
                s1["business_name"], s1["business_address"], s1["country"], max_candidates=60
            )
            # Remove any inadvertent self-references or S1 prefixes
            valid_cands = [c for c in cands if (c.startswith("S2-") or c.startswith("S3-")) and c in cand_records]
            
            # Write to candidate_pairs.tsv
            cand_str = ",".join(valid_cands)
            cand_f.write(f"{s1_id}\t{cand_str}\n")
            total_candidates_count += len(valid_cands)
            
            # Compute features & predict matches
            if valid_cands:
                pair_features = []
                for cid in valid_cands:
                    c_name, c_addr, c_country = cand_records[cid]
                    feats = compute_pair_features(
                        s1["business_name"], s1["business_address"], s1["country"],
                        c_name, c_addr, c_country
                    )
                    pair_features.append(feats)
                    
                probs = model.predict_proba(np.array(pair_features))[:, 1]
                matches = [cid for cid, p in zip(valid_cands, probs) if p >= threshold]
            else:
                matches = []
                
            # Write to matching_results.tsv
            match_str = ",".join(matches)
            match_f.write(f"{s1_id}\t{match_str}\n")
            total_matches_count += len(matches)
            
            processed_count += 1
            if limit and processed_count >= limit:
                break
                
        print(f"Processed {processed_count} S1 entities... (Candidates: {total_candidates_count}, Matches: {total_matches_count})")
        if limit and processed_count >= limit:
            break
            
    match_f.close()
    cand_f.close()
    
    print("\n=== INFERENCE COMPLETE ===")
    print(f"Results saved to:")
    print(f"  1. {matching_path}")
    print(f"  2. {candidate_path}")
    print(f"Processed Entities: {processed_count}")
    print(f"Total Matches Predicted: {total_matches_count}")
    print(f"Average Candidates per Entity: {total_candidates_count / max(1, processed_count):.2f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=r"G:\My Drive\ML Challenge")
    parser.add_argument("--output-dir", default="output")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    
    run_inference(data_dir=args.data_dir, output_dir=args.output_dir, limit=args.limit)
