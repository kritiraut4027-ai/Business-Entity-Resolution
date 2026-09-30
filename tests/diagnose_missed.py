import sys
import os
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from blocking import BlockingIndex, extract_blocking_keys

def diagnose_missed():
    data_dir = r"c:\Users\USER\Downloads\ML Challenge" if os.path.exists(r"c:\Users\USER\Downloads\ML Challenge") else r"G:\My Drive\ML Challenge"
    
    # Load 500 S1 records with matches
    gt = pd.read_csv(os.path.join(data_dir, "train_ground_truth.tsv"), sep="\t", nrows=500)
    
    ground_truth_map = {}
    target_ids = set()
    for _, row in gt.iterrows():
        s1_id = row["source1_entity_id"]
        matches_str = row["matched_entity_ids"]
        if pd.notna(matches_str) and str(matches_str).strip():
            m_list = [m.strip() for m in str(matches_str).split(",") if m.strip()]
            ground_truth_map[s1_id] = set(m_list)
            target_ids.update(m_list)
            
    s1_ids = set(ground_truth_map.keys())
    
    # Load S1, S2, S3
    s1_records = {}
    for chunk in pd.read_csv(os.path.join(data_dir, "train_source1.tsv"), sep="\t", chunksize=100000):
        for _, r in chunk[chunk["entity_id"].isin(s1_ids)].iterrows():
            s1_records[r["entity_id"]] = r
        if len(s1_records) == len(s1_ids):
            break
            
    target_records = {}
    for source_file in ["train_source2.tsv", "train_source3.tsv"]:
        for chunk in pd.read_csv(os.path.join(data_dir, source_file), sep="\t", chunksize=100000):
            for _, r in chunk[chunk["entity_id"].isin(target_ids)].iterrows():
                target_records[r["entity_id"]] = r
            if len(target_records) == len(target_ids):
                break

    indexer = BlockingIndex(max_bucket_size=5000)
    for eid, r in target_records.items():
        indexer.add_record(eid, r["business_name"], r["business_address"], r["country"])
        
    missed_count = 0
    print("=== DIAGNOSING MISSED TRUE MATCHES ===")
    
    for s1_id, true_matches in ground_truth_map.items():
        if s1_id not in s1_records:
            continue
        s1 = s1_records[s1_id]
        s1_keys = extract_blocking_keys(s1["business_name"], s1["business_address"], s1["country"])
        candidates = indexer.get_candidates_for_record(
            s1["business_name"], s1["business_address"], s1["country"], max_candidates=200
        )
        
        for m in true_matches:
            if m not in candidates and m in target_records:
                target = target_records[m]
                target_keys = extract_blocking_keys(target["business_name"], target["business_address"], target["country"])
                
                print(f"\n[MISSED MATCH]")
                print(f"  S1:     {s1_id} | Name: '{s1['business_name']}' | Addr: '{s1['business_address']}' | Country: '{s1['country']}'")
                print(f"  Target: {m} | Name: '{target['business_name']}' | Addr: '{target['business_address']}' | Country: '{target['country']}'")
                print(f"  Shared Keys: {s1_keys.intersection(target_keys)}")
                print(f"  S1 Keys: {list(s1_keys)[:3]}")
                print(f"  Target Keys: {list(target_keys)[:3]}")
                
                missed_count += 1
                if missed_count >= 8:
                    return

if __name__ == "__main__":
    diagnose_missed()
