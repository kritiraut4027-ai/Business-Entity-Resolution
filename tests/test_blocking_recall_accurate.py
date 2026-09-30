import sys
import os
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from blocking import BlockingIndex

def test_blocking_recall_accurate():
    data_dir = r"c:\Users\USER\Downloads\ML Challenge" if os.path.exists(r"c:\Users\USER\Downloads\ML Challenge") else r"G:\My Drive\ML Challenge"
    print("Loading Ground Truth for 1000 S1 entities...")
    
    # 1. Sample 1000 S1 records with matches
    gt = pd.read_csv(os.path.join(data_dir, "train_ground_truth.tsv"), sep="\t", nrows=1000)
    
    ground_truth_map = {}
    target_s2_ids = set()
    target_s3_ids = set()
    total_true_links = 0
    
    for _, row in gt.iterrows():
        s1_id = row["source1_entity_id"]
        matches_str = row["matched_entity_ids"]
        if pd.notna(matches_str) and str(matches_str).strip():
            m_list = [m.strip() for m in str(matches_str).split(",") if m.strip()]
            ground_truth_map[s1_id] = set(m_list)
            for m in m_list:
                if m.startswith("S2-"):
                    target_s2_ids.add(m)
                elif m.startswith("S3-"):
                    target_s3_ids.add(m)
            total_true_links += len(m_list)
        else:
            ground_truth_map[s1_id] = set()

    s1_ids = set(ground_truth_map.keys())
    print(f"Sample S1 count: {len(s1_ids)} | True links to recover: {total_true_links}")
    print(f"Required targets: S2={len(target_s2_ids)}, S3={len(target_s3_ids)}")
    
    # 2. Load S1 records
    s1_records = {}
    for chunk in pd.read_csv(os.path.join(data_dir, "train_source1.tsv"), sep="\t", chunksize=100000):
        for _, r in chunk[chunk["entity_id"].isin(s1_ids)].iterrows():
            s1_records[r["entity_id"]] = r
        if len(s1_records) == len(s1_ids):
            break
            
    # 3. Fast scan S2 to gather ALL target records + 25k distractors
    indexer = BlockingIndex(max_bucket_size=1000)
    print("Scanning S2 for all target matches + distractors...")
    found_s2 = 0
    distractors_s2 = 0
    for chunk in pd.read_csv(os.path.join(data_dir, "train_source2.tsv"), sep="\t", chunksize=200000):
        # Targets
        target_chunk = chunk[chunk["entity_id"].isin(target_s2_ids)]
        for _, r in target_chunk.iterrows():
            indexer.add_record(r["entity_id"], r["business_name"], r["business_address"], r["country"])
            found_s2 += 1
            
        # Distractors (first 25k)
        if distractors_s2 < 25000:
            dist_chunk = chunk[~chunk["entity_id"].isin(target_s2_ids)].head(25000 - distractors_s2)
            for _, r in dist_chunk.iterrows():
                indexer.add_record(r["entity_id"], r["business_name"], r["business_address"], r["country"])
                distractors_s2 += len(dist_chunk)
                
        if found_s2 == len(target_s2_ids) and distractors_s2 >= 25000:
            break
            
    print(f"Indexed S2: {found_s2}/{len(target_s2_ids)} targets + {distractors_s2} distractors")
    
    # 4. Fast scan S3 to gather ALL target records + 25k distractors
    print("Scanning S3 for all target matches + distractors...")
    found_s3 = 0
    distractors_s3 = 0
    for chunk in pd.read_csv(os.path.join(data_dir, "train_source3.tsv"), sep="\t", chunksize=200000):
        target_chunk = chunk[chunk["entity_id"].isin(target_s3_ids)]
        for _, r in target_chunk.iterrows():
            indexer.add_record(r["entity_id"], r["business_name"], r["business_address"], r["country"])
            found_s3 += 1
            
        if distractors_s3 < 25000:
            dist_chunk = chunk[~chunk["entity_id"].isin(target_s3_ids)].head(25000 - distractors_s3)
            for _, r in dist_chunk.iterrows():
                indexer.add_record(r["entity_id"], r["business_name"], r["business_address"], r["country"])
                distractors_s3 += len(dist_chunk)
                
        if found_s3 == len(target_s3_ids) and distractors_s3 >= 25000:
            break
            
    print(f"Indexed S3: {found_s3}/{len(target_s3_ids)} targets + {distractors_s3} distractors")
    
    # 5. Measure True Candidate Recall
    recovered_links = 0
    total_candidates_generated = 0
    
    for s1_id, true_matches in ground_truth_map.items():
        if s1_id not in s1_records:
            continue
        s1 = s1_records[s1_id]
        candidates = indexer.get_candidates_for_record(
            s1["business_name"], s1["business_address"], s1["country"], max_candidates=60
        )
        total_candidates_generated += len(candidates)
        
        for m in true_matches:
            if m in candidates:
                recovered_links += 1

    avg_candidates = total_candidates_generated / max(1, len(s1_records))
    print(f"\n==========================================")
    print(f"ACCURATE BLOCKING BENCHMARK REPORT")
    print(f"==========================================")
    print(f"Average candidates generated per S1 entity: {avg_candidates:.2f}")
    print(f"Recovered true links: {recovered_links} / {total_true_links}")
    print(f"True Candidate Recall: {recovered_links / max(1, total_true_links) * 100:.2f}%")
    print(f"==========================================")

if __name__ == "__main__":
    test_blocking_recall_accurate()
