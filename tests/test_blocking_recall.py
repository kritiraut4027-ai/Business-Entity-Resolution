import sys
import os
import pandas as pd

# Add src to python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from blocking import BlockingIndex

def test_blocking_recall():
    data_dir = r"c:\Users\USER\Downloads\ML Challenge" if os.path.exists(r"c:\Users\USER\Downloads\ML Challenge") else r"G:\My Drive\ML Challenge"
    print("Loading Ground Truth and Source records for blocking recall test...")
    
    # 1. Load sample ground truth (first 3000 S1 records)
    gt = pd.read_csv(os.path.join(data_dir, "train_ground_truth.tsv"), sep="\t", nrows=3000)
    
    # Collect true matches
    ground_truth_map = {}
    target_s2_s3_ids = set()
    total_true_links = 0
    
    for _, row in gt.iterrows():
        s1_id = row["source1_entity_id"]
        matches_str = row["matched_entity_ids"]
        if pd.notna(matches_str) and str(matches_str).strip():
            m_list = [m.strip() for m in str(matches_str).split(",") if m.strip()]
            ground_truth_map[s1_id] = set(m_list)
            target_s2_s3_ids.update(m_list)
            total_true_links += len(m_list)
        else:
            ground_truth_map[s1_id] = set()

    s1_ids = set(ground_truth_map.keys())
    print(f"Total S1 entities in test sample: {len(s1_ids)}")
    print(f"Total true links to recover: {total_true_links}")
    
    # 2. Load S1 records
    s1_records = {}
    for chunk in pd.read_csv(os.path.join(data_dir, "train_source1.tsv"), sep="\t", chunksize=100000):
        m = chunk[chunk["entity_id"].isin(s1_ids)]
        for _, r in m.iterrows():
            s1_records[r["entity_id"]] = r
        if len(s1_records) == len(s1_ids):
            break
            
    # 3. Load S2 and S3 records: ensure all true matches are loaded, plus distractors
    indexer = BlockingIndex(max_bucket_size=500)
    
    print("Indexing Source 2 records (including target matches)...")
    indexed_s2 = 0
    for chunk in pd.read_csv(os.path.join(data_dir, "train_source2.tsv"), sep="\t", chunksize=100000):
        # Index true targets or first 50k as distractors
        for _, r in chunk.iterrows():
            eid = r["entity_id"]
            if eid in target_s2_s3_ids or indexed_s2 < 50000:
                indexer.add_record(eid, r["business_name"], r["business_address"], r["country"])
                indexed_s2 += 1
        if indexed_s2 >= 60000 and len(target_s2_s3_ids - set(indexer.index.keys())) == 0:
            break
            
    print("Indexing Source 3 records (including target matches)...")
    indexed_s3 = 0
    for chunk in pd.read_csv(os.path.join(data_dir, "train_source3.tsv"), sep="\t", chunksize=100000):
        for _, r in chunk.iterrows():
            eid = r["entity_id"]
            if eid in target_s2_s3_ids or indexed_s3 < 50000:
                indexer.add_record(eid, r["business_name"], r["business_address"], r["country"])
                indexed_s3 += 1
        if indexed_s3 >= 60000:
            break
            
    print(f"Total candidate pool indexed: {indexed_s2 + indexed_s3} records")
    
    # 4. Measure candidate recall and candidate pool size
    recovered_links = 0
    total_candidates_generated = 0
    
    for s1_id, true_matches in ground_truth_map.items():
        if s1_id not in s1_records:
            continue
        s1 = s1_records[s1_id]
        candidates = indexer.get_candidates_for_record(
            s1["business_name"], 
            s1["business_address"], 
            s1["country"],
            max_candidates=50
        )
        total_candidates_generated += len(candidates)
        
        # Only check recall against matches that were actually loaded in our pool
        for m in true_matches:
            if m in candidates:
                recovered_links += 1

    # Count how many of target_s2_s3_ids were actually inside our 200k indexed records
    links_in_sample = sum(
        1 for s1_id, matches in ground_truth_map.items()
        for m in matches
        if (m.startswith("S2-") and int(m.split("-")[1]) is not None) # checked by pool
    )
    
    avg_candidates = total_candidates_generated / max(1, len(s1_records))
    print(f"\n=== BLOCKING EVALUATION REPORT ===")
    print(f"Average candidates generated per S1 entity: {avg_candidates:.2f}")
    print(f"Recovered true links: {recovered_links} / {total_true_links}")
    print(f"Candidate Recall (overall sample): {recovered_links / max(1, total_true_links) * 100:.2f}%")

if __name__ == "__main__":
    test_blocking_recall()
