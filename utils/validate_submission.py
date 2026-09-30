import os
import sys
import argparse

def validate(matching_file, candidate_file, test_dir=None):
    print("=== RUNNING SUBMISSION VALIDATION ===")
    errors = []
    
    if not os.path.exists(matching_file):
        errors.append(f"Missing matching_results file: {matching_file}")
    if not os.path.exists(candidate_file):
        errors.append(f"Missing candidate_pairs file: {candidate_file}")
        
    if errors:
        for e in errors:
            print(f"[FAIL] {e}")
        return False
        
    # 1. Validate candidate_pairs.tsv
    print(f"Checking {candidate_file}...")
    candidate_map = {}
    with open(candidate_file, "r", encoding="utf-8") as f:
        header = f.readline().strip().split("\t")
        if header != ["source1_entity_id", "candidate_entity_ids"]:
            errors.append(f"Invalid candidate_pairs header: {header}. Expected ['source1_entity_id', 'candidate_entity_ids']")
            
        for line_num, line in enumerate(f, 2):
            parts = line.strip().split("\t")
            if len(parts) == 1:
                s1_id = parts[0]
                cands = set()
            elif len(parts) == 2:
                s1_id = parts[0]
                raw_cands = [c.strip() for c in parts[1].split(",") if c.strip()]
                # Check for duplicates
                if len(raw_cands) != len(set(raw_cands)):
                    errors.append(f"Duplicate candidate IDs found at line {line_num} for {s1_id}")
                cands = set(raw_cands)
            else:
                errors.append(f"Invalid format at line {line_num}: expected 1 or 2 tab-separated columns")
                continue
                
            if s1_id in candidate_map:
                errors.append(f"Duplicate source1_entity_id in candidates: {s1_id} at line {line_num}")
            candidate_map[s1_id] = cands

    # 2. Validate matching_results.tsv
    print(f"Checking {matching_file}...")
    matching_map = {}
    with open(matching_file, "r", encoding="utf-8") as f:
        header = f.readline().strip().split("\t")
        if header != ["source1_entity_id", "matched_entity_ids"]:
            errors.append(f"Invalid matching_results header: {header}. Expected ['source1_entity_id', 'matched_entity_ids']")
            
        for line_num, line in enumerate(f, 2):
            parts = line.strip().split("\t")
            if len(parts) == 1:
                s1_id = parts[0]
                matches = set()
            elif len(parts) == 2:
                s1_id = parts[0]
                raw_matches = [m.strip() for m in parts[1].split(",") if m.strip()]
                if len(raw_matches) != len(set(raw_matches)):
                    errors.append(f"Duplicate matched IDs found at line {line_num} for {s1_id}")
                for m in raw_matches:
                    if m.startswith("S1-"):
                        errors.append(f"Self-match to Source 1 prohibited: {m} for {s1_id} at line {line_num}")
                    if not (m.startswith("S2-") or m.startswith("S3-")):
                        errors.append(f"Invalid entity ID prefix: {m} for {s1_id} at line {line_num}")
                matches = set(raw_matches)
            else:
                errors.append(f"Invalid format at line {line_num}: expected 1 or 2 tab-separated columns")
                continue
                
            if s1_id in matching_map:
                errors.append(f"Duplicate source1_entity_id in matching: {s1_id} at line {line_num}")
            matching_map[s1_id] = matches

    # 3. Check consistency: candidate_pairs and matching_results entity alignment
    if set(candidate_map.keys()) != set(matching_map.keys()):
        errors.append(f"Mismatch in source1_entity_ids between candidates ({len(candidate_map)}) and matching ({len(matching_map)})")

    # 4. Check subset condition: every matched ID MUST exist in candidate_pairs
    for s1_id, matches in matching_map.items():
        cands = candidate_map.get(s1_id, set())
        missing_cands = matches - cands
        if missing_cands:
            errors.append(f"Matched IDs not in candidate pool for {s1_id}: {missing_cands}")
            if len(errors) > 20:
                break

    if errors:
        print("\n=== VALIDATION FAILED ===")
        for i, err in enumerate(errors[:20], 1):
            print(f"{i}. {err}")
        if len(errors) > 20:
            print(f"... and {len(errors) - 20} more errors.")
        return False
    else:
        print("\n=== VALIDATION RESULT: PASS ===")
        print(f"Total Source 1 entities: {len(matching_map)}")
        total_candidates = sum(len(c) for c in candidate_map.values())
        total_matches = sum(len(m) for m in matching_map.values())
        singletons = sum(1 for m in matching_map.values() if len(m) == 0)
        print(f"Total Candidates: {total_candidates} (Avg: {total_candidates / max(1, len(matching_map)):.2f}/entity)")
        print(f"Total Final Matches: {total_matches}")
        print(f"Singletons (Zero matches): {singletons} ({singletons / max(1, len(matching_map)) * 100:.1f}%)")
        return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    default_matching = "output/matching_results_phase1.tsv" if os.path.exists("output/matching_results_phase1.tsv") else "output/matching_results.tsv"
    default_candidate = "output/candidate_pairs_phase1.tsv" if os.path.exists("output/candidate_pairs_phase1.tsv") else "output/candidate_pairs.tsv"
    parser.add_argument("--matching", default=default_matching)
    parser.add_argument("--candidate", default=default_candidate)
    parser.add_argument("--test-dir", default=None)
    args = parser.parse_args()
    
    success = validate(args.matching, args.candidate, args.test_dir)
    sys.exit(0 if success else 1)
