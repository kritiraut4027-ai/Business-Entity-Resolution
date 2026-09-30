import re
import numpy as np
from rapidfuzz import fuzz, distance
from normalization import normalize_business_name, normalize_address, normalize_country

def extract_numbers_set(text):
    if not text:
        return set()
    nums = re.findall(r'\b\d+\b', text)
    # Strip leading zeros
    return {re.sub(r'^0+', '', n) for n in nums if re.sub(r'^0+', '', n)}

def compute_pair_features(s1_name, s1_addr, s1_country, cand_name, cand_addr, cand_country):
    """
    Computes a vector of high-discriminative string, token, numeric, and geographic similarity features.
    """
    s1_clean_name, s1_core_name = normalize_business_name(s1_name)
    cand_clean_name, cand_core_name = normalize_business_name(cand_name)
    
    s1_norm_addr = normalize_address(s1_addr)
    cand_norm_addr = normalize_address(cand_addr)
    
    s1_norm_country = normalize_country(s1_country)
    cand_norm_country = normalize_country(cand_country)
    
    # 1. Name Similarities
    # Jaro-Winkler
    jw_sim = distance.JaroWinkler.similarity(s1_clean_name, cand_clean_name)
    core_jw_sim = distance.JaroWinkler.similarity(s1_core_name, cand_core_name)
    
    # RapidFuzz Ratios (0-100 scaled to 0-1)
    name_ratio = fuzz.ratio(s1_clean_name, cand_clean_name) / 100.0
    core_name_ratio = fuzz.ratio(s1_core_name, cand_core_name) / 100.0
    name_token_sort = fuzz.token_sort_ratio(s1_clean_name, cand_clean_name) / 100.0
    name_token_set = fuzz.token_set_ratio(s1_clean_name, cand_clean_name) / 100.0
    name_partial = fuzz.partial_ratio(s1_clean_name, cand_clean_name) / 100.0
    
    # Token Overlap
    s1_tokens = set(s1_clean_name.split())
    cand_tokens = set(cand_clean_name.split())
    common_name_tokens = len(s1_tokens.intersection(cand_tokens))
    name_jaccard = common_name_tokens / max(1, len(s1_tokens.union(cand_tokens)))
    
    # Name Length Difference
    name_len_diff = abs(len(s1_clean_name) - len(cand_clean_name))
    
    # 2. Address Similarities
    has_s1_addr = bool(s1_norm_addr)
    has_cand_addr = bool(cand_norm_addr)
    is_addr_missing = 1.0 if (not has_s1_addr or not has_cand_addr) else 0.0
    
    if has_s1_addr and has_cand_addr:
        addr_ratio = fuzz.ratio(s1_norm_addr, cand_norm_addr) / 100.0
        addr_token_set = fuzz.token_set_ratio(s1_norm_addr, cand_norm_addr) / 100.0
        addr_partial = fuzz.partial_ratio(s1_norm_addr, cand_norm_addr) / 100.0
        
        s1_addr_tokens = set(s1_norm_addr.split())
        cand_addr_tokens = set(cand_norm_addr.split())
        common_addr_tokens = len(s1_addr_tokens.intersection(cand_addr_tokens))
        addr_jaccard = common_addr_tokens / max(1, len(s1_addr_tokens.union(cand_addr_tokens)))
        
        # Numeric match in address (street numbers, PIN, phone)
        s1_nums = extract_numbers_set(s1_norm_addr)
        cand_nums = extract_numbers_set(cand_norm_addr)
        if s1_nums and cand_nums:
            num_overlap = len(s1_nums.intersection(cand_nums))
            has_matching_number = 1.0 if num_overlap > 0 else 0.0
            num_jaccard = num_overlap / len(s1_nums.union(cand_nums))
        else:
            has_matching_number = 0.5 # neutral
            num_jaccard = 0.5
    else:
        addr_ratio = 0.0
        addr_token_set = 0.0
        addr_partial = 0.0
        common_addr_tokens = 0
        addr_jaccard = 0.0
        has_matching_number = 0.5
        num_jaccard = 0.5
        
    # 3. Country Match
    country_match = 1.0 if (s1_norm_country == cand_norm_country and s1_norm_country != "") else 0.0
    
    return [
        name_ratio,
        core_name_ratio,
        name_token_sort,
        name_token_set,
        name_partial,
        jw_sim,
        core_jw_sim,
        name_jaccard,
        float(common_name_tokens),
        float(name_len_diff),
        is_addr_missing,
        addr_ratio,
        addr_token_set,
        addr_partial,
        addr_jaccard,
        float(common_addr_tokens),
        has_matching_number,
        num_jaccard,
        country_match
    ]

FEATURE_NAMES = [
    "name_ratio",
    "core_name_ratio",
    "name_token_sort",
    "name_token_set",
    "name_partial",
    "jw_sim",
    "core_jw_sim",
    "name_jaccard",
    "common_name_tokens",
    "name_len_diff",
    "is_addr_missing",
    "addr_ratio",
    "addr_token_set",
    "addr_partial",
    "addr_jaccard",
    "common_addr_tokens",
    "has_matching_number",
    "num_jaccard",
    "country_match"
]
