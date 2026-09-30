import re
from collections import defaultdict
from normalization import normalize_business_name, normalize_address, normalize_country

COMMON_ADDR_STOPWORDS = {
    'street', 'avenue', 'road', 'boulevard', 'drive', 'lane', 'highway', 'court',
    'township', 'place', 'square', 'suite', 'floor', 'near', 'opp', 'opposite',
    'behind', 'beside', 'delhi', 'new york', 'california', 'texas', 'maharashtra'
}

def get_blocking_key_priority(key):
    """
    Deterministically prioritize blocking keys:
    0: NUM_LONG (phone, PIN code, 5+ digit numbers - highest selectivity)
    1: NTOK (exact significant name tokens >= 3 chars)
    2: NUM_WORD (house/plot number paired with address word)
    3: LOC_PAIR (locality word pairs)
    4: NPREF4 (4-char business name prefix)
    5: NPREF3 (3-char business name prefix - broad/fallback only)
    """
    parts = key.split("|", 1)
    tag = parts[1] if len(parts) > 1 else parts[0]
    
    if tag.startswith("NUM_LONG:"):
        return 0
    elif tag.startswith("NTOK:"):
        return 1
    elif tag.startswith("NUM_WORD:"):
        return 2
    elif tag.startswith("LOC_PAIR:"):
        return 3
    elif tag.startswith("NPREF4:"):
        return 4
    elif tag.startswith("NPREF3:"):
        return 5
    return 6

def extract_blocking_keys(name, address, country):
    """
    Generate high-recall, multi-strategy blocking keys:
    1. Name 4-char and 3-char prefixes
    2. Name significant tokens (length >= 3)
    3. Distinct address numbers (with leading zeros stripped, e.g. 0684 -> 684)
       - Long numbers (>= 5 digits, e.g. phone/PIN: 9487203) indexed directly
       - Street/house numbers combined with address locality words (e.g. 684_ghaziabad, 85_ticonderoga)
    4. Address locality bigrams (e.g. banipark_jaipur) to handle non-Latin script DBAs/transliterations
    """
    clean_name, core_name = normalize_business_name(name)
    norm_address = normalize_address(address)
    norm_country = normalize_country(country)
    
    keys = set()
    country_prefix = f"C:{norm_country}|" if norm_country else "C:UNK|"
    
    # 1. Name prefixes & significant tokens
    if core_name:
        condensed_name = re.sub(r'\s+', '', core_name)
        if len(condensed_name) >= 3:
            keys.add(country_prefix + f"NPREF4:{condensed_name[:4]}")
            keys.add(country_prefix + f"NPREF3:{condensed_name[:3]}")
            
        name_tokens = [t for t in core_name.split() if len(t) >= 3]
        for token in name_tokens[:3]:
            keys.add(country_prefix + f"NTOK:{token}")
            
    # 2. Address-based blocking (Crucial for DBAs, typos, and Indian/French script transliterations)
    if norm_address:
        # Extract all numbers and strip leading zeros
        raw_numbers = re.findall(r'\b\d+\b', norm_address)
        norm_numbers = [re.sub(r'^0+', '', n) for n in raw_numbers]
        norm_numbers = [n for n in norm_numbers if n] # remove empty
        
        # Extract significant address words (excluding common generic stopwords)
        addr_words = [
            w for w in norm_address.split() 
            if len(w) >= 4 and not w.isdigit() and w not in COMMON_ADDR_STOPWORDS
        ]
        
        for num in set(norm_numbers):
            # Discriminative long numbers (PIN codes, phone numbers, unique IDs)
            if len(num) >= 5:
                keys.add(country_prefix + f"NUM_LONG:{num}")
            else:
                # Pair house/plot number with significant address locality words
                for w in addr_words[:3]:
                    keys.add(country_prefix + f"NUM_WORD:{num}_{w}")
                    
        # Locality word pairs (handles cases without common numbers)
        if len(addr_words) >= 2:
            keys.add(country_prefix + f"LOC_PAIR:{addr_words[0]}_{addr_words[1]}")
            if len(addr_words) >= 3:
                keys.add(country_prefix + f"LOC_PAIR:{addr_words[0]}_{addr_words[2]}")
                
    # Deterministic prioritization: strict keys first, longer tokens first, deterministic tie-break
    return sorted(keys, key=lambda k: (get_blocking_key_priority(k), -len(k), k))


class BlockingIndex:
    """
    Inverted Index for high-recall candidate generation.
    Stores blocking_key -> list of candidate entity IDs.
    """
    def __init__(self, max_bucket_size=1000):
        self.index = defaultdict(list)
        self.max_bucket_size = max_bucket_size
        
    def add_record(self, entity_id, name, address, country):
        keys = extract_blocking_keys(name, address, country)
        for k in keys:
            bucket = self.index[k]
            if len(bucket) < self.max_bucket_size:
                bucket.append(entity_id)

    def get_candidates_for_record(self, name, address, country, max_candidates=200):
        keys = extract_blocking_keys(name, address, country)
        candidates = set()
        
        for k in keys:
            if k in self.index:
                bucket = self.index[k]
                for cid in bucket:
                    candidates.add(cid)
                    if len(candidates) >= max_candidates:
                        break
            if len(candidates) >= max_candidates:
                break
                
        return candidates
