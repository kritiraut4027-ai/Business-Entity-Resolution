# ML Challenge 2026: Business Entity Resolution Solution

**Team Name:** AI Engineering Team  
**Submission Date:** September 2026  

---

## 1. Executive Summary
We designed and implemented a two-stage Business Entity Resolution framework capable of resolving entities across 26 million noisy multi-source records under stringent memory constraints. Our architecture pairs an inverted multi-strategy locality and prefix blocking engine (achieving 78.08% candidate recall and eliminating >99.999% of the Cartesian search space) with an 18-feature supervised LightGBM binary classifier tuned for the precision-weighted Macro F0.5 metric, achieving a validation Macro F0.5 score of **0.8109** with **100% singleton accuracy**.

---

## 2. Methodology

### 2.1 Problem Analysis
During our empirical exploratory data audit of the 2.4 GB dataset across US, India, and France records, we identified critical real-world noise patterns:
1. **Transliteration & Multi-Script Mismatch:** Source 1 contains English transliterations for Indian entities, whereas Sources 2 and 3 frequently store business names in native Indic scripts (Hindi/Devanagari, Telugu). Direct string distance on names between scripts yields 0 similarity, necessitating address-based anchor blocking.
2. **Missing Field Imbalance:** Address is missing (`NaN`) in a substantial percentage of Source 2 and Source 3 records. Address features must gracefully accommodate missing values without corrupting model scores.
3. **Leading Zeros in Numerical Components:** Municipal street and plot numbers differ by formatting (e.g., `AF-0684` vs `Af-684`).
4. **Ordinals and Legal Suffixes:** Suffixes vary across jurisdictions (`LLP`, `Pvt Ltd`, `Private Limited`, `SARL`, `SAS`, `Inc`, `LLC`), and address ordinals vary between words and digits (`10th` vs `Tenth`).

### 2.2 Solution Strategy
**Approach Type:** Multi-Strategy Inverted Blocking + Precision-Calibrated Gradient Boosted Trees (LightGBM).  
**Core Innovation:** A cross-lingual locality anchor blocking scheme that pairs normalized numerical components (stripped of leading zeros) with locality bigrams and phone/PIN numbers to bridge script gaps, combined with an asymmetric F0.5 decision threshold that strictly eliminates singleton false merges.

---

## 3. Candidate Generation (Blocking)
To reduce 22.4 trillion potential pairs down to a tractable pool:
- **Blocking keys used:**
  1. `Country` + 4-char and 3-char core name prefix (`NPREF4`, `NPREF3`)
  2. `Country` + significant core name tokens (`NTOK`)
  3. `Country` + High-entropy numbers (>= 5 digits, e.g., phone numbers `9487203`, PIN codes)
  4. `Country` + Street numbers (leading zeros stripped) paired with significant address locality words (`NUM_WORD:684_ghaziabad`, `NUM_WORD:85_ticonderoga`)
  5. `Country` + Locality bigrams (`LOC_PAIR:banipark_jaipur`)
- **Candidate pairs generated:** Average 52.77 candidates per Source 1 entity (Reduction ratio: >99.999%).
- **Ensuring true matches were not lost:** Multi-index inverted posting lists index complementary attributes so that if a name fails (due to typos or script transliteration), the address anchor captures the entity, yielding **78.08% candidate recall** on full-coverage ground-truth benchmarks.

---

## 4. Matching Model

**Features used (18 total):**
- **Name features:** Levenshtein ratio, Core legal-stripped ratio, Token sort ratio (word order permutations), Token set ratio (extra words/suffixes), Partial ratio (substrings/DBAs), Jaro-Winkler, Core Jaro-Winkler, Jaccard token overlap, Common tokens count, Character length delta.
- **Address features:** Missing address indicator, Levenshtein ratio, Token set ratio, Partial ratio, Token Jaccard overlap, Common address tokens, Shared number match indicator, Number Jaccard overlap.
- **Geographic features:** Normalized open-set country equality.

**Model type:** LightGBM Gradient Boosted Decision Trees (150 estimators, learning_rate=0.08, num_leaves=31, balanced class weights).  
**Threshold selection method:** Grid sweep across `[0.50, 0.60, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]` on held-out validation entities evaluating entity-level Macro F0.5. Optimal threshold **0.95** was selected to maximize precision on singletons.

---

## 5. Results & Error Analysis

- **Macro F0.5 Score:** **0.8109** (Validation Macro Precision: 0.8569, Macro Recall: 0.7364, Singleton Accuracy: 100.0%).
- **Common false positives (wrong merges):** Franchises and branches of nationwide chains sharing identical names in different postal zones when address was missing. Calibrating threshold to 0.95 effectively neutralized these false positives.
- **Common false negatives (missed matches):** Entities where both the name was a complete DBA alias and the address was completely missing (`NaN`) in Source 2/3, leaving no shared surface signals.

---

## 6. Conclusion
By systematically diagnosing actual ground-truth noise patterns (transliterations, leading zero padding, missing addresses), we created a robust, scalable entity resolution pipeline. The system passes all official submission constraints, conforms to the MIT/Apache license requirements, and achieves superior precision with an 0.8109 Macro F0.5 score.

---

## Appendix

### A. Code Artefacts
- `src/normalization.py`: Unicode-safe, diacritic-preserving text cleaner with multi-country legal suffix and ordinal parsers.
- `src/blocking.py`: High-recall inverted index blocking engine.
- `src/features.py`: 18 C++-accelerated string, token, and numeric feature extractors using RapidFuzz.
- `src/train.py`: Supervised model training, validation split, and F0.5 threshold tuner.
- `src/predict.py`: Memory-efficient streaming inference engine producing `matching_results.tsv` and `candidate_pairs.tsv`.
- `utils/validate_submission.py`: Official submission validation checking format, IDs, duplicates, and subset constraints.
