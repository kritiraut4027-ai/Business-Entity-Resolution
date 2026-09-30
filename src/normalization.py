import re
import unicodedata
import pandas as pd

# Legal entity suffixes across US, India, France, and generic international
LEGAL_SUFFIX_PATTERNS = [
    # Multi-word Indian suffixes
    r'\b(pvt|private)\s+(ltd|limited)\b',
    r'\bpublic\s+limited\b',
    r'\bllp\b',
    r'\bltd\b',
    r'\blimited\b',
    r'\bco\s+operative\b',
    r'\bcooperative\b',
    # French suffixes
    r'\bsarl\b',
    r'\bsas\b',
    r'\bsa\b',
    r'\beurl\b',
    r'\bsci\b',
    r'\bsnc\b',
    # US & General English suffixes
    r'\bincorporated\b',
    r'\binc\b',
    r'\bcorporation\b',
    r'\bcorp\b',
    r'\bcompany\b',
    r'\bco\b',
    r'\bllc\b',
    r'\bl\.l\.c\b',
    r'\blc\b',
    r'\blp\b',
    r'\bgroup\b',
    r'\bholdings\b',
    r'\benterprises\b',
    r'\bservices\b'
]
LEGAL_SUFFIX_REGEX = re.compile('|'.join(LEGAL_SUFFIX_PATTERNS), re.IGNORECASE)

# Domain name patterns
DOMAIN_REGEX = re.compile(
    r'^(https?:\/\/)?(www\.)?([a-zA-Z0-9_\-]+)\.(com|org|net|in|fr|co|io|biz|info|ai|tech|us|me)(\/.*)?$',
    re.IGNORECASE
)

# Address abbreviations dictionary
ADDRESS_ABBREVIATIONS = {
    r'\bave\b': 'avenue',
    r'\bav\b': 'avenue',
    r'\bst\b': 'street',
    r'\brd\b': 'road',
    r'\bblvd\b': 'boulevard',
    r'\bdr\b': 'drive',
    r'\bln\b': 'lane',
    r'\bhwy\b': 'highway',
    r'\bct\b': 'court',
    r'\btwp\b': 'township',
    r'\bpl\b': 'place',
    r'\bsq\b': 'square',
    r'\bste\b': 'suite',
    r'\bapt\b': 'apartment',
    r'\bdept\b': 'department',
    r'\bfl\b': 'floor',
    r'\bpk\b': 'park',
    r'\bpkwy\b': 'parkway',
    r'\bctr\b': 'center',
    r'\bpo\s+box\b': 'pobox',
    r'\bp\.o\.\s*box\b': 'pobox',
    # Ordinal mappings
    r'\bfirst\b': '1st',
    r'\bsecond\b': '2nd',
    r'\bthird\b': '3rd',
    r'\bfourth\b': '4th',
    r'\bfifth\b': '5th',
    r'\bsixth\b': '6th',
    r'\bseventh\b': '7th',
    r'\beighth\b': '8th',
    r'\bninth\b': '9th',
    r'\btenth\b': '10th'
}

STATE_MAPPINGS = {
    # US Common States
    r'\bny\b': 'new york',
    r'\bca\b': 'california',
    r'\btx\b': 'texas',
    r'\bnc\b': 'north carolina',
    r'\bok\b': 'oklahoma',
    r'\bfl\b': 'florida',
    r'\bil\b': 'illinois',
    r'\boh\b': 'ohio',
    r'\bpa\b': 'pennsylvania',
    r'\bga\b': 'georgia',
    r'\bmi\b': 'michigan',
    r'\bnj\b': 'new jersey',
    r'\bva\b': 'virginia',
    r'\bwa\b': 'washington',
    r'\bma\b': 'massachusetts',
    # India Common States / Territories
    r'\bdl\b': 'delhi',
    r'\btn\b': 'tamil nadu',
    r'\bka\b': 'karnataka',
    r'\bmh\b': 'maharashtra',
    r'\bhr\b': 'haryana',
    r'\bup\b': 'uttar pradesh',
    r'\bwb\b': 'west bengal',
    r'\bgj\b': 'gujarat',
    r'\brj\b': 'rajasthan',
    r'\bts\b': 'telangana',
    r'\bap\b': 'andhra pradesh',
    r'\bkl\b': 'kerala'
}


def clean_unicode_and_spaces(text):
    """Normalize unicode, lowercase, remove control characters and clean spaces."""
    if text is None or pd.isna(text):
        return ""
    text = str(text).strip()
    # Normalize unicode accents by removing combining marks (e.g. e + accent -> e)
    text = "".join(c for c in unicodedata.normalize('NFD', text) if unicodedata.category(c) != 'Mn')
    # Lowercase
    text = text.lower()
    # Replace non-word / non-space punctuation with single space
    # Keeps alphanumeric characters across languages (Devanagari, Latin, etc.)
    text = re.sub(r'[^\w\s]', ' ', text)
    # Collapse multiple whitespaces
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def normalize_business_name(name):
    """
    Normalizes business name:
    1. Checks if it's a domain name (e.g. maurewilliamscolombier.com -> maurewilliamscolombier)
    2. Cleans punctuation, unicode, case
    3. Strips standard corporate legal suffixes
    Returns (cleaned_name, core_name) where core_name has legal suffixes removed.
    """
    if name is None or pd.isna(name):
        return "", ""

    raw_text = str(name).strip()
    # Check domain name pattern
    domain_match = DOMAIN_REGEX.match(raw_text)
    if domain_match:
        raw_text = domain_match.group(3)

    cleaned = clean_unicode_and_spaces(raw_text)

    # Core name: remove legal suffixes
    core = LEGAL_SUFFIX_REGEX.sub('', cleaned)
    core = re.sub(r'\s+', ' ', core).strip()

    # Fallback to cleaned if suffix stripping removed everything
    if not core:
        core = cleaned

    return cleaned, core


def normalize_address(address):
    """
    Normalizes address text:
    1. Lowercase, Unicode NFKD
    2. Expand common street / unit abbreviations
    3. Expand common state abbreviations
    4. Clean punctuation and spaces
    Returns normalized string.
    """
    if address is None or pd.isna(address):
        return ""

    cleaned = clean_unicode_and_spaces(address)
    if not cleaned:
        return ""

    # Replace road/building abbreviations
    for pattern, replacement in ADDRESS_ABBREVIATIONS.items():
        cleaned = re.sub(pattern, replacement, cleaned)

    # Replace common state abbreviations
    for pattern, replacement in STATE_MAPPINGS.items():
        cleaned = re.sub(pattern, replacement, cleaned)

    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    return cleaned


def normalize_country(country):
    """
    Standardize country labels as an open-set string:
    - US / USA / United States -> US
    - India / IN -> India
    - France / FR -> France
    - Any other country is stripped and capitalized.
    """
    if country is None or pd.isna(country):
        return ""

    c = str(country).strip()
    c_lower = c.lower()

    if c_lower in ['us', 'usa', 'united states', 'united states of america']:
        return 'US'
    if c_lower in ['india', 'in', 'ind']:
        return 'India'
    if c_lower in ['france', 'fr', 'fra']:
        return 'France'

    # Open-set general formatting
    return c.title()
