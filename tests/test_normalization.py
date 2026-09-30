import sys
import os

# Add src to python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from normalization import normalize_business_name, normalize_address, normalize_country

def test_normalization():
    print("Testing Normalization...")
    
    # Test Business Names
    cases_name = [
        ("Maure Williams Colombier Inc", "maure williams colombier inc", "maure williams colombier"),
        ("maurewilliamscolombier.com", "maurewilliamscolombier", "maurewilliamscolombier"),
        ("Raj Investments LLP", "raj investments llp", "raj investments"),
        ("Marina École France Sàrl", "marina ecole france sarl", "marina ecole france"),
        ("International South Consultants Private Ltd", "international south consultants private ltd", "international south consultants"),
        (None, "", "")
    ]
    
    for raw, expected_clean, expected_core in cases_name:
        clean, core = normalize_business_name(raw)
        assert clean == expected_clean, f"Clean mismatch: expected '{expected_clean}', got '{clean}'"
        assert core == expected_core, f"Core mismatch: expected '{expected_core}', got '{core}'"
        print(f"  [PASS] Name: '{raw}' -> Clean: '{clean}' | Core: '{core}'")
        
    # Test Addresses
    cases_addr = [
        ("85 Wayne Avenue, Ticonderoga, NY", "85 wayne avenue ticonderoga new york"),
        ("Wayne Ave, Ticonderoga Townshiip, New York", "wayne avenue ticonderoga townshiip new york"),
        ("KH NO. -570/13, NEW DELHI, WEST DELHI, DL", "kh no 570 13 new delhi west delhi delhi"),
        (None, "")
    ]
    
    for raw, expected in cases_addr:
        addr = normalize_address(raw)
        assert addr == expected, f"Addr mismatch: expected '{expected}', got '{addr}'"
        print(f"  [PASS] Addr: '{raw}' -> '{addr}'")

    # Test Country
    assert normalize_country("US") == "US"
    assert normalize_country("united states") == "US"
    assert normalize_country("india") == "India"
    assert normalize_country("france") == "France"
    assert normalize_country("germany") == "Germany"
    print("  [PASS] Country normalization tests")
    
    print("\nALL NORMALIZATION TESTS PASSED SUCCESSFULLY!")

if __name__ == "__main__":
    test_normalization()
