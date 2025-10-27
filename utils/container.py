# utils/container.py
import re
from typing import Optional
 
 
def extract_container_number(text: str) -> Optional[str]:
    """
    Detect a container number in free‑form text.
    Handles:
        - “container ABCD1234567”
        - the raw pattern “ABCD1234567”
        - purely numerical container numbers like “8663022072”
    Returns the cleaned, upper‑cased number or None.
    """
    patterns = [
        r'container\s*([A-Z0-9]+)',          # with the word “container”
        r'([A-Z]{4,6}\d{7,20})',                  # just the raw 4‑letters + 7‑digits
        r'\b(\d{7,20})\b'                    # purely numerical (7-20 digits, adjust as needed)
    ]
    for pat in patterns:
        m = re.search(pat, text, flags=re.IGNORECASE)
        if m:
            return re.sub(r'[^A-Z0-9]', '', m.group(1).upper())
    return None
 
 
def extract_po_number(text: str) -> Optional[str]:
    """
    Detect a PO number in free-form text.
    Handles:
        - “PO 1234567890”
        - “purchase order 1234567890”
        - raw pattern “1234567890” (10+ digits)
    Returns the cleaned number or None.
    """
    patterns = [
        r'po\s*([0-9]{6,})',                  # with the word “PO”
        r'purchase order\s*([0-9]{6,})',      # with the phrase “purchase order”
        r'\b([0-9]{6,})\b'                    # just the raw number (6+ digits)
    ]
    for pat in patterns:
        m = re.search(pat, text, flags=re.IGNORECASE)
        if m:
            return re.sub(r'\D', '', m.group(1))
    return None
 
 
def extract_ocean_bl_number(text: str) -> Optional[str]:
    """
    Detect an Ocean BL number in free-form text.
    Handles:
        - "ocean bl HDMUJKTM12358602"
        - "bill of lading 2260329046"
        - alphanumeric pattern "HDMUJKTM12358602"
        - purely numerical pattern "2260329046" or "17418207"
    Returns the cleaned number or None.
    """
    patterns = [
        r'ocean bl\s*([A-Z0-9]+)',            # with the phrase "ocean bl"
        r'bill of lading\s*([A-Z0-9]+)',      # with the phrase "bill of lading"
        r'([A-Z]{4,}\d{6,})',                 # alphanumeric pattern (4+ letters + 6+ digits)
        r'\b([0-9]{6,})\b'                    # purely numerical (6+ digits)
    ]
    for pat in patterns:
        m = re.search(pat, text, flags=re.IGNORECASE)
        if m:
            return re.sub(r'[^A-Z0-9]', '', m.group(1).upper())
    return None
