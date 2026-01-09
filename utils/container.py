# utils/container.py
import re
from typing import Optional
 
 
def extract_container_number(text: str) -> Optional[str]:
    """
    Detect a container number in free‑form text.
    Handles:
        - "container ABCD1234567"
        - the raw pattern "ABCD1234567"
        - purely numerical container numbers like "8663022072"
    Returns the cleaned, upper‑cased number or None.
    
    Avoids false positives from consignee codes, user IDs, etc.
    """
    # **CRITICAL**: Skip numbers that appear in consignee/user context
    # Match patterns like "consignee 0028664", "user 0028664", "for consignee 0028664"
    if re.search(r'\b(?:consignee|user)(?:\s+code)?\s+\d{7}', text, re.IGNORECASE):
        # Extract the consignee number to exclude it
        consignee_match = re.search(r'(?:consignee|user)(?:\s+code)?\s+(\d{7})', text, re.IGNORECASE)
        if consignee_match:
            consignee_num = consignee_match.group(1)
            # Remove it from consideration
            text = text.replace(consignee_num, '')
    
    patterns = [
        r'container\s+([A-Z]{4}\d{7})',      # "container" + 4 letters + 7 digits (ISO standard)
        r'container\s+(\d{7,12})',           # "container" + 7-12 digit number
        r'\b([A-Z]{4}\d{7})\b',              # just the raw 4‑letters + 7‑digits (with word boundaries)
        r'\b(\d{7,12})\b'                    # purely numerical (7-12 digits) with word boundaries
    ]
    for pat in patterns:
        m = re.search(pat, text, flags=re.IGNORECASE)
        if m:
            candidate = re.sub(r'[^A-Z0-9]', '', m.group(1).upper())
            # Validate: must be at least 7 characters and not just from word "containers" or "container"
            if len(candidate) >= 7:
                return candidate
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
