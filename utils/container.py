# utils/container.py
import re
from typing import Optional


def extract_container_number(text: str) -> Optional[str]:
    """
    Detect a container number in free‑form text.
    Handles:
        - "container ABCD1234567"
        - the raw pattern "ABCD1234567" (ISO standard)
        - numeric-only if prefixed with "container"

    Returns the cleaned, upper‑cased number or None.

    Avoids false positives from consignee codes, user IDs, etc.
    """
    if not text:
        return None

    # **CRITICAL**: Skip numbers that appear in consignee/user context
    # Match patterns like "consignee 0028664", "user 0028664", "for consignee 0028664"
    if re.search(r"\b(?:consignee|user)(?:\s+code)?\s+\d{7}", text, re.IGNORECASE):
        # Extract the consignee number to exclude it
        consignee_match = re.search(
            r"(?:consignee|user)(?:\s+code)?\s+(\d{7})", text, re.IGNORECASE
        )
        if consignee_match:
            consignee_num = consignee_match.group(1)
            # Remove it from consideration by replacing with placeholder
            text = text.replace(consignee_num, "CONSIGNEE_CODE_REMOVED")

    # Priority 1: ISO Standard (4 letters + 7 digits) - Most reliable
    # e.g., MSCU1234567
    iso_match = re.search(r"\b([A-Z]{4}\d{7})\b", text, flags=re.IGNORECASE)
    if iso_match:
        return iso_match.group(1).upper()

    # Priority 2: Explicit "container" prefix + numeric
    # e.g., "container 1234567"
    prefix_match = re.search(
        r"container\s+(?:number\s+)?([A-Z0-9]{7,12})", text, flags=re.IGNORECASE
    )
    if prefix_match:
        return re.sub(r"[^A-Z0-9]", "", prefix_match.group(1).upper())

    # Note: Removed the "raw 7-12 digit" regex `r'\b(\d{7,12})\b'` without prefix.
    # It causes massive overlap with PO numbers and phone numbers.
    # If a user types just numbers, it is assumed to be a PO or BL unless explicitly labelled "container".

    return None


def extract_po_number(text: str) -> Optional[str]:
    """
    Detect a PO number in free-form text.
    Handles:
        - “PO 1234567890”
        - “purchase order 1234567890”
        - raw pattern “1234567890” (6-20 digits)
    Returns the cleaned number or None.
    """
    if not text:
        return None

    patterns = [
        r"po\s+(?:number\s+)?([0-9]{6,})",  # with the word “PO”
        r"purchase order\s+([0-9]{6,})",  # with the phrase “purchase order”
        # Fallback: Just numbers, but avoid common dates (8 digits starting with 19/20) if possible?
        # For now, keep it simple but greedy.
        r"\b([0-9]{6,20})\b",  # just the raw number (6-20 digits)
    ]
    for pat in patterns:
        m = re.search(pat, text, flags=re.IGNORECASE)
        if m:
            return re.sub(r"\D", "", m.group(1))
    return None


def extract_ocean_bl_number(text: str) -> Optional[str]:
    """
    Detect an Ocean BL number in free-form text.
    Handles:
        - "ocean bl HDMUJKTM12358602"
        - "bill of lading 2260329046"
        - alphanumeric pattern (4+ letters + 6+ digits) if strict
    Returns the cleaned number or None.
    """
    if not text:
        return None

    patterns = [
        r"ocean bl\s+([A-Z0-9]+)",  # with the phrase "ocean bl"
        r"bill of lading\s+([A-Z0-9]+)",  # with the phrase "bill of lading"
        r"(?:bl|b/l)\s+([A-Z0-9]+)",  # "BL" or "B/L"
        # Strict alphanumeric: Must have letters AND numbers to distinguish from POs
        r"\b([A-Z]{4,}\d{6,})\b",
    ]
    for pat in patterns:
        m = re.search(pat, text, flags=re.IGNORECASE)
        if m:
            return re.sub(r"[^A-Z0-9]", "", m.group(1).upper())

    # Note: We do NOT match pure numbers here as `extract_po_number` covers that.
    # If a BL is purely numeric, the user MUST prefix it with "BL" or "Bill of lading" to distinguish from PO.

    return None


def extract_booking_number(text: str) -> Optional[str]:
    """
    Detect a booking number in free-form text.
    Handles:
        - "booking EG2002468"
        - "booking number CN2229273"
        - "bkg EG2002468"
        - alphanumeric pattern (2-4 letters + 6-15 digits/letters)
    Returns the cleaned, upper-cased booking number or None.

    Note: Booking numbers are typically alphanumeric (mix of letters and numbers)
    and are 6-20 characters long. They differ from container numbers which are
    exactly 4 letters + 7 digits.
    """
    if not text:
        return None

    # Priority 1: Explicit "booking" or "bkg" prefix
    patterns = [
        r"booking\s+(?:number\s+)?(?:no\.?\s+)?([A-Z0-9]{6,20})",  # "booking EG2002468"
        r"bkg\s+(?:number\s+)?(?:no\.?\s+)?([A-Z0-9]{6,20})",  # "bkg EG2002468"
        r"booking#\s*([A-Z0-9]{6,20})",  # "booking#EG2002468"
        r"bkg#\s*([A-Z0-9]{6,20})",  # "bkg#EG2002468"
    ]

    for pat in patterns:
        m = re.search(pat, text, flags=re.IGNORECASE)
        if m:
            return m.group(1).upper()

    # Priority 2: Alphanumeric pattern (NOT matching container format)
    # Booking numbers typically have letters + numbers mixed, 6-20 chars
    # Exclude container format (exactly 4 letters + 7 digits)
    # Pattern: 2-4 letters followed by 6-15 digits/letters OR similar variations
    alphanumeric_pattern = r"\b([A-Z]{2,4}\d{6,15})\b"
    m = re.search(alphanumeric_pattern, text, flags=re.IGNORECASE)
    if m:
        candidate = m.group(1).upper()
        # Exclude if it matches container format exactly (4 letters + 7 digits)
        if not re.fullmatch(r"[A-Z]{4}\d{7}", candidate):
            # Additional check: must have at least some digits
            if re.search(r"\d", candidate):
                return candidate

    return None


def extract_supplier_vendor_name(text: str) -> Optional[str]:
    """
    Detect a supplier/vendor name in free-form text.
    Handles:
        - "supplier ABC Corporation"
        - "vendor XYZ Ltd"
        - "from Acme Industries"
        - "shipper Global Logistics"
    Returns the extracted name (cleaned and title-cased) or None.
    """
    if not text:
        return None

    patterns = [
        r"supplier\s+(?:name\s+)?(?:is\s+)?([A-Za-z0-9\s\.,&\-']+?)(?:\s+(?:for|with|has|is|on|in|at|,)|$)",
        r"vendor\s+(?:name\s+)?(?:is\s+)?([A-Za-z0-9\s\.,&\-']+?)(?:\s+(?:for|with|has|is|on|in|at|,)|$)",
        r"from\s+(?:supplier\s+)?(?:vendor\s+)?([A-Za-z0-9\s\.,&\-']+?)(?:\s+(?:for|with|has|is|on|in|at|,)|$)",
        r"shipper\s+(?:name\s+)?(?:is\s+)?([A-Za-z0-9\s\.,&\-']+?)(?:\s+(?:for|with|has|is|on|in|at|,)|$)",
    ]

    for pat in patterns:
        m = re.search(pat, text, flags=re.IGNORECASE)
        if m:
            # Extract and clean the name
            name = m.group(1).strip()
            # Remove trailing punctuation
            name = re.sub(r"[,.\s]+$", "", name)
            # Clean up extra whitespace
            name = re.sub(r"\s+", " ", name)
            # Return if valid length (at least 2 characters)
            if len(name) >= 2:
                return name.title()

    return None
