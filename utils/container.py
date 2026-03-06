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


def extract_job_no(text: str) -> Optional[str]:
    """
    Detect a job number in free-form text.
    Handles:
        - "job number 1234567"
        - "job no 1234567"
        - "job 1234567"
        - "job# 1234567"
        - alphanumeric patterns (6-20 characters)
    Returns the cleaned, upper-cased job number or None.

    Note: Job numbers can be purely numeric or alphanumeric.
    This function prioritizes explicit "job" prefixes to avoid false positives.
    """
    if not text:
        return None

    # Priority 1: Explicit "job" prefix with various formats
    patterns = [
        r"job\s+(?:number\s+)?(?:no\.?\s+)?([A-Z0-9]{6,20})",  # "job number 1234567" or "job no. 1234567"
        r"job#\s*([A-Z0-9]{6,20})",  # "job#1234567"
        r"job-([A-Z0-9]{6,20})",  # "job-1234567"
    ]

    for pat in patterns:
        m = re.search(pat, text, flags=re.IGNORECASE)
        if m:
            return m.group(1).upper()

    # Priority 2: If asking about job associated with other identifiers,
    # we don't extract standalone numbers to avoid confusion with container/PO/BL numbers
    # The job number should be explicitly marked with "job" keyword in such queries

    return None


def extract_vessel_name(text: str) -> Optional[str]:
    """
    Detect a vessel name in free-form text.
    Handles:
        - "vessel MAERSK FORTALEZA"
        - "ship MSC FLAMINIA"
        - "MAERSK FORTALEZA"
        - "which jobs associated with MAERSK FORTALEZA"
        - "vessel name CMA CGM BALBOA"
    Returns the extracted vessel name (cleaned and upper-cased) or None.

    Note: Vessel names are typically multiple words, all caps in databases.
    This function looks for explicit vessel/ship keywords or standalone caps patterns.
    """
    if not text:
        return None

    # Priority 1: Explicit vessel/ship prefix
    patterns = [
        r"vessel\s+(?:name\s+)?(?:is\s+)?([A-Z][A-Z0-9\s\-']+?)(?:\s+(?:is|has|at|in|on|for|with|,|:)|$)",
        r"ship\s+(?:name\s+)?(?:is\s+)?([A-Z][A-Z0-9\s\-']+?)(?:\s+(?:is|has|at|in|on|for|with|,|:)|$)",
    ]

    for pat in patterns:
        m = re.search(pat, text, flags=re.IGNORECASE)
        if m:
            # Extract and clean the name
            name = m.group(1).strip()
            # Remove trailing punctuation and common stopwords
            name = re.sub(r"[,.:;\s]+$", "", name)
            # Clean up extra whitespace
            name = re.sub(r"\s+", " ", name)
            # Return if valid length (at least 3 characters, typically vessel names are longer)
            if len(name) >= 3:
                return name.upper()

    # Priority 2: "associated with", "with", "for" followed by vessel name
    # Handles: "which jobs associated with MAERSK FORTALEZA", "jobs for MSC FLAMINIA"
    associated_patterns = [
        r"associated\s+with\s+([A-Z][A-Z0-9\s\-']{5,})(?:\s*[,;]|$)",
        r"(?:for|with)\s+([A-Z][A-Z0-9\s\-']{5,})(?:\s*[,;]|$)",
    ]

    for pat in associated_patterns:
        m = re.search(pat, text, flags=re.IGNORECASE)
        if m:
            name = m.group(1).strip()
            # Remove trailing punctuation
            name = re.sub(r"[,.:;\s]+$", "", name)
            # Clean up extra whitespace
            name = re.sub(r"\s+", " ", name)
            name_upper = name.upper()

            # Validate: Must be 2+ words of 3+ chars each (typical vessel name pattern)
            words = name_upper.split()
            if len(words) >= 2 and all(len(w) >= 3 for w in words):
                # Exclude common question/query words
                excluded_words = {
                    "WHICH",
                    "WHAT",
                    "WHERE",
                    "WHEN",
                    "WHO",
                    "HOW",
                    "JOB",
                    "NUMBER",
                    "CONTAINER",
                    "BOOKING",
                    "STATUS",
                    "THE",
                    "AND",
                    "FOR",
                    "WITH",
                    "ASSOCIATED",
                }
                # Check if the name contains mostly non-excluded words
                valid_words = [w for w in words if w not in excluded_words]
                if len(valid_words) >= 2:  # At least 2 valid words
                    return name_upper

    # Priority 3: Standalone all-caps pattern (2+ words, each 3+ chars)
    # Examples: "MAERSK FORTALEZA", "MSC FLAMINIA", "EVER GIVEN"
    # Pattern: 2+ consecutive capitalized words - but exclude question words
    text_upper = text.upper()

    # Find all potential vessel name patterns (2+ consecutive uppercase words)
    matches = re.finditer(r"\b([A-Z]{3,}(?:\s+[A-Z]{3,})+)\b", text_upper)

    for match in matches:
        name = match.group(1).strip()

        # Basic length check
        if len(name) < 7:
            continue

        # Exclude common non-vessel phrases
        if name in [
            "USA",
            "EUR",
            "USD",
            "PO NUMBER",
            "CONTAINER NUMBER",
            "JOB NUMBER",
            "BOOKING NUMBER",
            "OCEAN BL",
        ]:
            continue

        # Split into words and validate
        words = name.split()

        # Must be 2+ words
        if len(words) < 2:
            continue

        # Exclude if it contains too many question/query words
        excluded_words = {
            "WHICH",
            "WHAT",
            "WHERE",
            "WHEN",
            "WHO",
            "HOW",
            "JOB",
            "NUMBER",
            "CONTAINER",
            "BOOKING",
            "STATUS",
            "IS",
            "ARE",
            "WAS",
            "WERE",
            "THE",
            "AND",
            "OR",
            "ASSOCIATED",
            "WITH",
            "FOR",
        }

        # Count excluded words
        excluded_count = sum(1 for w in words if w in excluded_words)

        # If more than half the words are excluded, skip this match
        if excluded_count > len(words) // 2:
            continue

        # If at least 2 valid words remain, this is likely a vessel name
        valid_words = [w for w in words if w not in excluded_words]
        if len(valid_words) >= 2:
            return name

    return None
