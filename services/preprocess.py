# services/preprocess.py

# Comprehensive data preprocessing for shipment data

import logging
import re
import pandas as pd
import warnings
from datetime import datetime

logger = logging.getLogger("shipping_chatbot")


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocesses shipment data with comprehensive date parsing,
    duplicate removal, and data normalization.
    """
    logger.info(f"Starting preprocessing on {len(df)} rows")
    original_row_count = len(df)

    # ── 1️ Drop duplicates ─────────────────────
    df = df.drop_duplicates()
    logger.info(f"Removed {original_row_count - len(df)} duplicate rows")

    # ── 2️ Normalise column names ───────────────
    df.columns = [c.strip().lower() for c in df.columns]

    # ── 3️ Convert ALL known date columns ───────────
    # List of date-related columns in shipment data
    date_columns = [
        # Empty Container Management
        'empty_container_dispatch_date',    # When empty container was dispatched
        
        # Gate & Receipt Operations
        'in_gate_date',                     # In-gate at facility
        
        # Vehicle Movement Timeline
        'carrier_vehicle_load_date',        # When container loaded onto vehicle
        'vehicle_departure_date',           # When vehicle departed
        'vehicle_arrival_date',             # When vehicle arrived
        'carrier_vehicle_unload_date',      # When container unloaded from vehicle
        
        # Rail Transport Timeline
        'rail_load_dp_date',                # Rail load at discharge port
        'rail_departure_dp_date',           # Rail departure from discharge port
        'out_gate_date_from_dp',            # Out gate from discharge port
        'rail_arrival_destination_date',    # Rail arrival at destination
        
        # Load Port Operations
        'etd_lp',                           # Estimated Time of Departure - Load Port
        'atd_lp',                           # Actual Time of Departure - Load Port
        'cargo_received_date_multiple',     # Cargo received at load port
        
        # First Load Port Operations
        'etd_flp',                          # Estimated Time of Departure - First Load Port
        'atd_flp',                          # Actual Time of Departure - First Load Port
        'ata_flp',                          # Actual Time of Arrival - First Load Port
        
        # Discharge Port Operations
        'eta_dp',                           # Estimated Time of Arrival - Discharge Port
        'ata_dp',                           # Actual Time of Arrival - Discharge Port
        'derived_ata_dp',                   # Derived Actual Time of Arrival - Discharge Port
        'revised_eta',                      # Revised ETA - Discharge Port
        'predictive_eta',                   # Predictive ETA - Discharge Port
        
        # Final Destination Operations
        'eta_fd',                           # Estimated Time of Arrival - Final Destination
        'revised_eta_fd',                   # Revised ETA - Final Destination
        'predictive_eta_fd',                # Predictive ETA - Final Destination
        
        # Container Yard (CY) Operations
        'equipment_arrived_at_last_cy',     # Container arrival at last CY
        'out_gate_at_last_cy',              # Out gate from last CY
        
        # Final Delivery & Return
        'delivery_date_to_consignee',       # Final delivery to consignee
        'empty_container_return_date',      # Empty container return
        
        # Documentation & Cargo Readiness
        'cargo_ready_date',                 # Cargo ready for shipment
        'in-dc_date',                       # In-DC (Distribution Center) date
        'get_isf_submission_date',          # ISF (Importer Security Filing) submission date
    ]
    
    # Convert only columns that exist in the dataframe
    converted_count = 0
    
    # Suppress pandas warnings for date parsing once
    warnings.filterwarnings("ignore", message=r"Could not infer format", category=UserWarning)
    warnings.filterwarnings("ignore", message=r"In a future version of pandas", category=FutureWarning)

    for col in date_columns:
        if col in df.columns:
            try:
                # Use errors='coerce' to turn invalid parsing into NaT (Not a Time)
                # This is CRITICAL: it ensures we don't have strings like 'NAN' or 'TBD' mixed with datetimes
                df[col] = pd.to_datetime(df[col], errors='coerce')
                converted_count += 1
                logger.debug(f"Converted {col} → datetime")
            except Exception as e:
                logger.warning(f"Failed to convert {col} to datetime: {e}")
    
    logger.info(f"Converted {converted_count} date columns to datetime format")

    # ── 4️ Clean container numbers ─────────────
    if 'container_number' in df.columns:
        df['container_number'] = (
            df['container_number']
            .astype(str)
            .apply(lambda x: re.sub(r'[^A-Z0-9]', '', x.upper()) if pd.notnull(x) else x)
        )
        logger.debug("Cleaned container_number column")

    # ── 5️ Normalise "multiple" columns ───────
    multiple_cols = [c for c in df.columns if 'multiple' in c]
    for col in multiple_cols:
        if col in df.columns:
            df[col] = (
                df[col]
                .astype(str)
                .apply(lambda x: re.sub(r'[;|]', ',', x) if pd.notnull(x) else "")
                .str.replace(r'\s+,\s+', ', ', regex=True)
                .str.strip()
            )
    
    if multiple_cols:
        logger.info(f"Normalized {len(multiple_cols)} 'multiple' columns")

    # ── 6️ Derived fields ────────────────────

    # Delay days calculation
    # CRITICAL FIX: Use current date properly and handle NaT comparisons safely.
    today = pd.Timestamp.now().normalize() # Normalize to midnight for fair day comparison

    def delta_dp(eta_dp, ata_dp, derived_ata_dp):
        """
        Calculate delay (delta) at discharge port based on arrival status
        
        Returns:
        --------
        int : Delay in days (positive = late, negative = early, 0 = on time/not yet due)
        """
        
        # If eta_dp is missing/NaT, we cannot calculate delay, though it was not possible but for safety
        if pd.isna(eta_dp):
            return 0 
                
        # Case 1: eta_dp is in the future (not yet due)
        if eta_dp > today:
            return 0  # No delay calculation yet
        
        # Case 2: eta_dp is today or in the past
        # Sub-case 2A: Actual arrival exists ata_dp
        if pd.notna(ata_dp):
            return (ata_dp - eta_dp).days
        
        # Sub-case 2B: No actual arrival, but derived_ata_dp exists
        elif pd.notna(derived_ata_dp):
            # If derived_ata_dp is in the past, use it
            if derived_ata_dp < today:
                return (derived_ata_dp - eta_dp).days
            # If derived_ata_dp is in the future, calculate from today (as it's technically late as of now)
            else:
                return (today - eta_dp).days
        
        # Sub-case 2C: No arrival data at all, calculate from today
        else:
            return (today - eta_dp).days

    # Apply delay calculation only if required columns exist
    if 'eta_dp' in df.columns and ('ata_dp' in df.columns or 'derived_ata_dp' in df.columns):
        # Ensure columns exist, fill missing with NaT for safety in apply
        if 'ata_dp' not in df.columns: df['ata_dp'] = pd.NaT
        if 'derived_ata_dp' not in df.columns: df['derived_ata_dp'] = pd.NaT
        
        df['delay_days'] = df.apply(lambda row: delta_dp(row['eta_dp'], row['ata_dp'], row['derived_ata_dp']), axis=1)
        logger.info("Added `delay_days` column with FIXED logic")

    # Actual Transit time calculation
    if {'atd_lp', 'ata_dp'}.issubset(df.columns):
        # Vectorized calculation is faster and safer
        df['actual_transit_days'] = (df['ata_dp'] - df['atd_lp']).dt.days
        logger.info("Added `actual_transit_days` column")
    
    # Estimated transit time
    if {'etd_lp', 'eta_dp'}.issubset(df.columns):
        df['estimated_transit_days'] = (df['eta_dp'] - df['etd_lp']).dt.days
        logger.info("Added `estimated_transit_days` column")

    # Extract port codes from discharge_port
    def extract_last_code(text):
        """Extract port code from format like 'PORT NAME(CODE)'"""
        if pd.isna(text): return None
        matches = re.findall(r'\(([^()]*)\)', str(text))
        return matches[-1] if matches else None
    
    if 'discharge_port' in df.columns:
        df['port_code'] = df['discharge_port'].apply(extract_last_code)
        logger.debug("Extracted port_code from discharge_port")
    
    # Extract load port codes
    if 'load_port' in df.columns:
        df['load_port_code'] = df['load_port'].apply(extract_last_code)
        logger.debug("Extracted load_port_code from load_port")

    logger.info(f"Pre-processing finished - {len(df)} rows remain")
    return df
