# # services/preprocess.py
# import logging
# import re
# import pandas as pd

# logger = logging.getLogger("shipping_chatbot")


# def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
#     """
#     Your original `preprocess_data` body – unchanged, only minor
#     formatting to fit inside this module.
#     """
#     logger.info(f"Starting preprocessing on {len(df)} rows")
#     original_row_count = len(df)

#     # ── 1️⃣ Drop duplicates ─────────────────────
#     df = df.drop_duplicates()
#     logger.info(f"Removed {original_row_count - len(df)} duplicate rows")

#     # ── 2️⃣ Normalise column names ───────────────
#     df.columns = [c.strip().lower() for c in df.columns]

#     # ── 3️⃣ Convert known date columns ───────────
#     date_columns = [
#         'etd_lp', 'etd_flp', 'eta_dp', 'eta_fd', 'revised_eta',
#         'predictive_eta', 'atd_lp', 'ata_flp', 'atd_flp', 'ata_dp',
#         'revised_eta_fd', 'predictive_eta_fd', 'cargo_received_date_multiple',
#         'carrier_vehicle_load_date', 'vehicle_departure_date',
#         'vehicle_arrival_date', 'carrier_vehicle_unload_date',
#         'out_gate_date_from_dp', 'equipment_arrived_at_last_cy',
#         'out_gate_at_last_cy', 'delivery_date_to_consignee',
#         'empty_container_return_date'
#     ]
#     for col in date_columns:
#         if col in df.columns:
#             df[col] = pd.to_datetime(df[col], errors='coerce')
#             logger.debug(f"Converted {col} → datetime")

#     # ── 4️⃣ Clean container numbers ─────────────
#     if 'container_number' in df.columns:
#         df['container_number'] = (
#             df['container_number']
#             .astype(str)
#             .apply(lambda x: re.sub(r'[^A-Z0-9]', '', x.upper()) if pd.notnull(x) else x)
#         )

#     # ── 5️⃣ Normalise “multiple” columns ───────
#     for col in [c for c in df.columns if 'multiple' in c]:
#         if col in df.columns:
#             df[col] = (
#                 df[col]
#                 .astype(str)
#                 .apply(lambda x: re.sub(r'[;|]', ',', x) if pd.notnull(x) else "")
#                 .str.replace(r'\s+,\s+', ', ', regex=True)
#                 .str.strip()
#             )

#     # ── 6️⃣ Derived field – delay_days ────────
#     if {'eta_dp', 'ata_dp'}.issubset(df.columns):
#         df['delay_days'] = (df['ata_dp'] - df['eta_dp']).dt.days
#         logger.info("Added `delay_days` column")

#     def extract_last_code(text):
#         matches = re.findall(r'\(([^()]*)\)', str(text))
#         return matches[-1] if matches else None
#     df['port_code'] = df['discharge_port'].apply(extract_last_code)

#     logger.info(f"Pre‑processing finished – {len(df)} rows remain")
#     return df
# services/preprocess.py

# Comprehensive data preprocessing for shipment data

import logging
import re
import pandas as pd
import warnings

logger = logging.getLogger("shipping_chatbot")


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocesses shipment data with comprehensive date parsing,
    duplicate removal, and data normalization.
    """
    logger.info(f"Starting preprocessing on {len(df)} rows")
    original_row_count = len(df)

    # ── 1️⃣ Drop duplicates ─────────────────────
    df = df.drop_duplicates()
    logger.info(f"Removed {original_row_count - len(df)} duplicate rows")

    # ── 2️⃣ Normalise column names ───────────────
    df.columns = [c.strip().lower() for c in df.columns]

    # ── 3️⃣ Convert ALL known date columns ───────────
    # Comprehensive list of date-related columns in shipment data
    date_columns = [
        # Load Port dates
        'etd_lp',                           # Estimated Time of Departure - Load Port
        'atd_lp',                           # Actual Time of Departure - Load Port
        'cargo_received_date_multiple',     # Cargo received at load port
        
        # First Load Port dates
        'etd_flp',                          # Estimated Time of Departure - First Load Port
        'ata_flp',                          # Actual Time of Arrival - First Load Port
        'atd_flp',                          # Actual Time of Departure - First Load Port
        
        # Discharge Port dates
        'eta_dp',                           # Estimated Time of Arrival - Discharge Port
        'ata_dp',                           # Actual Time of Arrival - Discharge Port
        'derived_ata_dp',                   # Derived Actual Time of Arrival - Discharge Port
        'revised_eta',                      # Revised ETA - Discharge Port
        'predictive_eta',                   # Predictive ETA - Discharge Port
        
        # Final Destination dates
        'eta_fd',                           # Estimated Time of Arrival - Final Destination
        'ata_fd',                           # Actual Time of Arrival - Final Destination
        'revised_eta_fd',                   # Revised ETA - Final Destination
        'predictive_eta_fd',                # Predictive ETA - Final Destination
        
        # Container/Vehicle Movement dates
        'carrier_vehicle_load_date',        # When container loaded onto vehicle
        'vehicle_departure_date',           # When vehicle departed
        'vehicle_arrival_date',             # When vehicle arrived
        'carrier_vehicle_unload_date',      # When container unloaded from vehicle
        
        # Container Yard (CY) dates
        'equipment_arrived_at_last_cy',     # Container arrival at last CY
        'out_gate_at_last_cy',              # Out gate from last CY
        'out_gate_date_from_dp',            # Out gate from discharge port
        
        # Delivery dates
        'delivery_date_to_consignee',       # Final delivery to consignee
        'empty_container_return_date',      # Empty container return
        
        # Additional common date fields (add if they exist in your data)
        'rail_arrival_destination_date',                     
        'rail_departure_dp_date',                  
        'rail_load_dp_date',                  
                            # Payment date
    ]
    
    # Convert only columns that exist in the dataframe
    converted_count = 0
    for col in date_columns:
        if col in df.columns:
            try:
                # Pandas can emit noisy warnings when parsing heterogeneous date formats.
                # We keep parsing behavior identical (errors='coerce') and only silence
                # the known non-fatal warnings produced by pd.to_datetime.
                with warnings.catch_warnings():
                    warnings.filterwarnings(
                        "ignore",
                        message=r"Could not infer format, so each element will be parsed individually.*",
                        category=UserWarning,
                    )
                    warnings.filterwarnings(
                        "ignore",
                        message=r"In a future version of pandas, parsing datetimes with mixed time zones.*",
                        category=FutureWarning,
                    )
                    df[col] = pd.to_datetime(df[col], errors='coerce')
                converted_count += 1
                logger.debug(f"Converted {col} → datetime")
            except Exception as e:
                logger.warning(f"Failed to convert {col} to datetime: {e}")
    
    logger.info(f"Converted {converted_count} date columns to datetime format")

    # ── 4️⃣ Clean container numbers ─────────────
    if 'container_number' in df.columns:
        df['container_number'] = (
            df['container_number']
            .astype(str)
            .apply(lambda x: re.sub(r'[^A-Z0-9]', '', x.upper()) if pd.notnull(x) else x)
        )
        logger.debug("Cleaned container_number column")

    # ── 5️⃣ Normalise "multiple" columns ───────
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

    # ── 6️⃣ Derived fields ────────────────────

    # Delay days calculation
    if {'eta_dp', 'ata_dp'}.issubset(df.columns):
        df['delay_days'] = (df['ata_dp'] - df['eta_dp']).dt.days
        logger.info("Added `delay_days` column")

    # Actual Transit time calculation
    if {'atd_lp', 'ata_dp'}.issubset(df.columns):
        df['actual_transit_days'] = (df['ata_dp'] - df['atd_lp']).dt.days
        logger.info("Added `actual_transit_days` column")
    
    # Estimated transit time
    if {'etd_lp', 'eta_dp'}.issubset(df.columns):
        df['estimated_transit_days'] = (df['eta_dp'] - df['etd_lp']).dt.days
        logger.info("Added `estimated_transit_days` column")

    # Extract port codes from discharge_port
    def extract_last_code(text):
        """Extract port code from format like 'PORT NAME(CODE)'"""
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
