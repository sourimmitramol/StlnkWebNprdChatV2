# services/preprocess.py
import logging
import re
import pandas as pd

logger = logging.getLogger("shipping_chatbot")


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Your original `preprocess_data` body – unchanged, only minor
    formatting to fit inside this module.
    """
    logger.info(f"Starting preprocessing on {len(df)} rows")
    original_row_count = len(df)

    # ── 1️⃣ Drop duplicates ─────────────────────
    df = df.drop_duplicates()
    logger.info(f"Removed {original_row_count - len(df)} duplicate rows")

    # ── 2️⃣ Normalise column names ───────────────
    df.columns = [c.strip().lower() for c in df.columns]

    # ── 3️⃣ Convert known date columns ───────────
    date_columns = [
        'etd_lp', 'etd_flp', 'eta_dp', 'eta_fd', 'revised_eta',
        'predictive_eta', 'atd_lp', 'ata_flp', 'atd_flp', 'ata_dp',
        'revised_eta_fd', 'predictive_eta_fd', 'cargo_received_date_multiple',
        'carrier_vehicle_load_date', 'vehicle_departure_date',
        'vehicle_arrival_date', 'carrier_vehicle_unload_date',
        'out_gate_date_from_dp', 'equipment_arrived_at_last_cy',
        'out_gate_at_last_cy', 'delivery_date_to_consignee',
        'empty_container_return_date'
    ]
    for col in date_columns:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')
            logger.debug(f"Converted {col} → datetime")

    # ── 4️⃣ Clean container numbers ─────────────
    if 'container_number' in df.columns:
        df['container_number'] = (
            df['container_number']
            .astype(str)
            .apply(lambda x: re.sub(r'[^A-Z0-9]', '', x.upper()) if pd.notnull(x) else x)
        )

    # ── 5️⃣ Normalise “multiple” columns ───────
    for col in [c for c in df.columns if 'multiple' in c]:
        if col in df.columns:
            df[col] = (
                df[col]
                .astype(str)
                .apply(lambda x: re.sub(r'[;|]', ',', x) if pd.notnull(x) else "")
                .str.replace(r'\s+,\s+', ', ', regex=True)
                .str.strip()
            )

    # ── 6️⃣ Derived field – delay_days ────────
    if {'eta_dp', 'ata_dp'}.issubset(df.columns):
        df['delay_days'] = (df['ata_dp'] - df['eta_dp']).dt.days
        logger.info("Added `delay_days` column")

    def extract_last_code(text):
        matches = re.findall(r'\(([^()]*)\)', str(text))
        return matches[-1] if matches else None
    df['port_code'] = df['discharge_port'].apply(extract_last_code)

    logger.info(f"Pre‑processing finished – {len(df)} rows remain")
    return df
