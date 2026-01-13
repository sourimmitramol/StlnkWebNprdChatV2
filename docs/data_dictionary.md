# Shipment Data Dictionary & Business Rules

This document serves as the "Knowledge Base" for the AI Agent's Data Analyst Engine. It defines the schema, column aliases, and the business logic applied to the shipment dataset.

## 1. Primary Identifiers
| Column Name | Description | Examples |
|-------------|-------------|----------|
| `container_number` | Unique ID for a shipping container. | `MSBU4522691`, `TCLU4703170` |
| `po_number_multiple` | Comma-separated list of Purchase Orders in the shipment. | `5302997239`, `5302943326` |
| `ocean_bl_no_multiple` | Bill of Lading identifiers. | `MOLWMNL2400017` |
| `booking_number_multiple` | Booking references associated with the shipment. | `GT3000512`, `VN2084805` |

## 2. Temporal Fields (Dates)
All dates are normalized to `YYYY-MM-DD`.

### Load Port (Origin)
- `etd_lp`: Estimated Time of Departure from Load Port.
- `atd_lp`: Actual Time of Departure from Load Port.
- `load_port`: Name and code of the origin port (e.g., `SHANGHAI(CNSHA)`).

### Discharge Port (Arrival)
- `eta_dp`: Original Estimated Time of Arrival at Discharge Port.
- `revised_eta`: Updated ETA if the original schedule changed.
- `ata_dp`: Actual Time of Arrival at Discharge Port.
- `derived_ata_dp`: System-calculated arrival date when ATA is missing.
- `discharge_port`: Name and code of the destination port.

### Final Destination & Delivery
- `eta_fd`: Estimated Arrival at Final Destination (Warehouse/DC).
- `delivery_date_to_consignee`: Date the cargo was delivered to the customer.
- `empty_container_return_date`: Date the empty container was returned to the yard.

## 3. Derived Analytics Fields
These fields are calculated during preprocessing and are preferred for analytics queries.

| Field Name | Definition / Logic |
|------------|-------------------|
| `delay_days` | Number of days late. **Logic:** `ata_dp - eta_dp`. If `ata_dp` is null, uses `derived_ata_dp` or `today - eta_dp`. Positive means late; 0 or negative means on-time/early. |
| `actual_transit_days` | Total days spent at sea. **Logic:** `ata_dp - atd_lp`. |
| `estimated_transit_days`| Planned days at sea. **Logic:** `eta_dp - etd_lp`. |
| `port_code` | Extracted code from `discharge_port` (e.g., `USLGB`). |
| `load_port_code` | Extracted code from `load_port` (e.g., `CNSHA`). |

## 4. Key Business Logic Rules
1. **Shipment Status**:
   - **Arrived**: `ata_dp` is NOT null OR `derived_ata_dp` <= Today.
   - **In Transit**: `atd_lp` is NOT null AND `ata_dp` is null.
   - **Not Yet Departed**: `atd_lp` is null.
2. **Hot/Priority**:
   - Containers with `hot_container_flag == TRUE` are high priority.
3. **Carrier Selection**:
   - Use `final_carrier_name` for the entity responsible for the final leg.
4. **Consignee Filtering**:
   - Users are filtered by `consignee_code_multiple`. Data access must always verify the user's authorized code exists in this field.

## 5. Metadata for SQL/Pandas Transformer
- When filtering for a month without a year, assume the year is **2025**.
- Use `.str.contains(code, na=False)` for "multiple" columns (like `po_number_multiple`).
