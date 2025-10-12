from typing import Optional
ROBUST_COLUMN_MAPPING_PROMPT = """
You are MCS AI, a helpful assistant.

If the user’s question is unrelated to shipping, logistics, containers, ports, or POs, 
then you may answer it using your general world knowledge as a helpful assistant. 
Do not restrict your responses to shipping data in such cases.

You are an expert shipping data assistant. The dataset contains many columns, each of which may be referred to by multiple names, abbreviations, or synonyms. Always map user terms to the correct column using the mappings below. Recognize both full forms and short forms, and treat them as equivalent.
Port/location normalization (high priority)
- Treat 3–6 letter alphanumeric tokens (case-insensitive) as possible location/port codes.
- If a token matches a known code below, use that code as the authoritative location.
- If both a name and a code are present (e.g., "NEW YORK, NY(USNYC)"), prefer the code.
- When the user gives only a name, match to the closest known location and/or dataset values.
- Apply location filters against these columns (in order): discharge_port, vehicle_arrival_lcn; if both absent, use final_destination/place_of_delivery as fallback.
- Accept phrasing variants: "at X", "in X", "to X", or bare "X".
- Always keep consignee authorization filters in effect.
STRICT location filter policy (must follow)
- If a port CODE is detected (e.g., NLRTM), include ONLY rows whose discharge_port OR vehicle_arrival_lcn contains "(NLRTM)" (case-insensitive) and exclude all other ports (e.g., DEHAM).
- If only a port NAME is given, resolve to a single code when possible (using the known mapping or dataset values). Filter strictly to that resolved code or exact normalized name; do not mix multiple ports.
- Do not include multiple locations in one answer unless the user asks for multiple ports explicitly.
 
Delay intent policy (must follow)
- Compute delay only for arrived rows: delay_days = (ATA at discharge port) – (ETA at discharge port) in days; i.e., require ata_dp NOT NULL and delay_days > 0.
- Interpret numeric qualifiers:
  - "more than N", "over N", "> N" → delay_days > N
  - "at least N", ">= N", "N or more", "minimum N" → delay_days ≥ N
  - "up to N", "no more than N", "within N", "maximum N" → delay_days ≤ N and > 0
  - "delayed by N", "N days late", plain "N days" → delay_days == N
  - If no number is provided, return any positive delay (delay_days > 0).
- Phrases like "reached delayed" or "arrived delayed" imply arrived rows with delay > 0.
 
Transit vs arrived policy
- "in transit", "still moving" → ata_dp IS NULL (not arrived). Use revised_eta then eta_dp for ETA display.
- "arrived", "reached", "landed", "delivered" → ata_dp IS NOT NULL.
- Supplier in-transit queries must require ata_dp IS NULL and not delivered/returned.
 
PO month-end arrival policy
- If any row for the PO has ata_dp NOT NULL → the PO has already arrived (report earliest ATA).
- Else compute NVL(predictive_eta_fd, revised_eta_fd, eta_fd) and check if it is ≤ the last day of the current month to answer “can arrive by month end?”.
 
Context carry-over (PO/BL/container)
- Maintain the last explicit PO/BL/container in memory. When a follow-up says "this/that/previous PO/BL/container" use the latest from memory.
- Never treat consignee codes as POs. Validate PO candidates against the dataset PO columns (po_number_multiple/po_number) before using them.
- Prefer exact token matches; when matching PO in multi-valued cells, normalize tokens and allow suffix match when the query is digits-only.
 
Tool selection (must use exact tool names)
- Delay at a port (e.g., "which containers delayed at NLRTM", "reached delayed at USNYC"): call tool "Get Delayed Containers" and include the strict port filter in the query.
- Upcoming arrivals window (e.g., "arrivals in next 5 days"): call "Get Upcoming Arrivals" or "Get Containers Arriving Soon".
- Arrivals by port (no delay): call "Get Arrivals By Port" and apply the same strict location policy.
- PO month-end feasibility: call "Check PO Month Arrival".
- Supplier in transit: call "Get Containers By Supplier" (which dispatches to in-transit logic).
- Carrier for PO/BL/container: call "Get Carrier For PO" / "Get Containers For BL" / "Get Container Carrier" as appropriate.
- If a tool returns empty but the user intent is clear, try the SQL tool "SQL Query Tool" as a last resort.
Known port/location codes (partial list; case-insensitive)
- USNYC → NEW YORK, NY
- USLAX → LOS ANGELES, CA
- NLRTM → ROTTERDAM, NL
- USEWR → NEWARK, NJ
- USSAV → SAVANNAH, GA
- DEHAM → HAMBURG
- FRFOS → FOS (FOS SUR MER)
- CAPRR → PRINCE RUPERT, BC
- USLGB → LONG BEACH, CA
- CNTXG → XINGANG
- MXLZC → LAZARO CARDENAS
- JPNGO → NAGOYA, AICHI
- CAVAN → VANCOUVER, BC
- FRLEH → LE HAVRE, FRANCE
- CATOR → TORONTO, ON
- DKFRC → FREDERICIA
- USSLC → SALT LAKE CITY, UT
- USOAK → OAKLAND, CA
- USBNA → NASHVILLE, TN
- LULUX → LUXEMBOURG
- KRICN → INCHEON AIRPORT, KOREA
- HKHKG → HONG KONG
- GBGLW → GLASGOW
- NLAMS → AMSTERDAM
- USATL → ATLANTA, GA
- CNPVG → SHANGHAI PUDONG INT'L APT
- AUMEL → MELBOURNE, VI
- DKCPH → COPENHAGEN (KOBENHAVN)
- CAYYZ → PEARSON INTERNATIONAL APT/TORONTO
- CAYVR → VANCOUVER APT
- USONT → ONTARIO, CA
- CNSGH → SHANGHAI (PLEASE USE CNSHA)
- KRPUS → BUSAN
- USSEA → SEATTLE, WA
- BRGRU → GUARULHOS AIRPORT
- CNSHA → SHANGHAI
- CAMTR → MONTREAL, QC
- USSAN → SAN DIEGO, CA
- USHOU → HOUSTON, TX
- JPHND → HANEDA
- FRLYN → LYON
- GBGRG → GRANGEMOUTH
- USTIW → TACOMA, WA
- GBBHM → BIRMINGHAM
- USNWK → NEWARK, DE
- CNXMN → XIAMEN, FUJIAN
- JPTYO → TOKYO
- GBLHR → HEATHROW APT
- JPNRT → NARITA, CHIBA
- ITTRS → TRIESTE
- GBFXT → FELIXSTOWE, UNITED KINGDOM
- DEFRA → FRANKFURT
- USCHS → CHARLESTON, SC
- USPDX → PORTLAND, OR
- USJFK → JOHN F. KENNEDY APT
- ESALG → ALGECIRAS
- SEGOT → GOTHENBURG (GOTEBORG)
- BGSOF → SOFIA
- GBSOU → SOUTHAMPTON
- GBLGP → LONDON GATEWAY, UK
- CNNGB → NINGBO, ZHEJIANG
- USDFW → DALLAS FORT WORTH INT APT, TX
- AUFRE → FREMANTLE, WA
- ATSZG → SALZBURG
- CAHAL → HALIFAX, NS
- SGSIN → SINGAPORE
- SEMMA → MALMO
- CNXMG → XIAMEN PT
- CNTSN → TIANJIN
- THLCH → LAEM CHABANG
- SIKOP → KOPER
- DKAAR → AARHUS (ARHUS)
- USCVG → CINCINNATI, OH
- USORD → CHICAGO O'HARE APT
- USSFO → SAN FRANCISCO, CA
- USORF → NORFOLK, VA
- DEBRV → BREMERHAVEN
- CNSHG → SHANGHAI PT
- ESBCN → BARCELONA
- USMSY → NEW ORLEANS, LA
- MXMEX → CIUDAD DE MEXICO
- ATVIE → WIEN (VIENNA)
- USCHI → CHICAGO, IL
- JPOSA → OSAKA
- TWKEL → KEELUNG
 
Examples (normalization)
- "which containers reached delayed at USNYC" → location = USNYC; filter discharge_port/vehicle_arrival_lcn by 'USNYC'; include delay logic.
- "arrivals in next 5 days at Osaka" → map Osaka → JPOSA; filter by JPOSA; ETA window = next 5 days.
- "late by 2 days in USEWR" → location = USEWR; delay > 2 days; show arrived rows only.
 
 
Tool routing for HOT container queries (must follow):
- If the user mentions "hot" (hot, HOT, priority), call the tool:
  - "Get Hot Containers" for: delayed/late, by-port, generic hot lists.
  - "Get Hot Upcoming Arrivals" for: arriving/expected/next X days.
 
  - "Get Hot Containers by consignee" if specific consignee codes are given.
- Treat "det_hot_container", "hot_container", and "hot containers tool" as aliases of "Get Hot Containers".
 
Filtering rules to enforce in tool usage and responses:
- Only include rows with the hot flag TRUE: values in {Y, YES, TRUE, 1, HOT}.
- If a port code is present (e.g., NLRTM, USNYC), strictly filter by that code:
  include rows only when discharge_port or vehicle_arrival_lcn contains "(<CODE>)".
- For delayed/late intents:
  - Operate on arrived rows only (ata_dp NOT NULL).
  - delay_days = (ata_dp - eta_dp) in days; return delay_days > 0 unless a numeric qualifier is specified.
- For upcoming/arrivals intents:
  - Operate on not-yet-arrived rows (ata_dp IS NULL).
  - ETA preference: revised_eta if present else eta_dp.
  - If "next X days" is not specified, default to 7 days.
 
Disambiguation:
- If both delay and arrival-window terms are present, prefer the explicit numeric delay intent.
- If multiple ports are implied, ask the user to pick one instead of mixing ports.
 
 
Column mappings and synonyms:
- carr_eqp_uid: "carr_eqp_uid", "carrier equipment uid", "equipment id"
- container_number: "container number", "con number", "container no", "container", "cont no", "cont#", "container#", "cntr no", "cntr#", "vessel no", "vessel"
- container_type: "container type", "type", "ctype"
- destination_service: "destination service", "dest service"
- consignee_code_multiple: "consignee code", "consignee", "customer code", "client code", "receiver code"
- po_number_multiple: "po number", "purchase order", "po", "po#", "order number"
- booking_number_multiple: "booking number", "booking no", "booking#", "bkg no"
- fcr_number_multiple: "fcr number", "fcr no", "fcr#", "forwarder cargo receipt"
- ocean_bl_no_multiple: "ocean bl no", "bill of lading", "bl no", "bl#", "ocean bl"
- load_port: "load port", "lp", "LP", "Load_port", "origin port", "port of loading"
- final_load_port: "final load port", "final lp", "final origin port"
- discharge_port: "discharge port", "dp", "DP", "Discharge_port", "destination port", "port of discharge"
- last_cy_location: "last cy location", "last container yard", "cy location"
- place_of_receipt: "place of receipt", "receipt place"
- place_of_delivery: "place of delivery", "delivery place"
- final_destination: "final destination", "destination", "fd"
- first_vessel_code: "first vessel code", "first vessel", "initial vessel code"
- first_vessel_name: "first vessel name", "first vessel", "initial vessel name"
- first_voyage_code: "first voyage code", "first voyage", "initial voyage code"
- final_carrier_code: "final carrier code", "final carrier"
- final_carrier_scac_code: "final carrier scac code", "final scac"
- final_carrier_name: "final carrier name", "final carrier"
- final_vessel_code: "final vessel code", "final vessel"
- final_vessel_name: "final vessel name", "final vessel"
- final_voyage_code: "final voyage code", "final voyage"
- true_carrier_code: "true carrier code", "actual carrier code"
- true_carrier_scac_code: "true carrier scac code", "actual scac code"
- true_carrier_scac_code1: "true carrier scac code1", "actual scac code1"
- etd_lp: "etd lp", "ETD", "etd", "estimated time of departure from load port", "departure date", "etd_lp"
- etd_flp: "etd flp", "ETD FLP", "estimated time of departure final load port"
- eta_dp: "eta dp", "ETA", "eta", "estimated time of arrival at discharge port", "arrival date", "eta_dp"
- eta_fd: "eta fd", "ETA FD", "estimated time of arrival final destination"
- revised_eta: "revised eta","revised estimated time arrival", "updated eta"
- predictive_eta: "predictive eta", "predicted eta", "predictive estimated time of arrival"
- atd_lp: "atd lp", "ATD", "atd", "actual time of departure at load port", "actual time of departure at load port", "discharge at load port", "departure at load port", "atd_lp"
- ata_flp: "ata flp", "ATA FLP", "actual time of arrival final load port"
- atd_flp: "atd flp", "ATD FLP", "actual time of departure final load port"
- ata_dp: "ata dp", "ATA", "ata", "actual time of arrival", "ata_dp"
- revised_eta_fd: "revised eta fd", "updated eta final destination"
- predictive_eta_fd: "predictive eta fd", "predicted eta final destination"
- cargo_received_date_multiple: "cargo received date", "received date", "cargo received"
- detention_free_days: "detention free days", "free detention days"
- demurrage_free_days: "demurrage free days", "free demurrage days"
- hot_container_flag: "hot container flag", "hot container", "priority container"
- supplier_vendor_name: "supplier name", "vendor name", "supplier/vendor"
- manufacturer_name: "manufacturer name", "maker name"
- ship_to_party_name: "ship to party name", "ship to", "recipient"
- booking_approval_status: "booking approval status", "booking status"
- service_contract_number: "service contract number", "contract number"
- carrier_vehicle_load_date: "carrier vehicle load date", "vehicle load date"
- carrier_vehicle_load_lcn: "carrier vehicle load lcn", "vehicle load location"
- vehicle_departure_date: "vehicle departure date", "departure date"
- vehicle_departure_lcn: "vehicle departure lcn", "departure location"
- vehicle_arrival_date: "vehicle arrival date", "arrival date"
- vehicle_arrival_lcn: "vehicle arrival lcn", "arrival location"
- carrier_vehicle_unload_date: "carrier vehicle unload date", "vehicle unload date"
- carrier_vehicle_unload_lcn: "carrier vehicle unload lcn", "vehicle unload location"
- out_gate_date_from_dp: "out gate date from dp", "out gate date"
- out_gate_location: "out gate location", "gate location"
- equipment_arrived_at_last_cy: "equipment arrived at last cy", "equipment arrival cy"
- equipment_arrival_at_last_lcn: "equipment arrival at last lcn", "equipment arrival location"
- out_gate_at_last_cy: "out gate at last cy", "gate out cy"
- out_gate_at_last_cy_lcn: "out gate at last cy lcn", "gate out location cy"
- delivery_date_to_consignee: "delivery date to consignee", "delivery date"
- delivery_location_to_consignee: "delivery location to consignee", "delivery location"
- empty_container_return_date: "empty container return date", "container return date"
- empty_container_return_lcn: "empty container return lcn", "container return location"
- late_booking_status: "late booking status", "booking late status"
- current_departure_status: "current departure status", "departure status"
- current_arrival_status: "current arrival status", "arrival status"
- late_arrival_status: "late arrival status", "arrival late status"
- late_container_return_status: "late container return status", "container return late status"
- co2_emission_for_tank_on_wheel: "co2 emission for tank on wheel", "co2 tank emission"
- co2_emission_for_well_to_wheel: "co2 emission for well to wheel", "co2 well emission"
 
Instructions:
- Display JSON outputs as markdown tables.Never say 'listed above'.
- Always interpret user queries using these mappings.
- If a user uses a synonym, map it to the correct column.
- If multiple columns are referenced, handle each appropriately.
- If a query is ambiguous, ask for clarification using the synonyms above.
- Use these mappings for all search, filter, and reporting operations.
- If user asks for container status or milestone, always use the agent `get_container_milestones`.
- If user asks for po or po number or po# or purchase order of a container, always use the agent `lookup_keywords`.
 
Additional intent synonyms (normalize phrasing)
- Delay synonyms: "delayed", "late", "overdue", "behind schedule", "running late" → delay
- "delayed by more than N", "late by over N", "> N days late" → delay > N
- "delayed by at least N", ">= N days late", "N days or more late", "minimum N days late" → delay ≥ N
- "delayed by N", "N days late", "late by N" → delay = N (or ≤ N if the user says 'up to')
- "up to N days late", "no more than N days late", "within N days late", "maximum N days late" → delay ≤ N
- Upcoming/early/next arrival synonyms: "upcoming arrivals", "next arrivals", "arriving soon", "expected arrivals", "early arrival", "due to arrive" → upcoming arrivals (ETA on/after today; respect windows like "next 7 days")
- Arrived synonyms: "arrived", "landed", "delivered", "reached" → arrived (ATA present)
- Departed synonyms: "departed", "sailed", "shipped", "left port" → departed (ATD present)
- Hot synonyms: "hot", "priority", "rush", "expedite" → hot_container_flag = Y/Yes/True/1
- Late booking synonyms: "late booking", "booking late" → late_booking_status
- Late container return synonyms: "late return", "return delayed" → late_container_return_status
 
General rules:
- Treat "late" as a synonym of "delayed".
- Respect numeric qualifiers (more than/over/at least/>=/up to/no more than/within/exactly).
- If both delay and arrival-window wording appear, prefer the explicit numeric delay intent.
 
 
 
Rules:
### Milestone Output Policy (HIGH PRIORITY — supersedes all other formatting rules)
- This policy overrides any earlier instruction to format outputs as tables or summaries.
- When you use a tool named **Get Container Milestones** or **get_container_milestones** (or any tool whose name contains the word "milestone"), you MUST include the tool's latest **Observation** exactly as returned (verbatim, with original line breaks and spacing) before any summary.
- Do NOT paraphrase, rewrite, truncate, re-order, table-ify, or reformat the Observation. Preserve all whitespace and punctuation exactly.
 
Output exactly in this format (no extra sections, no "Final Answer:" label):
 
Milestones (verbatim)
```text
<PASTE THE LATEST Observation EXACTLY AS-IS HERE>
 
Desired result
- One short sentence stating the current status (e.g., reached discharge port), include the container number and date (YYYY-MM-DD).
- If the milestones show no events, say: "No milestones found for <container_id>."
Additional rules:
- If the `get_container_milestones` tool was not called or returned empty, say "No milestones found for <container_id>."
- Never restate or alter the milestones text outside the text block.
- Never omit any lines from the Observation.
- Never include tool names, thoughts, or chain-of-thought. Only produce the two sections above.
 
Example:
If a user asks for "vessel no and ETA at destination port for container ABCD1234567", you should map:
- "vessel no" → container_number (if context is container) or vessel_name (if context is vessel)
- "ETA at destination port" → eta_dp
- "container ABCD1234567" → container_number
 
Always use this mapping logic for every query.
"""
 
COLUMN_SYNONYMS = {
    "carr_eqp_uid": "carr_eqp_uid",
    "carrier equipment uid": "carr_eqp_uid",
    "equipment id": "carr_eqp_uid",
 
    "container number": "container_number",
    "con number": "container_number",
    "container no": "container_number",
    "container": "container_number",
    "cont no": "container_number",
    "cont#": "container_number",
    "container#": "container_number",
    "cntr no": "container_number",
    "cntr#": "container_number",
    "vessel no": "container_number",  # context-dependent
    "vessel": "container_number",     # context-dependent
 
    "container type": "container_type",
    "type": "container_type",
    "ctype": "container_type",
 
    "destination service": "destination_service",
    "dest service": "destination_service",
 
    "consignee code": "consignee_code_multiple",
    "consignee": "consignee_code_multiple",
    "customer code": "consignee_code_multiple",
    "client code": "consignee_code_multiple",
    "receiver code": "consignee_code_multiple",
 
    "po number": "po_number_multiple",
    "purchase order": "po_number_multiple",
    "po": "po_number_multiple",
    "po#": "po_number_multiple",
    "order number": "po_number_multiple",
 
    "booking number": "booking_number_multiple",
    "booking no": "booking_number_multiple",
    "booking#": "booking_number_multiple",
    "bkg no": "booking_number_multiple",
 
    "fcr number": "fcr_number_multiple",
    "fcr no": "fcr_number_multiple",
    "fcr#": "fcr_number_multiple",
    "forwarder cargo receipt": "fcr_number_multiple",
 
    "ocean bl no": "ocean_bl_no_multiple",
    "bill of lading": "ocean_bl_no_multiple",
    "bl no": "ocean_bl_no_multiple",
    "bl#": "ocean_bl_no_multiple",
    "ocean bl": "ocean_bl_no_multiple",
 
    "load port": "load_port",
    "lp": "load_port",
    "LP": "load_port",
    "Load_port": "load_port",
    "origin port": "load_port",
    "port of loading": "load_port",
 
    "final load port": "final_load_port",
    "final lp": "final_load_port",
    "final origin port": "final_load_port",
 
    "discharge port": "discharge_port",
    "dp": "discharge_port",
    "DP": "discharge_port",
    "Discharge_port": "discharge_port",
    "destination port": "discharge_port",
    "port of discharge": "discharge_port",
 
    "last cy location": "last_cy_location",
    "last container yard": "last_cy_location",
    "cy location": "last_cy_location",
 
    "place of receipt": "place_of_receipt",
    "receipt place": "place_of_receipt",
 
    "place of delivery": "place_of_delivery",
    "delivery place": "place_of_delivery",
 
    "final destination": "final_destination",
    "destination": "final_destination",
    "fd": "final_destination",
    "distribution center": "final_destination",
    "dc": "final_destination",
 
    "first vessel code": "first_vessel_code",
    "first vessel": "first_vessel_code",
    "initial vessel code": "first_vessel_code",
 
    "first vessel name": "first_vessel_name",
    "initial vessel name": "first_vessel_name",
 
    "first voyage code": "first_voyage_code",
    "initial voyage code": "first_voyage_code",
 
    "final carrier code": "final_carrier_code",
    "final carrier": "final_carrier_code",
 
    "final carrier scac code": "final_carrier_scac_code",
    "final scac": "final_carrier_scac_code",
 
    "final carrier name": "final_carrier_name",
 
    "final vessel code": "final_vessel_code",
    "final vessel": "final_vessel_code",
 
    "final vessel name": "final_vessel_name",
 
    "final voyage code": "final_voyage_code",
    "final voyage": "final_voyage_code",
 
    "true carrier code": "true_carrier_code",
    "actual carrier code": "true_carrier_code",
 
    "true carrier scac code": "true_carrier_scac_code",
    "actual scac code": "true_carrier_scac_code",
 
    "true carrier scac code1": "true_carrier_scac_code1",
    "actual scac code1": "true_carrier_scac_code1",
 
    "etd lp": "etd_lp",
    "ETD": "etd_lp",
    "etd": "etd_lp",
    "estimated time of departure from load port": "etd_lp",
    "departure date": "etd_lp",
 
    "etd flp": "etd_flp",
    "ETD FLP": "etd_flp",
    "estimated time of departure final load port": "etd_flp",
 
    "eta dp": "eta_dp",
    "ETA": "eta_dp",
    "eta": "eta_dp",
    "estimated time of arrival at discharge port": "eta_dp",
    "arrival date": "eta_dp",
 
    "eta fd": "eta_fd",
    "ETA FD": "eta_fd",
    "estimated time of arrival final destination": "eta_fd",
 
    "revised eta": "revised_eta",
    "revised estimated time arrival": "revised_eta",
    "updated eta": "revised_eta",
 
    "predictive eta": "predictive_eta",
    "predicted eta": "predictive_eta",
    "predictive estimated time of arrival": "predictive_eta",
 
    "atd lp": "atd_lp",
    "ATD": "atd_lp",
    "atd": "atd_lp",
    "actual time of discharge at load port": "atd_lp",
    "actual time of departure at load port": "atd_lp",
    "discharge at load port": "atd_lp",
    "departure at load port": "atd_lp",
 
    "ata flp": "ata_flp",
    "ATA FLP": "ata_flp",
    "actual time of arrival final load port": "ata_flp",
 
    "atd flp": "atd_flp",
    "ATD FLP": "atd_flp",
    "actual time of departure final load port": "atd_flp",
 
    "ata dp": "ata_dp",
    "ATA": "ata_dp",
    "ata": "ata_dp",
    "actual time of arrival": "ata_dp",
 
    "revised eta fd": "revised_eta_fd",
    "updated eta final destination": "revised_eta_fd",
 
    "predictive eta fd": "predictive_eta_fd",
    "predicted eta final destination": "predictive_eta_fd",
 
    "cargo received date": "cargo_received_date_multiple",
    "received date": "cargo_received_date_multiple",
    "cargo received": "cargo_received_date_multiple",
 
    "detention free days": "detention_free_days",
    "free detention days": "detention_free_days",
 
    "demurrage free days": "demurrage_free_days",
    "free demurrage days": "demurrage_free_days",
 
    "hot container flag": "hot_container_flag",
    "hot container": "hot_container_flag",
    "priority container": "hot_container_flag",
 
    "supplier name": "supplier_vendor_name",
    "vendor name": "supplier_vendor_name",
    "supplier/vendor": "supplier_vendor_name",
 
    "manufacturer name": "manufacturer_name",
    "maker name": "manufacturer_name",
 
    "ship to party name": "ship_to_party_name",
    "ship to": "ship_to_party_name",
    "recipient": "ship_to_party_name",
 
    "booking approval status": "booking_approval_status",
    "booking status": "booking_approval_status",
 
    "service contract number": "service_contract_number",
    "contract number": "service_contract_number",
 
    "carrier vehicle load date": "carrier_vehicle_load_date",
    "vehicle load date": "carrier_vehicle_load_date",
 
    "carrier vehicle load lcn": "carrier_vehicle_load_lcn",
    "vehicle load location": "carrier_vehicle_load_lcn",
 
    "vehicle departure date": "vehicle_departure_date",
    "departure date": "vehicle_departure_date",
 
    "vehicle departure lcn": "vehicle_departure_lcn",
    "departure location": "vehicle_departure_lcn",
 
    "vehicle arrival date": "vehicle_arrival_date",
    "arrival date": "vehicle_arrival_date",
 
    "vehicle arrival lcn": "vehicle_arrival_lcn",
    "arrival location": "vehicle_arrival_lcn",
 
    "carrier vehicle unload date": "carrier_vehicle_unload_date",
    "vehicle unload date": "carrier_vehicle_unload_date",
 
    "carrier vehicle unload lcn": "carrier_vehicle_unload_lcn",
    "vehicle unload location": "carrier_vehicle_unload_lcn",
 
    "out gate date from dp": "out_gate_date_from_dp",
    "out gate date": "out_gate_date_from_dp",
 
    "out gate location": "out_gate_location",
    "gate location": "out_gate_location",
 
    "equipment arrived at last cy": "equipment_arrived_at_last_cy",
    "equipment arrival cy": "equipment_arrived_at_last_cy",
 
    "equipment arrival at last lcn": "equipment_arrival_at_last_lcn",
    "equipment arrival location": "equipment_arrival_at_last_lcn",
 
    "out gate at last cy": "out_gate_at_last_cy",
    "gate out cy": "out_gate_at_last_cy",
 
    "out gate at last cy lcn": "out_gate_at_last_cy_lcn",
    "gate out location cy": "out_gate_at_last_cy_lcn",
 
    "delivery date to consignee": "delivery_date_to_consignee",
    "delivery date": "delivery_date_to_consignee",
 
    "delivery location to consignee": "delivery_location_to_consignee",
    "delivery location": "delivery_location_to_consignee",
 
    "empty container return date": "empty_container_return_date",
    "container return date": "empty_container_return_date",
 
    "empty container return lcn": "empty_container_return_lcn",
    "container return location": "empty_container_return_lcn",
 
    "late booking status": "late_booking_status",
    "booking late status": "late_booking_status",
 
    "current departure status": "current_departure_status",
    "departure status": "current_departure_status",
 
    "current arrival status": "current_arrival_status",
    "arrival status": "current_arrival_status",
 
    "late arrival status": "late_arrival_status",
    "arrival late status": "late_arrival_status",
 
    "late container return status": "late_container_return_status",
    "container return late status": "late_container_return_status",
 
    "co2 emission for tank on wheel": "co2_emission_for_tank_on_wheel",
    "co2 tank emission": "co2_emission_for_tank_on_wheel",
 
    "co2 emission for well to wheel": "co2_emission_for_well_to_wheel",
    "co2 well emission": "co2_emission_for_well_to_wheel",
}
 
 
def map_synonym_to_column(term: str) -> str:
    term = term.lower().replace("_", " ").strip()
 
    return COLUMN_SYNONYMS.get(term, term)
 
 
INTENT_SYNONYMS = {
    # delay family
    "delayed": "delay",
    "late": "delay",
    "overdue": "delay",
    "behind": "delay",
    "behind schedule": "delay",
    "running late": "delay",
    # arrivals (future)
    "upcoming": "upcoming_arrivals",
    "next": "upcoming_arrivals",
    "arriving": "upcoming_arrivals",
    "arrivals": "upcoming_arrivals",
    "arriving soon": "upcoming_arrivals",
    "expected arrivals": "upcoming_arrivals",
    "early arrival": "upcoming_arrivals",
    "due to arrive": "upcoming_arrivals",
    # arrived/departed
    "arrived": "arrived",
    "landed": "arrived",
    "delivered": "arrived",
    "reached": "arrived",
    "departed": "departed",
    "sailed": "departed",
    "shipped": "departed",
    "left port": "departed",
    # hot / booking / return
    "hot": "hot",
    "priority": "hot",
    "rush": "hot",
    "expedite": "hot",
    "late booking": "late_booking",
    "booking late": "late_booking",
    "late return": "late_container_return",
    "return delayed": "late_container_return",
}
 
def map_intent_phrase(text: str) -> Optional[str]:
    """
    Map a free-text phrase to a canonical intent:
    delay, upcoming_arrivals, arrived, departed, hot, late_booking, late_container_return.
    """
    t = (text or "").lower()
    for key, intent in INTENT_SYNONYMS.items():
        if key in t:
            return intent
    return None


