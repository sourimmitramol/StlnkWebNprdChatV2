from typing import Optional

ROBUST_COLUMN_MAPPING_PROMPT = """
You are MCS AI, a helpful assistant.

If the user’s question is unrelated to shipping, logistics, containers, ports, or POs, 
then you may answer it using your general world knowledge as a helpful assistant. 
Do not restrict your responses to shipping data in such cases.
When you use a tool such as "Handle Non-shipping queries", include that tool’s full output 
as your final assistant answer instead of summarizing it.
Do not write summaries like "the user now has a detailed guide".

**RESPONSE FORMATTING RULES (Must Follow)**
When presenting information to users:
1. **Be Concise**: Provide direct, focused answers without unnecessary explanations or filler words
2. **Use Point-wise Format**: Structure responses using bullet points (•) or numbered lists when presenting multiple items or details
3. **Organize Information Logically**:
   - Start with the most important/requested information first
   - Group related details together
   - Use clear section headers for complex responses
4. **Format Guidelines**:
   - For single records: Present key details in bulleted format
   - For multiple records: Use tables or numbered lists with clear identifiers
   - For status/milestones: Present chronologically with clear labels
   - For dates: Use consistent format (YYYY-MM-DD) and label clearly
   - Highlight critical information (delays, urgent items, exceptions)
5. **Avoid**:
   - Long paragraphs and run-on text
   - Redundant or verbose explanations
   - Unnecessary introductory phrases like "Based on the data..." or "Here are the results..."
   - Repetition of the user's question

**Example Response Format**:
✓ GOOD:
**Container MSKU4343533 Status:**
• Current Status: Delivered at LAEM CHABANG (2025-12-15)
• PO Number: 5302982894
• Milestones:
  - Departed: SHANGHAI (2025-10-30)
  - Arrived: LAEM CHABANG (2025-11-10)
  - Delivered: LAEM CHABANG (2025-12-15)
• Empty Return: 2025-11-13

✗ POOR:
"Based on the data retrieved from the system, I found that the container MSKU4343533, which is associated with PO 5302982894, has the following status. The container departed from SHANGHAI on 2025-10-30 and then it reached LAEM CHABANG on 2025-11-10. After that, it was delivered on 2025-12-15 and the empty container was returned on 2025-11-13."

**CRITICAL: Date Handling Policy (Must Follow)**
When calling tools with date-related queries:
- ALWAYS pass the user's ORIGINAL date phrase EXACTLY AS-IS to the tool, preserving verb tense AND year information if provided
- If user says "Oct 2025" → pass "Oct 2025" to tool (DO NOT strip year, DO NOT make separate call without year)
- If user says "Oct" without year → pass "Oct" to tool (DO NOT add year)
- NEVER calculate or convert relative dates yourself (e.g., "yesterday", "day before yesterday", "N days ago")
- NEVER convert "this week" to "next 7 days" or "this month" to "next 30 days" - these have different meanings
- The tools have built-in date parsing that correctly handles current date context AND VERB TENSE
- ✓ CORRECT: User says "Oct 2025" → action_input = "containers delayed in Oct 2025" (preserves year)
- ✓ CORRECT: User says "Oct" → action_input = "containers delayed in Oct" (no year added)
- ✓ CORRECT: action_input = "containers going to ENFIELD in Oct" (preserves future tense "going to")
- ✓ CORRECT: action_input = "containers departed from QINGDAO day before yesterday"
- ✓ CORRECT: action_input = "containers scheduled for this week" (preserves "this week" = current calendar week)
- ✓ CORRECT: action_input = "delayed containers this month" (preserves "this month" = current calendar month)
- ✗ WRONG: User says "Oct 2025" but you call tool with "Oct" first, then "Oct 2025" (NEVER make two calls)
- ✗ WRONG: User says "Oct 2025" → action_input = "containers delayed in Oct" (DO NOT strip user-specified year)
- ✗ WRONG: User says "Oct" → action_input = "containers delayed in Oct 2025" (NEVER add year user didn't provide)
- ✗ WRONG: action_input = "containers scheduled to depart in next 7 days" when user said "this week"
- If user says "going to in Oct" (future tense), the tool interprets it as October 2026
- If user says "reached in Oct" (past tense), the tool interprets it as October 2025
- If user says "yesterday" and today is 2026-01-22, the tool will correctly interpret it as 2026-01-21
- If user says "this week" on 2026-01-22, the tool interprets it as Jan 19-25, 2026 (Monday-Sunday)
- If user says "this month" on 2026-01-22, the tool interprets it as Jan 1-31, 2026 (full month)

You are an expert shipping data assistant. The dataset contains many columns, each of which may be referred to by multiple names, abbreviations, or synonyms. Always map user terms to the correct column using the mappings below. Recognize both full forms and short forms, and treat them as equivalent.
Port/location normalization (high priority)
- Treat 3–6 letter alphanumeric tokens (case-insensitive) as possible location/port codes.
- If a token matches a known code below, use that code as the authoritative location.
- If both a name and a code are present (e.g., "NEW YORK, NY(USNYC)"), prefer the code.
- When the user gives only a name, match to the closest known location and/or dataset values.
- Apply location filters against these columns (in order): discharge_port; if both absent, use final_destination/place_of_delivery as fallback.
- Accept phrasing variants: "at X", "in X", "to X", or bare "X".
- Always keep consignee authorization filters in effect.
STRICT location filter policy (must follow)
- If a port CODE is detected (e.g., NLRTM), include ONLY rows whose discharge_port contains "(NLRTM)" (case-insensitive) and exclude all other ports (e.g., DEHAM).
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
- **Hot containers with delay thresholds** (e.g., "hot containers delayed by 7 days", "hot containers delayed more than 5 days"): call tool "Get Delayed Containers" (NOT "Get Hot Containers") - it has built-in hot filtering.
- **Delay with day thresholds** (e.g., "containers delayed by N days", "delayed more than/less than N days"): call tool "Get Delayed Containers".
- Delay at a port (e.g., "which containers delayed at NLRTM", "reached delayed at USNYC"): call tool "Get Delayed Containers" and include the strict port filter in the query.
- **Containers at DP but not at FD** (e.g., "arrived at discharge port but not delivered", "reached DP but not at final destination", "containers at port waiting for delivery"): call tool "Get Containers At DP Not FD".
- **Hot containers without delay thresholds** (e.g., "show hot containers", "list hot containers"): call tool "Get Hot Containers".
- Upcoming arrivals window (e.g., "arrivals in next 5 days"): call "Get Upcoming Arrivals" or "Get Containers Arriving Soon".
- Arrivals by port (no delay): call "Get Arrivals By Port" and apply the same strict location policy.
- PO month-end feasibility: call "Check PO Month Arrival".
- Supplier in transit: call "Get Containers By Supplier" (which dispatches to in-transit logic).
- Carrier for PO/BL/container: call "Get Carrier For PO" / "Get Containers For BL" / "Get Container Carrier" as appropriate.
- **Consignee information** (e.g., "what is the consignee of PO#X", "who is the consignee for container Y", "show consignee for OBL Z"): call tool "Get Consignee Info" with the full question including the identifier (PO/container/OBL). DO NOT route to "Get Container Milestones" or "Get Field Info" for consignee queries.
- Shipped quantity / cargo quantity (e.g., "shipped quantity for PO#X", "cargo quantity for container Y", "how many units shipped"): call tool "Get Shipped Quantity" with the full question including the identifier (PO/container/OBL/booking number).
- **Job number queries** (e.g., "what is the job number for container X", "job number associated with PO Y"): There is NO specific tool for job numbers. DO NOT attempt to answer job number queries using tools or by guessing. Simply state "I don't have explicit information about the job number" to trigger pandas fallback which can retrieve the job_no field from the dataset. NEVER respond with booking number, BL number, or PO number when asked for job number - they are different fields.
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
- Only include rows with the hot flag TRUE: values in {{{{Y, YES, TRUE, 1, HOT}}}}.
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
- job_no: "job number", "job no", "job#", "internal job number" - IMPORTANT: This is a UNIQUE internal job identifier and is NOT the same as booking number, BL number, or PO number. It is a separate field (job_no column) that identifies the internal job record for the shipment. When asked for job number, ALWAYS return the job_no field value, NEVER return booking_number_multiple or ocean_bl_no_multiple.
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

Final answer format for milestones (inside the JSON `action_input` string of the **Final Answer**):
- You MUST still follow the LangChain structured chat **FORMAT_INSTRUCTIONS** and respond with a JSON object.
- For the **Final Answer** step, set `"action": "Final Answer"` and put the ENTIRE user-visible text below into the `"action_input"` string value.
- Do **not** remove or change the JSON wrapper required by the format instructions; only control the text of `action_input`.

The `action_input` string for milestone queries MUST look like this (and only this, apart from the actual pasted content and summary text):

Milestones (verbatim)
```text
<PASTE THE LATEST Observation EXACTLY AS-IS HERE>
```

Desired result
- One short sentence stating the current status (e.g., reached discharge port), including the container number and date (YYYY-MM-DD).
- If the milestones show no events, say: "No milestones found for <container_id>."

Additional rules:
- If the `get_container_milestones` tool was not called or returned empty, say "No milestones found for <container_id>."
- Never restate or alter the milestones text outside the text block.
- Never omit any lines from the Observation.
- Never include tool names, thoughts, or chain-of-thought. Only produce the two sections above inside `action_input`.

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
    "vessel": "container_number",  # context-dependent
    "container type": "container_type",
    "type": "container_type",
    "ctype": "container_type",
    "seal number": "seal_number",
    "seal no": "seal_number",
    "seal": "seal_number",
    "seal#": "seal_number",
    "container seal": "seal_number",
    "seal_no": "seal_number",
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
    "job number": "job_no",
    "job no": "job_no",
    "job_no": "job_no",
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
    # ==========================================
    # DELAY FAMILY - All variations of "delayed/late"
    # ==========================================
    "delay": "delay",
    "delayed": "delay",
    "late": "delay",
    "overdue": "delay",
    "behind": "delay",
    "behind schedule": "delay",
    "running late": "delay",
    "past due": "delay",
    "missed deadline": "delay",
    "not on time": "delay",
    "tardy": "delay",
    "slow": "delay",
    "lagging": "delay",
    "backlog": "delay",
    "held up": "delay",
    "stuck": "delay",
    "waiting": "delay",
    "pending": "delay",
    "not arrived": "delay",
    "hasn't arrived": "delay",
    "still waiting": "delay",
    "taking too long": "delay",
    # ==========================================
    # UPCOMING/FUTURE ARRIVALS - Containers not yet arrived
    # ==========================================
    "upcoming": "upcoming_arrivals",
    "next": "upcoming_arrivals",
    "arriving": "upcoming_arrivals",
    "arrivals": "upcoming_arrivals",
    "arriving soon": "upcoming_arrivals",
    "expected arrivals": "upcoming_arrivals",
    "expected": "upcoming_arrivals",
    "due to arrive": "upcoming_arrivals",
    "scheduled": "upcoming_arrivals",
    "scheduled to arrive": "upcoming_arrivals",
    "coming": "upcoming_arrivals",
    "on the way": "upcoming_arrivals",
    "in transit": "upcoming_arrivals",
    "incoming": "upcoming_arrivals",
    "future": "upcoming_arrivals",
    "forecasted": "upcoming_arrivals",
    "anticipated": "upcoming_arrivals",
    "will arrive": "upcoming_arrivals",
    "going to arrive": "upcoming_arrivals",
    "eta": "upcoming_arrivals",
    "estimated arrival": "upcoming_arrivals",
    "early arrival": "upcoming_arrivals",
    "advance notice": "upcoming_arrivals",
    # ==========================================
    # ARRIVED - Containers that have reached destination
    # ==========================================
    "arrived": "arrived",
    "reached": "arrived",
    "landed": "arrived",
    "delivered": "arrived",
    "received": "arrived",
    "got in": "arrived",
    "came in": "arrived",
    "made it": "arrived",
    "at port": "arrived",
    "at destination": "arrived",
    "discharged": "arrived",
    "unloaded": "arrived",
    "completed": "arrived",
    "finished": "arrived",
    "ata": "arrived",
    "actual arrival": "arrived",
    "has arrived": "arrived",
    "already arrived": "arrived",
    # ==========================================
    # DEPARTED - Containers that have left origin
    # ==========================================
    "departed": "departed",
    "sailed": "departed",
    "shipped": "departed",
    "left": "departed",
    "left port": "departed",
    "sailed from": "departed",
    "departed from": "departed",
    "set off": "departed",
    "moved out": "departed",
    "loaded": "departed",
    "picked up": "departed",
    "atd": "departed",
    "actual departure": "departed",
    "has departed": "departed",
    "already departed": "departed",
    "on board": "departed",
    "vessel departed": "departed",
    # ==========================================
    # HOT CONTAINERS - Priority/urgent shipments
    # ==========================================
    "hot": "hot",
    "Hot": "hot",
    "HOT": "hot",
    "priority": "hot",
    "urgent": "hot",
    "rush": "hot",
    "expedite": "hot",
    "expedited": "hot",
    "critical": "hot",
    "high priority": "hot",
    "express": "hot",
    "fast track": "hot",
    "time sensitive": "hot",
    "asap": "hot",
    "emergency": "hot",
    "immediate": "hot",
    "vip": "hot",
    "top priority": "hot",
    # ==========================================
    # LATE BOOKING - Booking delays
    # ==========================================
    "late booking": "late_booking",
    "booking late": "late_booking",
    "booking delayed": "late_booking",
    "delayed booking": "late_booking",
    "booking not confirmed": "late_booking",
    "booking pending": "late_booking",
    "booking overdue": "late_booking",
    # ==========================================
    # LATE CONTAINER RETURN - Return delays
    # ==========================================
    "late return": "late_container_return",
    "return delayed": "late_container_return",
    "delayed return": "late_container_return",
    "return overdue": "late_container_return",
    "not returned": "late_container_return",
    "hasn't been returned": "late_container_return",
    "return pending": "late_container_return",
    "return late": "late_container_return",
}


def map_intent_phrase(text: str) -> Optional[str]:
    """
    Map a free-text phrase to a canonical intent using longest-match-first strategy.

    Supported intents:
    - delay: delayed, late, overdue, behind schedule, etc.
    - upcoming_arrivals: arriving, next, upcoming, expected, etc.
    - arrived: reached, landed, delivered, discharged, etc.
    - departed: sailed, shipped, left port, etc.
    - hot: priority, urgent, rush, expedited, etc.
    - late_booking: booking delays
    - late_container_return: container return delays

    Args:
        text: User query string

    Returns:
        Canonical intent string or None if no match

    Examples:
        >>> map_intent_phrase("Show me delayed containers")
        'delay'

        >>> map_intent_phrase("Which containers are arriving next week?")
        'upcoming_arrivals'

        >>> map_intent_phrase("Hot containers at USNYC")
        'hot'
    """
    if not text:
        return None

    text_lower = text.lower()

    # Sort keys by length (longest first) to match multi-word phrases before single words
    # E.g., "behind schedule" should match before "behind"
    sorted_keys = sorted(INTENT_SYNONYMS.keys(), key=len, reverse=True)

    for key in sorted_keys:
        # Use word boundary matching for better accuracy
        # This prevents "notified" from matching "not" in "not arrived"
        if len(key.split()) == 1:  # Single word
            pattern = r"\b" + re.escape(key.lower()) + r"\b"
        else:  # Multi-word phrase
            pattern = re.escape(key.lower())

        if re.search(pattern, text_lower):
            return INTENT_SYNONYMS[key]

    return None


# Optional: Reverse mapping for debugging/logging
INTENT_TO_SYNONYMS = {}
for synonym, intent in INTENT_SYNONYMS.items():
    if intent not in INTENT_TO_SYNONYMS:
        INTENT_TO_SYNONYMS[intent] = []
    INTENT_TO_SYNONYMS[intent].append(synonym)

# Example output:
# {
#     'delay': ['delay', 'delayed', 'late', 'overdue', ...],
#     'upcoming_arrivals': ['upcoming', 'next', 'arriving', ...],
#     ...
# }

import re
from datetime import datetime
from typing import Optional, Tuple

import pandas as pd

# ...existing code...


def parse_time_period(query: str) -> tuple[pd.Timestamp, pd.Timestamp, str]:
    """
    Parse natural language time expressions into (start_date, end_date, description).

    **CRITICAL FIX FOR YEAR DETECTION WITH VERB TENSE (January 22, 2026)**:
    - **VERB TENSE TAKES PRIORITY** over month-based heuristics:
      * Future tense ("going to", "will", "shall", "arriving") → use current year (2026)
      * Past tense ("reached", "arrived", "departed", "came") → use previous year (2025)
      * No clear tense → use month-based logic below

    - When only month names are provided without year:
      * Future tense: "going to ENFIELD in Oct" → October 2026 (not 2025)
      * Past tense: "reached ENFIELD in Oct" → October 2025
      * Neutral: "Oct" without verb → October 2025 (month hasn't occurred in 2026 yet)

    **Examples (assuming current date is January 22, 2026)**:
    - "going to ENFIELD in Oct" → October 1-31, 2026 (future tense overrides)
    - "containers arriving in Oct" → October 1-31, 2026 (future tense)
    - "reached in Oct" → October 1-31, 2025 (past tense)
    - "departed in Oct" → October 1-31, 2025 (past tense)
    - "Oct" (no verb) → October 1-31, 2025 (default: month hasn't occurred this year)
    - "December 2025" → December 1-31, 2025 (explicit year preserved)
    - "next 7 days" → today to today+6
    - "yesterday" → January 21, 2026

    Supported patterns:
    - Month ranges: "Jun-Sep", "June to September", "from June to September"
    - Months with year: "December 2025", "in dec 2025"
    - Single months: "December", "in dec", "in Oct"
    - Relative: "today", "tomorrow", "yesterday", "next week", "last month"
    - Ranges: "next 7 days", "last 30 days", "in next 5 days"
    - Explicit: "from 2025-01-15 to 2025-01-20", "between dates"

    Returns: (start_date, end_date, period_description)
    """
    import re

    import pandas as pd

    query = (query or "").strip().lower()
    today = pd.Timestamp.today().normalize()
    current_year = today.year  # 2026
    current_month = today.month  # 1 (January)

    # ========================================================================
    # ABSOLUTE HIGHEST PRIORITY: Explicit date patterns (on YYYY-MM-DD)
    # Check these FIRST to avoid agent-inserted incorrect dates
    # ========================================================================

    # Pattern 0: "on YYYY-MM-DD" or "at YYYY-MM-DD" (single date)
    single_date_match = re.search(
        r"\b(?:on|at)\s+(\d{4}[-/]\d{1,2}[-/]\d{1,2})\b", query
    )
    if single_date_match:
        target_date = pd.to_datetime(
            single_date_match.group(1), errors="coerce"
        ).normalize()
        if pd.notna(target_date):
            try:
                import logging

                logger = logging.getLogger(__name__)
                logger.info(
                    f"[parse_time_period] Matched 'on/at {single_date_match.group(1)}': {target_date.strftime('%Y-%b-%d')}"
                )
            except:
                pass
            # return target_date, target_date, f"on {target_date.strftime('%Y-%m-%d')}"
            return target_date, target_date, f"on {target_date.strftime('%Y-%b-%d')}"

    # ========================================================================
    # HIGHEST PRIORITY: Relative date patterns (today, yesterday, etc.)
    # MUST be checked FIRST before any month/year logic to avoid wrong year
    # ========================================================================

    # Pattern 1: "day before yesterday" or "2 days ago"
    if re.search(r"\b(?:day\s+before\s+yesterday|2\s+days?\s+ago)\b", query):
        target_date = today - pd.Timedelta(days=2)
        try:
            import logging

            logger = logging.getLogger(__name__)
            logger.info(
                f"[parse_time_period] Matched 'day before yesterday': {target_date.strftime('%Y-%m-%d')}"
            )
        except:
            pass
        return target_date, target_date, "day before yesterday"

    # Pattern 2: "N days ago" (general pattern for 3+, 4+, etc.)
    days_ago_match = re.search(r"\b(\d+)\s+days?\s+ago\b", query)
    if days_ago_match:
        n = int(days_ago_match.group(1))
        target_date = today - pd.Timedelta(days=n)
        try:
            import logging

            logger = logging.getLogger(__name__)
            logger.info(
                f"[parse_time_period] Matched '{n} days ago': {target_date.strftime('%Y-%m-%d')}"
            )
        except:
            pass
        return target_date, target_date, f"{n} days ago"

    # Pattern 3: "day after tomorrow"
    if re.search(r"\b(?:day\s+after\s+tomorrow)\b", query):
        target_date = today + pd.Timedelta(days=2)
        return target_date, target_date, "day after tomorrow"

    # Pattern 4: "today"
    if re.search(r"\btoday\b", query):
        return today, today, "today"

    # Pattern 5: "tomorrow"
    if re.search(r"\btomorrow\b", query):
        tomorrow = today + pd.Timedelta(days=1)
        return tomorrow, tomorrow, "tomorrow"

    # Pattern 6: "yesterday"
    if re.search(r"\byesterday\b", query):
        yesterday = today - pd.Timedelta(days=1)
        try:
            import logging

            logger = logging.getLogger(__name__)
            logger.info(
                f"[parse_time_period] Matched 'yesterday': {yesterday.strftime('%Y-%m-%d')}"
            )
        except:
            pass
        return yesterday, yesterday, "yesterday"

    # Month name to number mapping
    month_map = {
        "jan": 1,
        "january": 1,
        "feb": 2,
        "february": 2,
        "mar": 3,
        "march": 3,
        "apr": 4,
        "april": 4,
        "may": 5,
        "jun": 6,
        "june": 6,
        "jul": 7,
        "july": 7,
        "aug": 8,
        "august": 8,
        "sep": 9,
        "sept": 9,
        "september": 9,
        "oct": 10,
        "october": 10,
        "nov": 11,
        "november": 11,
        "dec": 12,
        "december": 12,
    }

    # ========================================================================
    # VERB TENSE DETECTION: Determine if query refers to future or past
    # ========================================================================
    # Future indicators: going to, will, shall, arriving, scheduled, expected, upcoming, next
    future_patterns = r"\b(going\s+to|will|shall|arriving|scheduled|expected|upcoming|are\s+going|is\s+going)\b"
    is_future_tense = bool(re.search(future_patterns, query, re.IGNORECASE))

    # Past indicators: reached, arrived, departed, came, have/has + past participle, was, were
    past_patterns = (
        r"\b(reached|arrived|departed|came|have\s+\w+ed|has\s+\w+ed|was|were|had)\b"
    )
    is_past_tense = bool(re.search(past_patterns, query, re.IGNORECASE))

    try:
        import logging

        logger = logging.getLogger(__name__)
        logger.info(
            f"[parse_time_period] Verb tense detection - Future: {is_future_tense}, Past: {is_past_tense}"
        )
    except:
        pass

    # **PRIORITY 1**: Month ranges without year (e.g., "Jun-Sep", "June to September")
    month_range_patterns = [
        r"\b(jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\s*[-–—]\s*(jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\b",
        r"\b(jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\s+to\s+(jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\b",
        r"\bfrom\s+(jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\s+to\s+(jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\b",
    ]

    for pattern in month_range_patterns:
        m = re.search(pattern, query, re.IGNORECASE)
        if m:
            start_month_str = m.group(1).lower()
            end_month_str = m.group(2).lower()

            # Normalize to shortest key
            start_month_key = (
                start_month_str[:3]
                if start_month_str[:3] in month_map
                else start_month_str
            )
            end_month_key = (
                end_month_str[:3] if end_month_str[:3] in month_map else end_month_str
            )

            start_month_num = month_map.get(start_month_key)
            end_month_num = month_map.get(end_month_key)

            if start_month_num and end_month_num:
                # **CRITICAL YEAR DETECTION LOGIC WITH VERB TENSE**
                # Current: January 2026 (current_month=1, current_year=2026)

                # VERB TENSE OVERRIDES MONTH-BASED LOGIC:
                # - Future tense ("going to", "will") → use current year or next year
                # - Past tense ("reached", "arrived") → use previous year
                # - No clear tense → use month-based heuristic

                if is_future_tense and not is_past_tense:
                    # Future tense: always use current year or future
                    # Example: In Jan 2026, "going to in Oct" means October 2026
                    year = current_year
                    try:
                        import logging

                        logger = logging.getLogger(__name__)
                        logger.info(
                            f"[parse_time_period] Month range {start_month_str}-{end_month_str}: "
                            f"FUTURE TENSE detected, using current year {year}"
                        )
                    except:
                        pass

                elif is_past_tense and not is_future_tense:
                    # Past tense: use previous year if month >= current_month
                    # Example: In Jan 2026, "reached in Oct" means October 2025
                    if start_month_num >= current_month:
                        year = current_year - 1
                    else:
                        year = current_year - 1  # Past months in past year
                    try:
                        import logging

                        logger = logging.getLogger(__name__)
                        logger.info(
                            f"[parse_time_period] Month range {start_month_str}-{end_month_str}: "
                            f"PAST TENSE detected, using previous year {year}"
                        )
                    except:
                        pass

                # No clear verb tense - use month-based heuristic (original logic)
                elif start_month_num > current_month:
                    # Month hasn't occurred this year yet
                    year = current_year - 1  # Assume last year
                    try:
                        import logging

                        logger = logging.getLogger(__name__)
                        logger.info(
                            f"[parse_time_period] Month range {start_month_str}-{end_month_str}: "
                            f"start_month ({start_month_num}) > current_month ({current_month}), "
                            f"using previous year {year}"
                        )
                    except:
                        pass

                elif start_month_num == current_month:
                    # Same month as current → use current year
                    year = current_year
                    try:
                        import logging

                        logger = logging.getLogger(__name__)
                        logger.info(
                            f"[parse_time_period] Month range {start_month_str}-{end_month_str}: "
                            f"start_month ({start_month_num}) == current_month, using current year {year}"
                        )
                    except:
                        pass

                else:  # start_month_num < current_month
                    # Past month in current year → use LAST YEAR for historical data
                    year = current_year - 1
                    try:
                        import logging

                        logger = logging.getLogger(__name__)
                        logger.info(
                            f"[parse_time_period] Month range {start_month_str}-{end_month_str}: "
                            f"start_month ({start_month_num}) < current_month ({current_month}), "
                            f"using previous year {year}"
                        )
                    except:
                        pass

                # Create start and end dates
                start_date = pd.Timestamp(
                    year=year, month=start_month_num, day=1
                ).normalize()

                # Handle end month (could be in next year if Dec->Jan range)
                if end_month_num < start_month_num:
                    # Range crosses year boundary (e.g., "Nov-Feb")
                    end_year = year + 1
                else:
                    end_year = year

                # Last day of end month
                if end_month_num == 12:
                    end_date = pd.Timestamp(
                        year=end_year + 1, month=1, day=1
                    ).normalize() - pd.Timedelta(days=1)
                else:
                    end_date = pd.Timestamp(
                        year=end_year, month=end_month_num + 1, day=1
                    ).normalize() - pd.Timedelta(days=1)

                # Format period description
                start_month_name = start_month_str.capitalize()
                end_month_name = end_month_str.capitalize()
                period_desc = (
                    f"{start_month_name} to {end_month_name} {year}"
                    if year == end_year
                    else f"{start_month_name} {year} to {end_month_name} {end_year}"
                )

                try:
                    import logging

                    logger = logging.getLogger(__name__)
                    logger.info(
                        f"[parse_time_period] Final period: {period_desc} "
                        f"({start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')})"
                    )
                except:
                    pass

                return start_date, end_date, period_desc

    # **PRIORITY 2**: Month with year (e.g., "December 2025", "in dec 2025")
    m_month_year = re.search(
        r"\b(?:in\s+)?(jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|"
        r"jul(?:y)?|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\s+(\d{4})\b",
        query,
    )
    if m_month_year:
        month_name = m_month_year.group(1)
        year = int(m_month_year.group(2))
        month = month_map.get(month_name)

        if month:
            start_date = pd.Timestamp(year=year, month=month, day=1).normalize()
            # Get last day of the month
            if month == 12:
                end_date = pd.Timestamp(year=year, month=12, day=31).normalize()
            else:
                end_date = (
                    pd.Timestamp(year=year, month=month + 1, day=1)
                    - pd.Timedelta(days=1)
                ).normalize()

            return start_date, end_date, f"{month_name.capitalize()} {year}"

    # **PRIORITY 3**: Single month without year (e.g., "December", "in dec")
    m_month = re.search(
        r"\b(?:in\s+)?(jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|"
        r"jul(?:y)?|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\b",
        query,
    )
    if m_month:
        month_name = m_month.group(1)
        month = month_map.get(month_name)

        if month:
            # **SMART YEAR DETECTION WITH VERB TENSE**
            # Future tense → use current year (2026)
            # Past tense → use previous year (2025) if month >= current_month
            # No clear tense → month-based heuristic

            if is_future_tense and not is_past_tense:
                # Future tense: use current year
                # Example: In Jan 2026, "going to in Oct" means October 2026
                year = current_year
                try:
                    import logging

                    logger = logging.getLogger(__name__)
                    logger.info(
                        f"[parse_time_period] Single month '{month_name}': "
                        f"FUTURE TENSE detected, using current year {year}"
                    )
                except:
                    pass

            elif is_past_tense and not is_future_tense:
                # Past tense: use previous year
                # Example: In Jan 2026, "reached in Oct" means October 2025
                if month >= current_month:
                    year = current_year - 1
                else:
                    year = current_year - 1
                try:
                    import logging

                    logger = logging.getLogger(__name__)
                    logger.info(
                        f"[parse_time_period] Single month '{month_name}': "
                        f"PAST TENSE detected, using previous year {year}"
                    )
                except:
                    pass

            # No clear verb tense - use month-based heuristic
            elif month > current_month:
                year = current_year - 1
                try:
                    import logging

                    logger = logging.getLogger(__name__)
                    logger.info(
                        f"[parse_time_period] Single month '{month_name}': "
                        f"month ({month}) > current_month ({current_month}), using previous year {year}"
                    )
                except:
                    pass
            else:
                year = current_year
                try:
                    import logging

                    logger = logging.getLogger(__name__)
                    logger.info(
                        f"[parse_time_period] Single month '{month_name}': "
                        f"month ({month}) <= current_month ({current_month}), using current year {year}"
                    )
                except:
                    pass

            start_date = pd.Timestamp(year=year, month=month, day=1).normalize()
            # Get last day of the month
            if month == 12:
                end_date = pd.Timestamp(year=year, month=12, day=31).normalize()
            else:
                end_date = (
                    pd.Timestamp(year=year, month=month + 1, day=1)
                    - pd.Timedelta(days=1)
                ).normalize()

            return start_date, end_date, f"{month_name.capitalize()} {year}"

    # Pattern 7: "next X days" or "in next X days" or "coming X days"
    m = re.search(
        r"(?:next|in\s+next|in|coming|in\s+coming)\s+(\d{1,3})\s+days?", query
    )
    if m:
        n = int(m.group(1))
        end_date = today + pd.Timedelta(days=n - 1)
        return today, end_date, f"next {n} days"

    # Pattern 8: "last X days" or "past X days" or "in last X days" (with or without space before "days")
    m = re.search(
        r"(?:in\s+)?(?:last|past|previous|the\s+last|the\s+past)\s+(\d{1,3})\s*days?",
        query,
    )
    if m:
        n = int(m.group(1))
        start_date = today - pd.Timedelta(days=n - 1)
        return start_date, today, f"last {n} days"

    # Pattern 9: "this week" (Monday to Sunday)
    if re.search(r"\bthis\s+week\b", query):
        start_of_week = today - pd.Timedelta(days=today.weekday())
        end_of_week = start_of_week + pd.Timedelta(days=6)
        return start_of_week, end_of_week, "this week"

    # Pattern 10: "next week"
    if re.search(r"\bnext\s+week\b", query):
        start_of_next_week = today + pd.Timedelta(days=7 - today.weekday())
        end_of_next_week = start_of_next_week + pd.Timedelta(days=6)
        return start_of_next_week, end_of_next_week, "next week"

    # Pattern 11: "last week"
    if re.search(r"\blast\s+week\b", query):
        start_of_last_week = today - pd.Timedelta(days=today.weekday() + 7)
        end_of_last_week = start_of_last_week + pd.Timedelta(days=6)
        return start_of_last_week, end_of_last_week, "last week"

    # Pattern 12: "this month"
    if re.search(r"\bthis\s+month\b", query):
        start_of_month = pd.Timestamp(
            year=today.year, month=today.month, day=1
        ).normalize()
        if today.month == 12:
            end_of_month = pd.Timestamp(year=today.year, month=12, day=31).normalize()
        else:
            end_of_month = (
                pd.Timestamp(year=today.year, month=today.month + 1, day=1)
                - pd.Timedelta(days=1)
            ).normalize()
        return start_of_month, end_of_month, "this month"

    # Pattern 13: "next month"
    if re.search(r"\bnext\s+month\b", query):
        if today.month == 12:
            start_of_next_month = pd.Timestamp(
                year=today.year + 1, month=1, day=1
            ).normalize()
            end_of_next_month = pd.Timestamp(
                year=today.year + 1, month=1, day=31
            ).normalize()
        else:
            start_of_next_month = pd.Timestamp(
                year=today.year, month=today.month + 1, day=1
            ).normalize()
            if today.month == 11:
                end_of_next_month = pd.Timestamp(
                    year=today.year, month=12, day=31
                ).normalize()
            else:
                end_of_next_month = (
                    pd.Timestamp(year=today.year, month=today.month + 2, day=1)
                    - pd.Timedelta(days=1)
                ).normalize()
        return start_of_next_month, end_of_next_month, "next month"

    # Pattern 14: "last month"
    if re.search(r"\blast\s+month\b", query):
        if today.month == 1:
            start_of_last_month = pd.Timestamp(
                year=today.year - 1, month=12, day=1
            ).normalize()
            end_of_last_month = pd.Timestamp(
                year=today.year - 1, month=12, day=31
            ).normalize()
        else:
            start_of_last_month = pd.Timestamp(
                year=today.year, month=today.month - 1, day=1
            ).normalize()
            end_of_last_month = (
                pd.Timestamp(year=today.year, month=today.month, day=1)
                - pd.Timedelta(days=1)
            ).normalize()
        return start_of_last_month, end_of_last_month, "last month"

    # Pattern 15: Explicit date ranges "from YYYY-MM-DD to YYYY-MM-DD"
    m = re.search(
        r"from\s+(\d{4}[-/]\d{1,2}[-/]\d{1,2})\s+to\s+(\d{4}[-/]\d{1,2}[-/]\d{1,2})",
        query,
    )
    if m:
        start_date = pd.to_datetime(m.group(1), errors="coerce").normalize()
        end_date = pd.to_datetime(m.group(2), errors="coerce").normalize()
        return (
            start_date,
            end_date,
            f"from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}",
        )

    # Pattern 16: "between DATE1 and DATE2"
    m = re.search(
        r"between\s+(\d{4}[-/]\d{1,2}[-/]\d{1,2})\s+and\s+(\d{4}[-/]\d{1,2}[-/]\d{1,2})",
        query,
    )
    if m:
        start_date = pd.to_datetime(m.group(1), errors="coerce").normalize()
        end_date = pd.to_datetime(m.group(2), errors="coerce").normalize()
        return (
            start_date,
            end_date,
            f"between {start_date.strftime('%Y-%m-%d')} and {end_date.strftime('%Y-%m-%d')}",
        )

    # Default: next 7 days
    end_date = today + pd.Timedelta(days=6)
    return today, end_date, "next 7 days (default)"


# def parse_time_period(query: str) -> tuple[pd.Timestamp, pd.Timestamp, str]:
#     """
#     Parse natural language time expressions into (start_date, end_date, description).

#     **CRITICAL FIX**: Properly handles month names (both with and without year):
#     - "December" → Full current/next December (Dec 1 to Dec 31)
#     - "December 2025" → Full December 2025 (Dec 1 to Dec 31)
#     - "in dec" → Full December month

#     Supported patterns:
#     - Relative: "today", "tomorrow", "yesterday", "next week", "last month"
#     - Ranges: "next 7 days", "last 30 days", "in next 5 days"
#     - Months: "December", "dec", "December 2025", "in December"
#     - Explicit: "from 2025-01-15 to 2025-01-20", "between dates"

#     Returns: (start_date, end_date, period_description)
#     """

#     query = (query or "").strip().lower()
#     today = pd.Timestamp.today().normalize()

#     # Month name to number mapping
#     month_map = {
#         'jan': 1, 'january': 1,
#         'feb': 2, 'february': 2,
#         'mar': 3, 'march': 3,
#         'apr': 4, 'april': 4,
#         'may': 5,
#         'jun': 6, 'june': 6,
#         'jul': 7, 'july': 7,
#         'aug': 8, 'august': 8,
#         'sep': 9, 'sept': 9, 'september': 9,
#         'oct': 10, 'october': 10,
#         'nov': 11, 'november': 11,
#         'dec': 12, 'december': 12
#     }

#     # **CRITICAL FIX**: Pattern 1 - Month with year (e.g., "December 2025", "in dec 2025")
#     m_month_year = re.search(
#         r'\b(?:in\s+)?(jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|'
#         r'jul(?:y)?|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\s+(\d{4})\b',
#         query
#     )
#     if m_month_year:
#         month_name = m_month_year.group(1)
#         year = int(m_month_year.group(2))
#         month = month_map.get(month_name)

#         if month:
#             start_date = pd.Timestamp(year=year, month=month, day=1).normalize()
#             # Get last day of the month
#             if month == 12:
#                 end_date = pd.Timestamp(year=year, month=12, day=31).normalize()
#             else:
#                 end_date = (pd.Timestamp(year=year, month=month+1, day=1) - pd.Timedelta(days=1)).normalize()

#             return start_date, end_date, f"{month_name.capitalize()} {year}"

#     # **CRITICAL FIX**: Pattern 2 - Month without year (e.g., "December", "in dec")
#     m_month = re.search(
#         r'\b(?:in\s+)?(jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|'
#         r'jul(?:y)?|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\b',
#         query
#     )
#     if m_month:
#         month_name = m_month.group(1)
#         month = month_map.get(month_name)

#         if month:
#             # Determine if it's current year or next year
#             current_month = today.month
#             current_year = today.year

#             # If the month has passed this year, use next year
#             if month < current_month:
#                 year = current_year + 1
#             else:
#                 year = current_year

#             start_date = pd.Timestamp(year=year, month=month, day=1).normalize()
#             # Get last day of the month
#             if month == 12:
#                 end_date = pd.Timestamp(year=year, month=12, day=31).normalize()
#             else:
#                 end_date = (pd.Timestamp(year=year, month=month+1, day=1) - pd.Timedelta(days=1)).normalize()

#             return start_date, end_date, f"{month_name.capitalize()} {year}"

#     # Pattern 3: "today"
#     if re.search(r'\btoday\b', query):
#         return today, today, "today"

#     # Pattern 4: "tomorrow"
#     if re.search(r'\btomorrow\b', query):
#         tomorrow = today + pd.Timedelta(days=1)
#         return tomorrow, tomorrow, "tomorrow"

#     # Pattern 5: "yesterday"
#     if re.search(r'\byesterday\b', query):
#         yesterday = today - pd.Timedelta(days=1)
#         return yesterday, yesterday, "yesterday"

#     # Pattern 6: "next X days" or "in next X days"
#     m = re.search(r'(?:next|in\s+next|in)\s+(\d{1,3})\s+days?', query)
#     if m:
#         n = int(m.group(1))
#         end_date = today + pd.Timedelta(days=n)
#         return today, end_date, f"next {n} days"

#     # Pattern 7: "last X days" or "past X days"
#     m = re.search(r'(?:last|past|previous)\s+(\d{1,3})\s+days?', query)
#     if m:
#         n = int(m.group(1))
#         start_date = today - pd.Timedelta(days=n)
#         return start_date, today, f"last {n} days"

#     # Pattern 8: "this week" (Monday to Sunday)
#     if re.search(r'\bthis\s+week\b', query):
#         start_of_week = today - pd.Timedelta(days=today.weekday())
#         end_of_week = start_of_week + pd.Timedelta(days=6)
#         return start_of_week, end_of_week, "this week"

#     # Pattern 9: "next week"
#     if re.search(r'\bnext\s+week\b', query):
#         start_of_next_week = today + pd.Timedelta(days=7-today.weekday())
#         end_of_next_week = start_of_next_week + pd.Timedelta(days=6)
#         return start_of_next_week, end_of_next_week, "next week"

#     # Pattern 10: "last week"
#     if re.search(r'\blast\s+week\b', query):
#         start_of_last_week = today - pd.Timedelta(days=today.weekday()+7)
#         end_of_last_week = start_of_last_week + pd.Timedelta(days=6)
#         return start_of_last_week, end_of_last_week, "last week"

#     # Pattern 11: "this month"
#     if re.search(r'\bthis\s+month\b', query):
#         start_of_month = pd.Timestamp(year=today.year, month=today.month, day=1).normalize()
#         if today.month == 12:
#             end_of_month = pd.Timestamp(year=today.year, month=12, day=31).normalize()
#         else:
#             end_of_month = (pd.Timestamp(year=today.year, month=today.month+1, day=1) - pd.Timedelta(days=1)).normalize()
#         return start_of_month, end_of_month, "this month"

#     # Pattern 12: "next month"
#     if re.search(r'\bnext\s+month\b', query):
#         if today.month == 12:
#             start_of_next_month = pd.Timestamp(year=today.year+1, month=1, day=1).normalize()
#             end_of_next_month = pd.Timestamp(year=today.year+1, month=1, day=31).normalize()
#         else:
#             start_of_next_month = pd.Timestamp(year=today.year, month=today.month+1, day=1).normalize()
#             if today.month == 11:
#                 end_of_next_month = pd.Timestamp(year=today.year, month=12, day=31).normalize()
#             else:
#                 end_of_next_month = (pd.Timestamp(year=today.year, month=today.month+2, day=1) - pd.Timedelta(days=1)).normalize()
#         return start_of_next_month, end_of_next_month, "next month"

#     # Pattern 13: "last month"
#     if re.search(r'\blast\s+month\b', query):
#         if today.month == 1:
#             start_of_last_month = pd.Timestamp(year=today.year-1, month=12, day=1).normalize()
#             end_of_last_month = pd.Timestamp(year=today.year-1, month=12, day=31).normalize()
#         else:
#             start_of_last_month = pd.Timestamp(year=today.year, month=today.month-1, day=1).normalize()
#             end_of_last_month = (pd.Timestamp(year=today.year, month=today.month, day=1) - pd.Timedelta(days=1)).normalize()
#         return start_of_last_month, end_of_last_month, "last month"

#     # Pattern 14: Explicit date ranges "from YYYY-MM-DD to YYYY-MM-DD"
#     m = re.search(r'from\s+(\d{4}[-/]\d{1,2}[-/]\d{1,2})\s+to\s+(\d{4}[-/]\d{1,2}[-/]\d{1,2})', query)
#     if m:
#         start_date = pd.to_datetime(m.group(1), errors='coerce').normalize()
#         end_date = pd.to_datetime(m.group(2), errors='coerce').normalize()
#         return start_date, end_date, f"from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}"

#     # Pattern 15: "between DATE1 and DATE2"
#     m = re.search(r'between\s+(\d{4}[-/]\d{1,2}[-/]\d{1,2})\s+and\s+(\d{4}[-/]\d{1,2}[-/]\d{1,2})', query)
#     if m:
#         start_date = pd.to_datetime(m.group(1), errors='coerce').normalize()
#         end_date = pd.to_datetime(m.group(2), errors='coerce').normalize()
#         return start_date, end_date, f"between {start_date.strftime('%Y-%m-%d')} and {end_date.strftime('%Y-%m-%d')}"

#     # Default: next 7 days
#     end_date = today + pd.Timedelta(days=7)
#     return today, end_date, "next 7 days (default)"


def format_date_for_display(dt: pd.Timestamp) -> str:
    """
    Format a pandas Timestamp for display.
    Returns format: YYYY-MM-DD
    """
    if pd.isna(dt):
        return None
    return dt.strftime("%Y-%m-%d")


def is_date_in_range(
    date: pd.Timestamp, start: pd.Timestamp, end: pd.Timestamp
) -> bool:
    """
    Check if a date falls within a date range (inclusive).
    """
    if pd.isna(date):
        return False
    return start <= date <= end
