import re
import threading

from agents.prompts import map_synonym_to_column
from agents.tools import _df  # Import the DataFrame function to test filtering
from agents.tools import get_hot_containers  # Add this missing import
from agents.tools import get_shipped_quantity  # Add shipped quantity function
from agents.tools import (  # Add the missing functions
    analyze_data_with_pandas, answer_with_column_mapping, check_arrival_status,
    check_po_month_arrival, check_transit_status, get_arrivals_by_port,
    get_bl_transit_analysis, get_blob_sql_engine, get_booking_details,
    get_bulk_container_transit_analysis, get_container_carrier,
    get_container_etd, get_container_milestones,
    get_container_transit_analysis, get_containers_arriving_soon,
    get_containers_at_dp_not_fd, get_containers_by_carrier,
    get_containers_by_etd_window, get_containers_departed_from_load_port,
    get_containers_departing_from_load_port, get_containers_missed_planned_etd,
    get_containers_PO_OBL_by_supplier, get_containers_still_at_load_port,
    get_delayed_containers, get_delayed_containers_not_arrived,
    get_delayed_pos, get_eta_for_booking, get_eta_for_po, get_field_info,
    get_hot_upcoming_arrivals, get_load_port_for_container,
    get_po_transit_analysis, get_upcoming_arrivals, get_upcoming_bls,
    get_upcoming_pos, get_vessel_info, get_weekly_status_changes,
    lookup_keyword, sql_query_tool, vector_search_tool)
from utils.logger import logger


def validate_consignee_filtering(consignee_codes: list) -> bool:
    """Validate that consignee filtering is working properly"""
    try:
        # Get filtered DataFrame
        df_filtered = _df()

        if df_filtered.empty:
            logger.warning(
                f"No data found after consignee filtering for codes: {consignee_codes}"
            )
            return False

        # Check if consignee codes are present in the filtered data
        if "consignee_code_multiple" in df_filtered.columns:
            # Extract numeric codes from consignee_codes
            import re

            numeric_codes = []
            for code in consignee_codes:
                # Extract numeric part (e.g., "0045831" from "EDDIE BAUER LLC(0045831)")
                match = re.search(r"\((\d+)\)", code)
                if match:
                    numeric_codes.append(match.group(1))
                else:
                    # If already just numeric, use as is
                    numeric_codes.append(code.strip())

            # Check if any of the codes appear in the filtered data
            pattern = r"|".join([rf"\b{re.escape(code)}\b" for code in numeric_codes])
            has_authorized_data = (
                df_filtered["consignee_code_multiple"]
                .astype(str)
                .apply(lambda x: bool(re.search(pattern, x)))
                .any()
            )

            if not has_authorized_data:
                logger.warning(
                    f"Filtered data doesn't contain authorized consignee codes: {numeric_codes}"
                )
                return False

            logger.info(
                f"Consignee filtering validated: {len(df_filtered)} rows for codes {numeric_codes}"
            )
            return True
        else:
            logger.warning("No consignee_code_multiple column found in filtered data")
            return False

    except Exception as e:
        logger.error(f"Error validating consignee filtering: {e}", exc_info=True)
        return False


def route_query(query: str, consignee_codes: list = None) -> str:
    """Route query with consignee authorization support"""
    try:
        # Set thread-local consignee codes for tools
        if consignee_codes:
            threading.current_thread().consignee_codes = consignee_codes
            logger.info(
                f"Router: Set consignee codes {consignee_codes} for query: {query}"
            )

            # ========== VALIDATE CONSIGNEE FILTERING ==========
            if not validate_consignee_filtering(consignee_codes):
                return f"No data available for your authorized consignee codes: {consignee_codes}"

        q = query.lower()

        # Enhanced container/PO/OBL/Booking detection (do this early)
        from utils.container import (extract_container_number,
                                     extract_ocean_bl_number,
                                     extract_po_number)

        container_no = extract_container_number(query)
        po_no = extract_po_number(query)
        try:
            obl_no = extract_ocean_bl_number(query)
        except Exception:
            obl_no = None

        # ========== NEW: BL/OBL upcoming/delayed queries ==========
        # Route BL queries with time windows or delay keywords to get_upcoming_bls
        # BUT NOT if query asks for transit/delay analysis (those go to get_bl_transit_analysis)
        bl_keywords = ["obl", "ocean bl", "bill of lading", "bl ", " bl", "bill"]
        has_bl_keyword = any(kw in q for kw in bl_keywords) or obl_no

        # Check if this is a TRANSIT/ANALYSIS query (should NOT route to upcoming/delayed BLs)
        is_analysis_query = any(
            kw in q
            for kw in [
                "transit",
                "transit time",
                "transit analysis",
                "transit performance",
                "journey",
                "delay analysis",
                "analysis",
            ]
        )

        if (
            has_bl_keyword
            and not is_analysis_query
            and (
                # Upcoming BLs
                re.search(
                    r"(?:upcoming|arriving|arrive|expected|coming|next|within)\s+\d{1,3}\s+days?",
                    q,
                )
                or "arriving" in q
                or "upcoming" in q
                or "expected" in q
                or
                # Delayed BLs (but NOT "delay analysis")
                any(w in q for w in ["delayed", "late", "overdue", "behind", "missed"])
                or
                # Hot BLs
                "hot" in q
            )
        ):
            logger.info(f"Router: Upcoming/Delayed BLs route for query: {query}")
            return get_upcoming_bls(query)

        if "eta" in q and (po_no or re.search(r"\bpo\b", q)):
            return get_eta_for_po(query)

        # ========== PRIORITY 1: Handle port/location queries ==========
        if any(
            phrase in q
            for phrase in ["list containers from", "containers from", "from port"]
        ):
            return get_arrivals_by_port(query)

        # ========== PRIORITY 2: Handle delay queries ==========
        # Check if query is about delayed containers arriving/expected at a port (not yet arrived)
        # Look for patterns: "delayed...arriving at", "delayed...which are arriving", "delayed...at [PORT]"
        is_delayed_not_arrived_query = (
            ("delay" in q or "late" in q or "overdue" in q)
            and "days" in q
            and (
                "arriving at" in q
                or "arriving in" in q
                or "which are arriving" in q
                or re.search(r"arriving\s+at\s+\w+", q)
                or re.search(
                    r"delayed.*at\s+[A-Z]{2,}", query
                )  # "delayed...at SHANGHAI"
            )
        )

        if is_delayed_not_arrived_query:
            # This is for containers delayed but not yet arrived
            logger.info(
                f"Router: Delayed containers not arrived route for query: {query}"
            )
            return get_delayed_containers_not_arrived(query)
        elif (
            "delay" in q or "late" in q or "overdue" in q or "missed" in q
        ) and "days" in q:
            # This is for containers that already arrived late
            # CRITICAL: This handles "hot containers delayed by X days" queries
            # because get_delayed_containers has built-in hot filtering
            logger.info(f"Router: Delayed containers route for query: {query}")
            return get_delayed_containers(query)

        # ========== PRIORITY 2.5: Handle "arrived at DP but not at FD" queries ==========
        # Check for queries about containers that reached discharge port but not delivered
        is_dp_not_fd_query = (
            ("arrived" in q or "reached" in q)
            and ("dp" in q or "discharge port" in q or "discharge" in q)
            and ("not" in q or "not yet" in q or "waiting" in q)
            and (
                "fd" in q
                or "final destination" in q
                or "delivered" in q
                or "delivery" in q
                or "consignee" in q
            )
        )

        if is_dp_not_fd_query:
            logger.info(f"Router: Containers at DP not FD route for query: {query}")
            return get_containers_at_dp_not_fd(query)

        # ========== PRIORITY 3: Handle carrier queries (NEW - SPECIFIC) ==========
        # Questions 15, 16, 17, 18: Carrier queries for PO/Container/OBL
        if ("carrier" in q or "who is" in q) and (po_no or container_no or "obl" in q):
            return get_container_carrier(query)

        # ========== PRIORITY 4: Hot containers routing ==========
        # NOTE: Queries with "hot" + delay thresholds ("delayed by X days")
        # are handled by PRIORITY 2 above, not here
        if (
            ("hot container" in q or "hot containers" in q)
            and "delay" not in q
            and "late" not in q
        ):
            return get_hot_containers(query)
        elif "hot container" in q or "hot containers" in q:
            # If hot + delay/late without specific day threshold, use get_hot_containers
            # Check if there's a specific day threshold
            has_day_threshold = re.search(r"\d+\s+days?", query, re.IGNORECASE)
            if has_day_threshold:
                logger.info(
                    f"Router: Hot containers with delay threshold -> get_delayed_containers for query: {query}"
                )
                return get_delayed_containers(query)
            else:
                return get_hot_containers(query)

        # ========== PRIORITY 5: Container status queries ==========
        if any(
            keyword in q
            for keyword in [
                "milestone",
                "status",
                "track",
                "event history",
                "journey",
                "where",
            ]
        ):
            return get_container_milestones(query)

        # ========== PRIORITY 6: Question 8 - POs shipping in coming days ==========
        elif (
            "po" in q and ("ship" in q or "etd" in q) and ("coming" in q or "next" in q)
        ):
            return get_upcoming_pos(query)

        # ========== PRIORITY 7: Question 14 - Cargo in transit ==========
        if ("cargo" in q or "po" in q) and "transit" in q:
            return check_transit_status(query)

        # ========== PRIORITY 8: Question 19-22 - Carrier/Supplier based queries ==========
        if (
            "carrier" in q
            and ("container" in q or "ship" in q)
            and ("last" in q or "days" in q)
        ):
            return get_containers_by_carrier(query)

        # Supplier/shipper queries:
        # Only route here when user explicitly asks "supplier for / shipper for ..."
        supplier_lookup_phrases = [
            "what is the supplier",
            "what is the shipper",
            "who is the supplier",
            "who is the shipper",
            "supplier for",
            "shipper for",
            "show supplier",
            "show shipper",
            "tell me the supplier",
            "tell me the shipper",
            "get supplier",
            "get shipper",
            "find supplier",
            "find shipper",
        ]
        has_supplier_lookup_phrase = any(p in q for p in supplier_lookup_phrases)

        if has_supplier_lookup_phrase and ("container" in q or "po" in q or "obl" in q):
            return get_containers_PO_OBL_by_supplier(query)

        # ========== PRIORITY 9: Question 24 - PO arrival by month end ==========
        if (
            "po" in q
            and ("arrive" in q or "destination" in q)
            and ("month" in q or "end" in q)
        ):
            return check_po_month_arrival(query)

        # ========== PRIORITY 10: Question 25 - Delayed POs ==========
        if "po" in q and ("delay" in q or "delaying" in q):
            return get_delayed_pos(query)

        # ========== PRIORITY 11: Question 27 - Weekly status changes ==========
        if "status" in q and ("week" in q or "change" in q):
            return get_weekly_status_changes(query)

        # ========== PRIORITY 11-2: Container by final destination queries ==========
        if ("container" in q or "containers" in q) and (
            ("final destination" in q)
            or ("fd " in q + " ")
            or ("dc " in q + " ")
            or ("distribution center" in q)
        ):
            from agents.tools import get_containers_by_final_destination

            return get_containers_by_final_destination(query)

        # ========== PRIORITY 12: Transit Analysis Queries ==========
        # Single container transit analysis with optional filters
        if container_no and any(
            keyword in q
            for keyword in [
                "transit",
                "journey",
                "delay",
                "transit time",
                "actual transit",
                "estimated transit",
                "transit analysis",
                "transit performance",
                "how long",
                "take to arrive",
            ]
        ):
            logger.info(f"Router: Single container transit analysis route for: {query}")
            return get_container_transit_analysis(query)

        # PO transit analysis (multi-container)
        if po_no and any(
            keyword in q
            for keyword in [
                "transit",
                "transit time",
                "transit analysis",
                "transit performance",
                "how are containers",
                "container performance",
            ]
        ):
            logger.info(f"Router: PO transit analysis route for: {query}")
            return get_po_transit_analysis(query)

        # BL transit analysis (multi-container)
        if (
            obl_no
            or any(
                bl_kw in q
                for bl_kw in ["bl ", " bl", "ocean bl", "bill of lading", "obl"]
            )
        ) and any(
            keyword in q
            for keyword in [
                "transit",
                "transit time",
                "transit analysis",
                "transit performance",
                "journey",
                "delay analysis",
                "how are containers",
                "performing",
            ]
        ):
            logger.info(f"Router: BL transit analysis route for: {query}")
            return get_bl_transit_analysis(query)

        # Bulk container transit analysis (route-based, time-based)
        if (
            ("transit" in q or "journey" in q)
            and not container_no
            and not po_no
            and (
                any(
                    port_kw in q
                    for port_kw in ["from", "to", "shanghai", "rotterdam", "port"]
                )
                or any(
                    time_kw in q for time_kw in ["this month", "last month", "average"]
                )
            )
        ):
            logger.info(f"Router: Bulk container transit analysis route for: {query}")
            return get_bulk_container_transit_analysis(query)

        # ========== PRIORITY 13: Upcoming arrivals ==========
        # Enhanced detection for arrival queries with dates, locations, or timeframes
        has_arrival_keyword = any(kw in q for kw in ["arriving", "arrive", "arrival"])
        has_time_keyword = any(
            kw in q
            for kw in [
                "next",
                "coming",
                "upcoming",
                "within",
                "in jan",
                "in feb",
                "in mar",
                "in apr",
                "in may",
                "in jun",
                "in jul",
                "in aug",
                "in sep",
                "in oct",
                "in nov",
                "in dec",
                "this month",
                "last month",
                "this week",
                "last week",
                "today",
                "tomorrow",
                "days",
            ]
        )
        has_going_to = "going to" in q or "scheduled to" in q or "scheduled for" in q
        has_location_pattern = re.search(
            r"\b(?:to|at|in)\s+[A-Z][A-Za-z\s,\.]+(?:,\s*[A-Z]{2})?\b",
            query,
            re.IGNORECASE,
        )

        # Route to get_upcoming_arrivals if:
        # 1. Has arrival keyword + time keyword
        # 2. Has "going to" + location
        # 3. Has location pattern + date/month
        if (
            (has_arrival_keyword and has_time_keyword)
            or (has_going_to and (has_location_pattern or has_time_keyword))
            or (has_location_pattern and has_time_keyword)
        ):
            logger.info(f"Router: Upcoming arrivals route for query: {query}")
            return get_upcoming_arrivals(query)

        # ========== PRIORITY 14: Question 7 - General field info (MOVED TO END) ==========
        # This is now more specific and won't interfere with carrier queries
        if any([container_no, po_no]) and not any(
            keyword in q
            for keyword in [
                "carrier",
                "status",
                "milestone",
                "arrived",
                "track",
                "delay",
                "ship",
                "transit",
                "supplier",
                "where",
                "location",
                "event history",
                "journey",
            ]
        ):
            return get_field_info(query)

        # Default fallback
        return "I couldn't understand your query. Please try rephrasing or provide more specific information."

    except Exception as e:
        logger.error(f"Error in route_query: {e}", exc_info=True)
        return f"Error processing your query: {str(e)}"
    finally:
        # ========== ALWAYS CLEAN UP CONSIGNEE CONTEXT ==========
        if consignee_codes and hasattr(threading.current_thread(), "consignee_codes"):
            delattr(threading.current_thread(), "consignee_codes")
            logger.debug("Cleaned up consignee codes from thread context")
