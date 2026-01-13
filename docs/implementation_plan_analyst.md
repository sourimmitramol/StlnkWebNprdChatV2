# Implementation Plan - Unified Data Analyst Engine

This plan outlines the transition from a multi-tool architecture (50+ tools) to a unified **Data Analyst Engine** using a Column-Aware Transformer and a Python REPL.

## Phase 1: Knowledge Base & Fundamentals
- [x] **Task 1.1**: Create `docs/data_dictionary.md` - Define all shipment columns, aliases, and Business Logic rules (e.g., Delay definitions).
- [x] **Task 1.2**: Implement `get_today_date()` tool - Provide the temporal "North Star" for all queries.

## Phase 2: The Analytic Engine
- [x] **Task 2.1**: Create `agents/analytics_engine.py` - Core logic for transforming NL to Python/Pandas code.
- [x] **Task 2.2**: Implement `ShipmentAnalyst` Tool - A high-capacity tool that uses the dictionary + REPL to solve complex data queries (except Milestones).
- [ ] **Task 2.3**: Security & Validation - Ensure the generated code is safe to execute and respects `consignee_code` filtering.

## Phase 3: Integration & Optimization
- [x] **Task 3.1**: Update `agents/azure_agent.py` - Refined system prompt and HARD TOOL RESTRICTION (The Big Three).
- [x] **Task 3.2**: Tool Consolidation - Removed legacy tool selection and routing from `prompts.py`.
- [x] **Task 3.3**: Milestone Preservation - Confirmed `get_container_milestones` remains the only tracking legacy tool.

## Phase 4: Verification
- [ ] **Task 4.1**: Test multi-turn analytics (e.g., "Show delays for Carrier X", then "Break them down by port").
- [ ] **Task 4.2**: Verify performance and latency.
