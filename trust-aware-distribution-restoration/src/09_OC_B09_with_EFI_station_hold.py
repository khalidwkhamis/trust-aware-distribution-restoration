"""
ECEPRY B09-Side Overcurrent with Useful EFI Evidence
Suspected ME204 Station Flashover — No Remote Re-energization
================================================================

PURPOSE
-------
This simulation tests the project O/C logic when EFI information IS useful.

Rule implemented:
1. If a feeder trips on O/C, check EFI evidence first.
2. If the EFI pattern provides useful directional evidence, use it to narrow
   the suspect region.
3. Do NOT treat the EFI boundary as deterministic cable-fault localization
   the way we do for a clean Earth Fault.
4. Because O/C may represent station flashover, the first station beyond the
   positive EFI evidence can be held out of service conservatively.
5. Restore healthy areas around the suspect station without remotely
   re-energizing it.
6. Field inspection is required before the suspect station can return.

TRIPPED SOURCE
--------------
B09

PRE-FAULT POWER-FLOW / EFI LOCALIZATION DIRECTION
-------------------------------------------------
B09 -> ME205 -> ECE205 -> ME204 -> ECE204 -> NOP

All 10 station EFI values are available to SCADA, but only this B09-fed chain
is used for this event's localization.

USEFUL EFI PATTERN
------------------
ME205  = Operated
ECE205 = Operated    <- last positive EFI evidence
ME204  = Not operated <- first station beyond positive EFI evidence
ECE204 = Not operated

This is a clean, monotonic B09-side EFI pattern.

For an Earth Fault, that boundary could be used to infer a cable.
For THIS O/C case, the controller uses it only to narrow the unsafe region.

PROJECT INTERPRETATION
----------------------
The O/C event plus EFI boundary makes ME204 a suspected unsafe station /
possible flashover location.

Therefore:
- B09 remains open while isolation is performed.
- Open ME204 toward ECE205.
- Open ME204 toward ECE204.
- Reclose B09 to restore ME205 and ECE205 only.
- Close the NOP from the A01 side to restore ECE204.
- Do NOT energize ME204 from either direction.
- ME204 remains isolated until field inspection.

EXPECTED FINAL SERVED LOAD
--------------------------
Total nominal load = 300 A
ME204 load = 35 A

Expected healthy served nominal load:
    300 A - 35 A = 265 A

Expected nominal source split:
    B09 = ME205 15 A + ECE205 35 A = 50 A
    A01 = original A01 side 200 A + ECE204 15 A = 215 A

Actual solved currents will be slightly higher because loads are constant P/Q
and cable impedance causes voltage drop/losses.

MODEL ASSUMPTIONS
-----------------
- Balanced 11 kV, three-phase steady-state model.
- PF = 0.95.
- Current operating limit = 350 A.
- Voltage safe band = 0.94-1.06 pu.
- Synthetic identical cable sections:
      length = 0.5 km
      R = 0.25 ohm/km
      X = 0.08 ohm/km
- A01 and B09 are ideal 1.0 pu sources.
- All station feeder switches are remotely controlled.
- NOP = ME203 feeder switch toward ECE204.
- No detailed arc or relay transient physics are modeled.
"""

import math
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import networkx as nx


# ============================================================
# 1. Project parameters
# ============================================================

V_LL = 11_000.0
V_PHASE = V_LL / math.sqrt(3)
POWER_FACTOR = 0.95
Q_FACTOR = math.sqrt(1 - POWER_FACTOR**2)

CURRENT_LIMIT_A = 350.0
V_MIN_PU = 0.94
V_MAX_PU = 1.06

CABLE_LENGTH_KM = 0.50
R_OHM_PER_KM = 0.25
X_OHM_PER_KM = 0.08
Z_SECTION = complex(
    R_OHM_PER_KM * CABLE_LENGTH_KM,
    X_OHM_PER_KM * CABLE_LENGTH_KM,
)

SUSPECT_STATION = "ME204"


# ============================================================
# 2. Feeder topology and station loads
# ============================================================

STATIONS_A_TO_B = [
    "ECE201", "ME201", "ECE202", "ME202", "ECE203",
    "ME203", "ECE204", "ME204", "ECE205", "ME205",
]

NOMINAL_LOAD_A = {
    "ECE201": 45,
    "ME201": 20,
    "ECE202": 45,
    "ME202": 25,
    "ECE203": 45,
    "ME203": 20,
    "ECE204": 15,
    "ME204": 35,
    "ECE205": 35,
    "ME205": 15,
}

NODES = ["A01"] + STATIONS_A_TO_B + ["B09"]
CABLES = list(zip(NODES[:-1], NODES[1:]))

EDGE_SWITCHES = {}
for left, right in CABLES:
    if left == "A01":
        EDGE_SWITCHES[(left, right)] = (
            "A01_breaker",
            f"{right}_to_A01",
        )
    elif right == "B09":
        EDGE_SWITCHES[(left, right)] = (
            f"{left}_to_B09",
            "B09_breaker",
        )
    else:
        EDGE_SWITCHES[(left, right)] = (
            f"{left}_to_{right}",
            f"{right}_to_{left}",
        )

NORMAL_SWITCH_STATE = {}
for devices in EDGE_SWITCHES.values():
    for device in devices:
        NORMAL_SWITCH_STATE[device] = True

NORMAL_SWITCH_STATE["ME203_to_ECE204"] = False

EFI_SCAN_CHAINS = {
    "A01": ["ECE201", "ME201", "ECE202", "ME202", "ECE203", "ME203"],
    "B09": ["ME205", "ECE205", "ME204", "ECE204"],
}


# ============================================================
# 3. Load conversion
# ============================================================

def three_phase_load_va(current_a):
    s_va = math.sqrt(3) * V_LL * current_a
    return complex(
        s_va * POWER_FACTOR,
        s_va * Q_FACTOR,
    )


S_LOAD_3PH = {
    station: three_phase_load_va(current)
    for station, current in NOMINAL_LOAD_A.items()
}


# ============================================================
# 4. O/C + EFI evidence analysis
# ============================================================

def analyze_oc_efi(tripped_source, all_efi_status):
    """
    Analyze EFI evidence in the pre-fault power-flow direction of the source
    that tripped.

    IMPORTANT:
    For O/C, this function does NOT declare a deterministic cable fault.
    It identifies:
      - whether the EFI pattern is useful/consistent,
      - the last operated EFI,
      - the first not-operated EFI,
      - the first station beyond positive EFI evidence.

    The supervisory controller can then conservatively hold that station/
    local region out of service.
    """
    scan = EFI_SCAN_CHAINS[tripped_source]

    seen_not = False
    contradictions = []

    for station in scan:
        operated = bool(all_efi_status[station])

        if not operated:
            seen_not = True
        elif seen_not:
            contradictions.append(station)

    if contradictions:
        return {
            "useful": False,
            "reason": (
                "EFI pattern is contradictory in the tripped source's "
                "pre-fault power-flow direction."
            ),
            "last_operated_station": None,
            "first_not_operated_station": None,
            "suspected_station_or_region": None,
        }

    first_not_index = next(
        (i for i, s in enumerate(scan) if not all_efi_status[s]),
        None,
    )

    if first_not_index is None:
        return {
            "useful": False,
            "reason": (
                "All relevant EFIs operated; EFI does not identify a useful "
                "transition boundary for this O/C event."
            ),
            "last_operated_station": scan[-1],
            "first_not_operated_station": None,
            "suspected_station_or_region": None,
        }

    first_not = scan[first_not_index]

    if first_not_index == 0:
        last_operated = None
    else:
        last_operated = scan[first_not_index - 1]

    return {
        "useful": True,
        "reason": (
            "EFI pattern is monotonic and provides directional evidence. "
            "For O/C, use the transition to narrow the suspect region; do "
            "not treat it as deterministic cable-fault localization."
        ),
        "last_operated_station": last_operated,
        "first_not_operated_station": first_not,
        "suspected_station_or_region": first_not,
    }


def build_efi_table(tripped_source, all_efi_status, analysis):
    scan = EFI_SCAN_CHAINS[tripped_source]
    order = {s: i + 1 for i, s in enumerate(scan)}

    rows = []

    for station in STATIONS_A_TO_B:
        relevant = station in scan

        if not relevant:
            interpretation = (
                "EFI available to SCADA but outside the B09 pre-fault "
                "localization chain for this trip."
            )
        elif station == analysis["last_operated_station"]:
            interpretation = (
                "Last positive EFI evidence in B09 power-flow direction."
            )
        elif station == analysis["first_not_operated_station"]:
            interpretation = (
                "First station beyond positive EFI evidence; for this O/C "
                "case it is conservatively treated as the suspect station/region."
            )
        elif all_efi_status[station]:
            interpretation = (
                "EFI operated; O/C disturbance evidence propagated through "
                "this station in the B09 power-flow direction."
            )
        else:
            interpretation = (
                "EFI not operated downstream of the useful transition boundary."
            )

        rows.append({
            "station": station,
            "EFI_status": "Operated" if all_efi_status[station] else "Not operated",
            "EFI_available_to_SCADA": True,
            "relevant_to_B09_localization": relevant,
            "scan_order_from_B09": order.get(station, ""),
            "is_last_operated_EFI": (
                station == analysis["last_operated_station"]
            ),
            "is_first_not_operated_EFI": (
                station == analysis["first_not_operated_station"]
            ),
            "interpretation": interpretation,
        })

    return pd.DataFrame(rows)


# ============================================================
# 5. Electrical model
# ============================================================

def active_cables(switch_state):
    return [
        cable
        for cable, devices in EDGE_SWITCHES.items()
        if all(switch_state[d] for d in devices)
    ]


def solve_radial_component(component_nodes, component_edges, source):
    graph = nx.Graph()
    graph.add_nodes_from(component_nodes)
    graph.add_edges_from(component_edges)

    if not nx.is_tree(graph):
        raise ValueError(
            "Energized component is not radial; source parallel/loop detected."
        )

    tree = nx.bfs_tree(graph, source)
    order = list(tree.nodes)

    parent = {source: None}
    children = {n: list(tree.successors(n)) for n in tree.nodes}

    for node in order[1:]:
        parent[node] = next(tree.predecessors(node))

    voltage = {n: complex(V_PHASE, 0.0) for n in component_nodes}
    branch_current = {}

    for _ in range(100):
        old = voltage.copy()

        injected = {}
        for node in component_nodes:
            s_phase = S_LOAD_3PH.get(node, 0j) / 3.0
            injected[node] = (
                0j if s_phase == 0
                else np.conj(s_phase / voltage[node])
            )

        downstream = {}
        for node in reversed(order):
            total = injected[node]

            for child in children.get(node, []):
                total += downstream[child]

            downstream[node] = total

            if parent[node] is not None:
                branch_current[(parent[node], node)] = total

        voltage[source] = complex(V_PHASE, 0.0)

        for node in order[1:]:
            p = parent[node]
            voltage[node] = voltage[p] - Z_SECTION * branch_current[(p, node)]

        if max(abs(voltage[n] - old[n]) for n in component_nodes) < 1e-6:
            break

    return voltage, branch_current, children


def evaluate_state(switch_state):
    active = active_cables(switch_state)

    graph = nx.Graph()
    graph.add_nodes_from(NODES)
    graph.add_edges_from(active)

    station_rows = []
    branch_rows = []
    source_rows = []
    parallel = False

    for component in nx.connected_components(graph):
        component = set(component)
        sources = [s for s in ("A01", "B09") if s in component]
        edges = [
            e for e in active
            if e[0] in component and e[1] in component
        ]
        loads = [n for n in component if n in STATIONS_A_TO_B]

        if len(sources) == 0:
            for station in loads:
                station_rows.append({
                    "station": station,
                    "energized": False,
                    "supplying_source": "",
                    "voltage_pu": 0.0,
                    "nominal_load_A": NOMINAL_LOAD_A[station],
                })
            continue

        if len(sources) > 1:
            parallel = True

            for station in loads:
                station_rows.append({
                    "station": station,
                    "energized": True,
                    "supplying_source": "INVALID A01+B09 PARALLEL",
                    "voltage_pu": float("nan"),
                    "nominal_load_A": NOMINAL_LOAD_A[station],
                })
            continue

        source = sources[0]

        voltage, branch_current, children = solve_radial_component(
            component,
            edges,
            source,
        )

        for station in loads:
            station_rows.append({
                "station": station,
                "energized": True,
                "supplying_source": source,
                "voltage_pu": abs(voltage[station]) / V_PHASE,
                "nominal_load_A": NOMINAL_LOAD_A[station],
            })

        for (parent, child), current in branch_current.items():
            physical = (
                (parent, child)
                if (parent, child) in CABLES
                else (child, parent)
            )

            branch_rows.append({
                "supplying_source": source,
                "from": physical[0],
                "to": physical[1],
                "current_A": abs(current),
            })

        source_current = abs(
            sum(
                (branch_current[(source, child)] for child in children[source]),
                0j,
            )
        )

        source_rows.append({
            "source": source,
            "current_A": source_current,
            "nominal_served_load_A": sum(
                NOMINAL_LOAD_A[n] for n in loads
            ),
        })

    stations_df = pd.DataFrame(station_rows)
    branches_df = pd.DataFrame(branch_rows)
    sources_df = pd.DataFrame(source_rows)

    max_current = (
        float(branches_df["current_A"].max())
        if not branches_df.empty
        else 0.0
    )

    energized_v = (
        stations_df.loc[stations_df["energized"], "voltage_pu"]
        if not stations_df.empty
        else pd.Series(dtype=float)
    )

    min_voltage = (
        float(energized_v.min())
        if not energized_v.empty
        else float("nan")
    )
    max_voltage = (
        float(energized_v.max())
        if not energized_v.empty
        else float("nan")
    )

    current_ok = max_current <= CURRENT_LIMIT_A
    voltage_ok = (
        energized_v.empty
        or (min_voltage >= V_MIN_PU and max_voltage <= V_MAX_PU)
    )

    return stations_df, branches_df, sources_df, {
        "max_branch_current_A": max_current,
        "min_voltage_pu": min_voltage,
        "max_voltage_pu": max_voltage,
        "current_limit_ok": bool(current_ok),
        "voltage_limit_ok": bool(voltage_ok),
        "unintended_source_parallel": bool(parallel),
        "electrically_admissible": bool(
            current_ok and voltage_ok and not parallel
        ),
    }


# ============================================================
# 6. B09 O/C + useful EFI scenario
# ============================================================

def run_scenario():
    tripped_source = "B09"

    # All station EFI readings remain available to SCADA.
    # Only the B09-fed pre-fault chain is relevant for this trip.
    all_efi = {
        "ECE201": False,
        "ME201": False,
        "ECE202": False,
        "ME202": False,
        "ECE203": False,
        "ME203": False,
        "ECE204": False,
        "ME204": False,
        "ECE205": True,
        "ME205": True,
    }

    analysis = analyze_oc_efi(
        tripped_source,
        all_efi,
    )

    if not analysis["useful"]:
        raise RuntimeError(
            "This test requires useful EFI evidence, but analysis rejected it."
        )

    if analysis["suspected_station_or_region"] != SUSPECT_STATION:
        raise RuntimeError(
            f"Expected suspect station {SUSPECT_STATION}, got "
            f"{analysis['suspected_station_or_region']}"
        )

    efi_table = build_efi_table(
        tripped_source,
        all_efi,
        analysis,
    )

    state = deepcopy(NORMAL_SWITCH_STATE)
    events = []
    reenergization_after_hold = 0
    suspect_hold_active = False

    def record(action, basis, protection_state="No new trip"):
        nonlocal reenergization_after_hold

        stations, branches, sources, electrical = evaluate_state(state)

        current = {
            row["source"]: row["current_A"]
            for _, row in sources.iterrows()
        }

        suspect_row = stations.loc[
            stations["station"] == SUSPECT_STATION
        ].iloc[0]

        suspect_energized = bool(suspect_row["energized"])

        if suspect_hold_active and suspect_energized:
            reenergization_after_hold += 1

        events.append({
            "step": len(events),
            "action": action,
            "decision_basis": basis,
            "protection_state": protection_state,
            "tripped_source": tripped_source,
            "EFI_last_operated": analysis["last_operated_station"],
            "EFI_first_not_operated": analysis[
                "first_not_operated_station"
            ],
            "EFI_suspected_station_or_region": analysis[
                "suspected_station_or_region"
            ],
            "suspect_hold_active": suspect_hold_active,
            "ME204_switch_toward_ECE205": (
                "Closed" if state["ME204_to_ECE205"] else "Open"
            ),
            "ME204_switch_toward_ECE204": (
                "Closed" if state["ME204_to_ECE204"] else "Open"
            ),
            "ME204_energized": suspect_energized,
            "A01_current_A": current.get("A01", 0.0),
            "B09_current_A": current.get("B09", 0.0),
            "served_nominal_load_A": float(
                stations.loc[
                    stations["energized"],
                    "nominal_load_A",
                ].sum()
            ),
            "min_voltage_pu": electrical["min_voltage_pu"],
            "max_branch_current_A": electrical["max_branch_current_A"],
            "current_limit_ok": electrical["current_limit_ok"],
            "voltage_limit_ok": electrical["voltage_limit_ok"],
            "unintended_source_parallel": electrical[
                "unintended_source_parallel"
            ],
            "reenergizations_after_suspect_hold": reenergization_after_hold,
        })

        return stations, branches, sources, electrical

    # --------------------------------------------------------
    # Step 0: B09 trips on O/C.
    # --------------------------------------------------------
    state["B09_breaker"] = False
    record(
        "B09 trips on Overcurrent",
        (
            "O/C protection event. Read network-wide EFI status and interpret "
            "the relevant B09-fed chain in the pre-fault power-flow direction."
        ),
        protection_state="B09 O/C trip",
    )

    # --------------------------------------------------------
    # EFI gives useful boundary:
    # ECE205 Operated -> ME204 Not operated.
    #
    # For O/C, do not declare a deterministic cable fault.
    # Hold ME204 / local region out of service conservatively.
    # --------------------------------------------------------
    suspect_hold_active = True

    state["ME204_to_ECE205"] = False
    record(
        "Open ME204 feeder switch toward ECE205",
        (
            "Useful EFI evidence: ECE205 is the last Operated EFI and ME204 "
            "is the first Not-operated EFI in the B09 flow direction. "
            "For O/C, treat ME204/local region as suspected unsafe rather than "
            "declaring a definite cable fault."
        ),
    )

    state["ME204_to_ECE204"] = False
    record(
        "Open ME204 feeder switch toward ECE204",
        (
            "Isolate the suspected ME204 station/local region from the "
            "opposite feeder direction as well. No remote re-energization "
            "will be permitted until field inspection."
        ),
    )

    # --------------------------------------------------------
    # Restore proven healthy B09-side upstream stations.
    # Because ME204 is isolated, only ME205 and ECE205 reconnect.
    # --------------------------------------------------------
    state["B09_breaker"] = True
    _, _, _, electrical = evaluate_state(state)

    if not electrical["electrically_admissible"]:
        raise RuntimeError("B09 healthy-side restoration failed checks.")

    record(
        "Reclose B09 after ME204 isolation",
        (
            "Restore only the proven healthy B09-side stations ME205 and "
            "ECE205. ME204 remains isolated."
        ),
        protection_state="No re-trip",
    )

    # --------------------------------------------------------
    # Restore healthy ECE204 from A01 through the NOP.
    # ME204_to_ECE204 remains open, so ME204 is not energized.
    # --------------------------------------------------------
    state["ME203_to_ECE204"] = True
    _, _, _, electrical = evaluate_state(state)

    if not electrical["electrically_admissible"]:
        raise RuntimeError("A01 alternate-path restoration failed checks.")

    final_state = record(
        "Close ME203_to_ECE204 NOP",
        (
            "Restore healthy ECE204 from A01. The ME204 feeder switch toward "
            "ECE204 remains open, so the suspected station is not energized "
            "from the alternate side."
        ),
        protection_state="No re-trip",
    )

    return {
        "analysis": analysis,
        "efi_table": efi_table,
        "events": pd.DataFrame(events),
        "final_stations": final_state[0],
        "final_branches": final_state[1],
        "final_sources": final_state[2],
        "final_electrical": final_state[3],
        "final_switch_state": state,
        "reenergization_after_hold": reenergization_after_hold,
    }


# ============================================================
# 7. Save results
# ============================================================

def main():
    output = Path("results")
    output.mkdir(exist_ok=True)

    result = run_scenario()

    result["efi_table"].to_csv(
        output / "01_network_wide_efi_evidence.csv",
        index=False,
    )
    result["events"].to_csv(
        output / "02_controller_action_sequence.csv",
        index=False,
    )
    result["final_stations"].to_csv(
        output / "03_final_station_results.csv",
        index=False,
    )
    result["final_branches"].to_csv(
        output / "04_final_branch_results.csv",
        index=False,
    )
    result["final_sources"].to_csv(
        output / "05_final_source_results.csv",
        index=False,
    )

    final_sources = {
        row["source"]: row["current_A"]
        for _, row in result["final_sources"].iterrows()
    }

    final_station_rows = result["final_stations"]
    me204_row = final_station_rows.loc[
        final_station_rows["station"] == "ME204"
    ].iloc[0]

    final_served = float(
        final_station_rows.loc[
            final_station_rows["energized"],
            "nominal_load_A",
        ].sum()
    )

    summary = pd.DataFrame([{
        "initial_trip": "B09 Overcurrent",
        "EFI_scan_direction": "B09->ME205->ECE205->ME204->ECE204->NOP",
        "EFI_evidence_useful": result["analysis"]["useful"],
        "last_operated_EFI": result["analysis"]["last_operated_station"],
        "first_not_operated_EFI": result["analysis"][
            "first_not_operated_station"
        ],
        "O_C_interpretation": (
            "EFI narrows suspect region; does not prove deterministic cable fault"
        ),
        "suspected_station_or_region": result["analysis"][
            "suspected_station_or_region"
        ],
        "ME204_final_energized": bool(me204_row["energized"]),
        "ME204_switch_toward_ECE205": (
            "Closed"
            if result["final_switch_state"]["ME204_to_ECE205"]
            else "Open"
        ),
        "ME204_switch_toward_ECE204": (
            "Closed"
            if result["final_switch_state"]["ME204_to_ECE204"]
            else "Open"
        ),
        "reenergizations_after_suspect_hold": result[
            "reenergization_after_hold"
        ],
        "final_A01_current_A": final_sources.get("A01", 0.0),
        "final_B09_current_A": final_sources.get("B09", 0.0),
        "final_nominal_served_load_A": final_served,
        "unserved_ME204_load_A": NOMINAL_LOAD_A["ME204"],
        "final_min_voltage_pu": result["final_electrical"]["min_voltage_pu"],
        "final_max_branch_current_A": result["final_electrical"][
            "max_branch_current_A"
        ],
        "final_current_limit_ok": result["final_electrical"][
            "current_limit_ok"
        ],
        "final_voltage_limit_ok": result["final_electrical"][
            "voltage_limit_ok"
        ],
        "final_unintended_source_parallel": result["final_electrical"][
            "unintended_source_parallel"
        ],
        "field_inspection_required_before_ME204_reenergization": True,
    }])

    summary.to_csv(
        output / "00_scenario_summary.csv",
        index=False,
    )

    print("\n=== B09 O/C WITH USEFUL EFI EVIDENCE ===")
    print("Initial trip: B09 Overcurrent")
    print("EFI scan: B09 -> ME205 -> ECE205 -> ME204 -> ECE204 -> NOP")
    print(
        "Useful boundary:",
        result["analysis"]["last_operated_station"],
        "Operated ->",
        result["analysis"]["first_not_operated_station"],
        "Not operated",
    )
    print(
        "O/C interpretation:",
        "narrow suspect region; do not declare deterministic cable fault",
    )

    print("\n=== EFI EVIDENCE ===")
    print(result["efi_table"].to_string(index=False))

    print("\n=== CONTROLLER ACTION SEQUENCE ===")
    print(
        result["events"][
            [
                "step",
                "action",
                "EFI_last_operated",
                "EFI_first_not_operated",
                "EFI_suspected_station_or_region",
                "ME204_switch_toward_ECE205",
                "ME204_switch_toward_ECE204",
                "ME204_energized",
                "A01_current_A",
                "B09_current_A",
                "served_nominal_load_A",
                "reenergizations_after_suspect_hold",
            ]
        ].to_string(index=False)
    )

    print("\n=== FINAL SOURCE LOADING ===")
    print(result["final_sources"].to_string(index=False))

    print("\n=== SUMMARY ===")
    print(summary.to_string(index=False))

    print("\n=== INTENDED INTERPRETATION ===")
    print(
        "1) B09 trips on O/C, so EFI is interpreted in the B09 pre-fault "
        "power-flow direction."
    )
    print(
        "2) ME205 and ECE205 report Operated; ME204 is the first Not-operated "
        "EFI. This is useful directional evidence."
    )
    print(
        "3) Because this is O/C, the controller does NOT call ECE205-ME204 a "
        "confirmed cable fault. It treats ME204/local region as suspect."
    )
    print(
        "4) Both ME204 feeder switches are opened, so ME204 cannot be energized "
        "from B09 or A01."
    )
    print(
        "5) B09 is reclosed to restore ME205 and ECE205 only."
    )
    print(
        "6) The NOP is closed so A01 restores ECE204 while ME204 remains isolated."
    )
    print(
        "7) ME204 is never re-energized after the suspect hold is declared."
    )
    print(
        "8) ME204's 35 A nominal load remains unserved until field inspection."
    )


if __name__ == "__main__":
    main()
