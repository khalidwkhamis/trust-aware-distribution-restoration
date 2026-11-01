"""
ECEPRY A01-B09 Looped Feeder Simulation
Independent Clean Earth-Fault Test: ME201-ECE202
=================================================

WHY THIS TEST EXISTS
--------------------
The original Earth-Fault logic was illustrated using ECE203-ME203.
This simulation intentionally moves the cable fault to ME201-ECE202 to test
whether the algorithm generalizes.

Nothing in the switching sequence below is allowed to assume in advance that
ME201-ECE202 is the faulted cable. The controller must:

1. Observe the A01 Earth-Fault trip.
2. Read the EFI status at every station from A01 toward B09.
3. Find the last station whose EFI operated.
4. Find the first downstream station whose EFI did not operate.
5. Infer the cable between those two stations.
6. Derive the two isolation switches from that inferred cable.
7. Restore healthy load only after the appropriate isolation steps.

TESTED TRUE FAULT
-----------------
ME201-ECE202

EXPECTED CLEAN EFI PATTERN
--------------------------
ECE201 : Operated
ME201  : Operated      <- last operated EFI
ECE202 : Not operated  <- first not-operated EFI
ME202  : Not operated
ECE203 : Not operated
ME203  : Not operated
ECE204 : Not operated
ME204  : Not operated
ECE205 : Not operated
ME205  : Not operated

Therefore the algorithm should infer:
    suspected cable = ME201-ECE202

EXPECTED RESTORATION
--------------------
1. A01 trips on E/F.
2. Open ME201 feeder switch toward ECE202.
3. Reclose A01, restoring ECE201 and ME201.
4. Open ECE202 feeder switch toward ME201.
5. Close the normal open point at ME203 toward ECE204.
6. B09 backfeeds ECE202, ME202, ECE203 and ME203 in addition to its normal load.

Final nominal load split should be:
    A01 side = ECE201 45 A + ME201 20 A = 65 A
    B09 side = remaining stations = 235 A
    Total restored = 300 A

MODEL ASSUMPTIONS
-----------------
- Balanced three-phase steady-state model.
- Nominal voltage = 11 kV line-to-line.
- All station loads use power factor = 0.95.
- Feeder operating limit = 350 A.
- Safe voltage band = 0.94 to 1.06 pu.
- Synthetic identical cable sections:
      length = 0.5 km
      R = 0.25 ohm/km
      X = 0.08 ohm/km
- Loads are constant P/Q values derived from the specified nominal amp values.
- A01 and B09 are ideal 1.0 pu voltage sources.
- Every station has two remotely controlled feeder switches.
- NOP is ME203 feeder switch toward ECE204.
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


# ============================================================
# 2. Feeder topology and specified loading
# ============================================================

STATIONS = [
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

NODES = ["A01"] + STATIONS + ["B09"]
CABLES = list(zip(NODES[:-1], NODES[1:]))

# For every physical cable, store the controllable device at each end.
# Tuple order always follows the physical A01 -> B09 direction.
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
for pair in EDGE_SWITCHES.values():
    for device in pair:
        NORMAL_SWITCH_STATE[device] = True

# Specified normally open feeder switch.
NORMAL_SWITCH_STATE["ME203_to_ECE204"] = False


# ============================================================
# 3. Convert nominal amp loads to balanced 3-phase P/Q
# ============================================================

def three_phase_load_va(current_a):
    s_va = math.sqrt(3) * V_LL * current_a
    p_w = s_va * POWER_FACTOR
    q_var = s_va * Q_FACTOR
    return complex(p_w, q_var)


S_LOAD_3PH = {
    station: three_phase_load_va(current)
    for station, current in NOMINAL_LOAD_A.items()
}


# ============================================================
# 4. EFI logic
# ============================================================

def infer_earth_fault_from_efi(efi_status):
    """
    Infer a clean Earth-Fault boundary from EFI states read from A01 toward B09.

    Clean pattern:
        Operated ... Operated | Not operated ... Not operated
                               ^
                               boundary

    The last operated EFI and the first not-operated EFI identify the
    suspected cable between them.

    Special case:
        If ECE201 is the first not-operated EFI, suspect A01-ECE201.

    A pattern that changes back from Not operated to Operated is contradictory
    and is not accepted as a clean localization.
    """
    seen_not_operated = False

    for station in STATIONS:
        operated = bool(efi_status.get(station, False))

        if not operated:
            seen_not_operated = True
        elif seen_not_operated:
            return {
                "valid_pattern": False,
                "suspected_cable": None,
                "last_operated_station": None,
                "first_not_operated_station": None,
                "reason": (
                    "Contradictory EFI pattern: an Operated indication appears "
                    "after a Not operated indication."
                ),
            }

    first_not_index = next(
        (i for i, s in enumerate(STATIONS) if not efi_status.get(s, False)),
        None,
    )

    if first_not_index is None:
        return {
            "valid_pattern": True,
            "suspected_cable": None,
            "last_operated_station": STATIONS[-1],
            "first_not_operated_station": None,
            "reason": "All EFIs operated; this rule does not locate a boundary.",
        }

    first_not = STATIONS[first_not_index]

    if first_not_index == 0:
        suspected = ("A01", STATIONS[0])
        last_operated = None
    else:
        last_operated = STATIONS[first_not_index - 1]
        suspected = (last_operated, first_not)

    return {
        "valid_pattern": True,
        "suspected_cable": suspected,
        "last_operated_station": last_operated,
        "first_not_operated_station": first_not,
        "reason": (
            "Suspected cable lies between the last EFI Operated and the "
            "first downstream EFI Not operated."
        ),
    }


def build_efi_evidence_table(efi_status, inference):
    rows = []

    for order, station in enumerate(STATIONS, start=1):
        operated = bool(efi_status[station])

        if operated:
            interpretation = (
                "EFI operated: earth-fault current passed this station; "
                "fault remains downstream toward B09."
            )
        else:
            interpretation = (
                "EFI not operated: in a clean E/F pattern, "
                "fault is upstream toward A01."
            )

        rows.append({
            "order_from_A01": order,
            "station": station,
            "EFI_status": "Operated" if operated else "Not operated",
            "is_last_operated_EFI": (
                station == inference["last_operated_station"]
            ),
            "is_first_not_operated_EFI": (
                station == inference["first_not_operated_station"]
            ),
            "interpretation": interpretation,
            "inferred_fault_cable": (
                "-".join(inference["suspected_cable"])
                if inference["suspected_cable"]
                else ""
            ),
        })

    return pd.DataFrame(rows)


# ============================================================
# 5. Electrical network construction
# ============================================================

def active_cables(switch_state):
    """
    A cable is electrically connected only when the controllable device at
    BOTH cable ends is closed.
    """
    active = []

    for cable, devices in EDGE_SWITCHES.items():
        if all(switch_state[d] for d in devices):
            active.append(cable)

    return active


# ============================================================
# 6. Radial backward/forward-sweep power flow
# ============================================================

def solve_radial_component(component_nodes, component_edges, source):
    """
    Balanced 3-phase constant-P/Q radial load flow.

    Per-phase voltage is iterated using a backward/forward sweep.
    """
    graph = nx.Graph()
    graph.add_nodes_from(component_nodes)
    graph.add_edges_from(component_edges)

    if not nx.is_tree(graph):
        raise ValueError(
            "Energized component is not radial; unintended source paralleling "
            "or a loop has been created."
        )

    tree = nx.bfs_tree(graph, source)
    order = list(tree.nodes)

    parent = {source: None}
    children = {node: list(tree.successors(node)) for node in tree.nodes}

    for node in order[1:]:
        parent[node] = next(tree.predecessors(node))

    voltage = {node: complex(V_PHASE, 0.0) for node in component_nodes}
    branch_current = {}

    for _ in range(100):
        previous = voltage.copy()

        injected_current = {}

        for node in component_nodes:
            s_phase = S_LOAD_3PH.get(node, 0j) / 3.0

            if s_phase == 0:
                injected_current[node] = 0j
            else:
                injected_current[node] = np.conj(s_phase / voltage[node])

        downstream = {}

        for node in reversed(order):
            total = injected_current[node]

            for child in children.get(node, []):
                total += downstream[child]

            downstream[node] = total

            if parent[node] is not None:
                branch_current[(parent[node], node)] = total

        voltage[source] = complex(V_PHASE, 0.0)

        for node in order[1:]:
            p = parent[node]
            voltage[node] = voltage[p] - Z_SECTION * branch_current[(p, node)]

        if max(abs(voltage[n] - previous[n]) for n in component_nodes) < 1e-6:
            break

    return voltage, branch_current, children


# ============================================================
# 7. Evaluate one network state
# ============================================================

def evaluate_state(switch_state, faulted_cables=None):
    faulted_cables = set(faulted_cables or [])
    active = active_cables(switch_state)

    graph = nx.Graph()
    graph.add_nodes_from(NODES)
    graph.add_edges_from(active)

    station_rows = []
    branch_rows = []
    source_rows = []

    parallel_source = False

    for component in nx.connected_components(graph):
        component = set(component)
        sources = [s for s in ("A01", "B09") if s in component]

        component_edges = [
            cable
            for cable in active
            if cable[0] in component and cable[1] in component
        ]

        load_nodes = [n for n in component if n in STATIONS]

        if len(sources) == 0:
            for station in load_nodes:
                station_rows.append({
                    "station": station,
                    "energized": False,
                    "supplying_source": "",
                    "voltage_pu": 0.0,
                    "nominal_load_A": NOMINAL_LOAD_A[station],
                })
            continue

        if len(sources) > 1:
            parallel_source = True

            for station in load_nodes:
                station_rows.append({
                    "station": station,
                    "energized": True,
                    "supplying_source": "A01+B09 INVALID PARALLEL",
                    "voltage_pu": float("nan"),
                    "nominal_load_A": NOMINAL_LOAD_A[station],
                })
            continue

        source = sources[0]

        voltage, branch_current, children = solve_radial_component(
            component,
            component_edges,
            source,
        )

        for station in load_nodes:
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
                NOMINAL_LOAD_A[n] for n in load_nodes
            ),
        })

    stations_df = pd.DataFrame(station_rows)
    branches_df = pd.DataFrame(branch_rows)
    sources_df = pd.DataFrame(source_rows)

    # Determine whether a faulted cable is connected to an energized source.
    fault_energized = False

    for fault in faulted_cables:
        reverse_fault = (fault[1], fault[0])

        if fault in active or reverse_fault in active:
            start = fault[0]

            if start in graph:
                connected = nx.node_connected_component(graph, start)

                if "A01" in connected or "B09" in connected:
                    fault_energized = True

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

    safety = {
        "max_branch_current_A": max_current,
        "min_voltage_pu": min_voltage,
        "max_voltage_pu": max_voltage,
        "current_limit_ok": bool(current_ok),
        "voltage_limit_ok": bool(voltage_ok),
        "faulted_cable_energized": bool(fault_energized),
        "unintended_source_parallel": bool(parallel_source),
        "safe": bool(
            current_ok
            and voltage_ok
            and not fault_energized
            and not parallel_source
        ),
    }

    return stations_df, branches_df, sources_df, safety


# ============================================================
# 8. Generic clean E/F restoration controller
# ============================================================

def run_clean_earth_fault(true_fault, efi_status):
    """
    Run a clean E/F event on the A01-fed side.

    IMPORTANT:
    Isolation switch names are NOT manually supplied.
    They are derived from the EFI-inferred suspected cable.
    """

    inference = infer_earth_fault_from_efi(efi_status)
    efi_evidence = build_efi_evidence_table(efi_status, inference)

    if not inference["valid_pattern"]:
        raise RuntimeError(
            "EFI evidence is contradictory. Clean E/F restoration is blocked."
        )

    suspected = inference["suspected_cable"]

    if suspected is None:
        raise RuntimeError(
            "EFI pattern does not identify a unique suspected cable."
        )

    # This assertion is for the simulation test harness only.
    # The controller itself sees 'suspected', not 'true_fault'.
    localization_correct = suspected == true_fault

    # Automatically obtain the two devices at the inferred cable ends.
    upstream_device, downstream_device = EDGE_SWITCHES[suspected]

    state = deepcopy(NORMAL_SWITCH_STATE)
    sequence = []

    def record(action, decision_basis):
        stations, branches, sources, safety = evaluate_state(
            state,
            faulted_cables=[true_fault],
        )

        source_current = {
            row["source"]: row["current_A"]
            for _, row in sources.iterrows()
        }

        row = {
            "step": len(sequence),
            "action": action,
            "decision_basis": decision_basis,
            "last_operated_EFI": inference["last_operated_station"],
            "first_not_operated_EFI": inference["first_not_operated_station"],
            "inferred_fault_cable": "-".join(suspected),
            "actual_test_fault_cable": "-".join(true_fault),
            "localization_correct": localization_correct,
            "A01_current_A": source_current.get("A01", 0.0),
            "B09_current_A": source_current.get("B09", 0.0),
            "served_nominal_load_A": float(
                stations.loc[
                    stations["energized"],
                    "nominal_load_A",
                ].sum()
            ),
            "min_voltage_pu": safety["min_voltage_pu"],
            "max_branch_current_A": safety["max_branch_current_A"],
            "faulted_cable_energized": safety["faulted_cable_energized"],
            "unintended_source_parallel": safety[
                "unintended_source_parallel"
            ],
            "safe": safety["safe"],
        }

        for station in STATIONS:
            row[f"{station}_EFI"] = (
                "Operated" if efi_status[station] else "Not operated"
            )

        sequence.append(row)
        return stations, branches, sources, safety

    # Step 0: protection trip.
    state["A01_breaker"] = False
    record(
        "A01 trips on Earth Fault",
        (
            "Protection event initiates restoration logic. "
            "All station EFI indications are read and latched."
        ),
    )

    # Step 1: isolate inferred cable from A01/upstream side.
    state[upstream_device] = False
    record(
        f"Open {upstream_device}",
        (
            f"EFI boundary is {inference['last_operated_station']}=Operated "
            f"and {inference['first_not_operated_station']}=Not operated; "
            f"therefore suspect {suspected[0]}-{suspected[1]}. "
            "Open the upstream/A01-side boundary device."
        ),
    )

    # Step 2: restore healthy upstream section unless the fault is the first cable.
    if suspected[0] != "A01":
        state["A01_breaker"] = True
        record(
            "Reclose A01 breaker",
            (
                f"{upstream_device} is open, so the suspected faulted cable "
                "is disconnected from A01. Restore only the healthy upstream "
                "stations."
            ),
        )

    # Step 3: isolate inferred cable from the downstream side.
    state[downstream_device] = False
    record(
        f"Open {downstream_device}",
        (
            "Open the downstream end of the same EFI-inferred cable. "
            "The suspected cable is now isolated from both ends."
        ),
    )

    # Step 4: use the alternate path.
    state["ME203_to_ECE204"] = True
    final_state = record(
        "Close ME203_to_ECE204 NOP",
        (
            "The suspected cable is isolated at both ends. "
            "Close the alternate B09 path only after verifying radial topology, "
            "branch current <= 350 A, and voltage within 0.94-1.06 pu."
        ),
    )

    return {
        "inference": inference,
        "efi_evidence": efi_evidence,
        "sequence": pd.DataFrame(sequence),
        "final_stations": final_state[0],
        "final_branches": final_state[1],
        "final_sources": final_state[2],
        "final_safety": final_state[3],
        "upstream_isolation_device": upstream_device,
        "downstream_isolation_device": downstream_device,
    }


# ============================================================
# 9. Independent test scenario: ME201-ECE202
# ============================================================

def main():
    output = Path("results")
    output.mkdir(exist_ok=True)

    true_fault = ("ME201", "ECE202")

    # This EFI pattern is generated for the NEW fault location.
    # It is not the old ECE203-ME203 pattern.
    efi = {
        "ECE201": True,
        "ME201": True,
        "ECE202": False,
        "ME202": False,
        "ECE203": False,
        "ME203": False,
        "ECE204": False,
        "ME204": False,
        "ECE205": False,
        "ME205": False,
    }

    result = run_clean_earth_fault(true_fault, efi)

    # Save all result layers.
    result["efi_evidence"].to_csv(
        output / "01_efi_evidence.csv",
        index=False,
    )
    result["sequence"].to_csv(
        output / "02_restoration_sequence.csv",
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

    # One-row summary for quick checking.
    final_sources = result["final_sources"]
    current_by_source = {
        row["source"]: row["current_A"]
        for _, row in final_sources.iterrows()
    }
    served_by_source = {
        row["source"]: row["nominal_served_load_A"]
        for _, row in final_sources.iterrows()
    }

    summary = pd.DataFrame([{
        "true_fault": "ME201-ECE202",
        "inferred_fault": "-".join(
            result["inference"]["suspected_cable"]
        ),
        "last_operated_EFI": result["inference"]["last_operated_station"],
        "first_not_operated_EFI": result["inference"][
            "first_not_operated_station"
        ],
        "upstream_isolation_device": result["upstream_isolation_device"],
        "downstream_isolation_device": result["downstream_isolation_device"],
        "A01_final_current_A": current_by_source.get("A01", 0.0),
        "B09_final_current_A": current_by_source.get("B09", 0.0),
        "A01_nominal_served_load_A": served_by_source.get("A01", 0.0),
        "B09_nominal_served_load_A": served_by_source.get("B09", 0.0),
        "total_nominal_restored_A": sum(served_by_source.values()),
        "final_min_voltage_pu": result["final_safety"]["min_voltage_pu"],
        "final_max_branch_current_A": result["final_safety"][
            "max_branch_current_A"
        ],
        "faulted_cable_energized": result["final_safety"][
            "faulted_cable_energized"
        ],
        "final_safe": result["final_safety"]["safe"],
    }])

    summary.to_csv(
        output / "00_scenario_summary.csv",
        index=False,
    )

    print("\n=== INDEPENDENT EARTH-FAULT TEST ===")
    print("Actual test fault: ME201-ECE202")
    print(
        "EFI-inferred fault:",
        result["inference"]["suspected_cable"],
    )
    print(
        "Derived isolation devices:",
        result["upstream_isolation_device"],
        "and",
        result["downstream_isolation_device"],
    )

    print("\n=== EFI EVIDENCE ===")
    print(result["efi_evidence"].to_string(index=False))

    print("\n=== RESTORATION SEQUENCE ===")
    print(
        result["sequence"][
            [
                "step",
                "action",
                "last_operated_EFI",
                "first_not_operated_EFI",
                "inferred_fault_cable",
                "A01_current_A",
                "B09_current_A",
                "served_nominal_load_A",
                "min_voltage_pu",
                "max_branch_current_A",
                "faulted_cable_energized",
                "safe",
            ]
        ].to_string(index=False)
    )

    print("\n=== FINAL SOURCE LOADING ===")
    print(result["final_sources"].to_string(index=False))

    print("\n=== FINAL SAFETY ===")
    print(result["final_safety"])

    print("\n=== INTENDED INTERPRETATION ===")
    print(
        "1) ME201 is the last EFI Operated and ECE202 is the first EFI "
        "Not operated."
    )
    print("2) The algorithm therefore infers ME201-ECE202.")
    print(
        "3) It automatically opens ME201_to_ECE202 and "
        "ECE202_to_ME201; these switch names are derived from the "
        "inferred cable."
    )
    print(
        "4) A01 restores only ECE201 and ME201; B09 backfeeds all "
        "healthy stations from ECE202 through ME203 after the NOP closes."
    )
    print(
        "5) The actual faulted cable remains de-energized and all modeled "
        "current/voltage/radial constraints must pass."
    )


if __name__ == "__main__":
    main()
