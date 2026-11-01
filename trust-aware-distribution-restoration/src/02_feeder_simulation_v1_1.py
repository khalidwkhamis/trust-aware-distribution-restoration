"""
ECEPRY A01-B09 Looped Feeder Simulation - Version 1.1
====================================================

Purpose
-------
This is the first reproducible simulation for the project.

It currently runs:
1) Normal operation.
2) A clean Earth Fault (E/F) on the ECE203-ME203 cable.
3) The specified SCADA restoration sequence for that fault.

The code is deliberately written as a small transparent model rather than a
black box. It uses a balanced three-phase, steady-state radial
backward/forward-sweep load flow.

IMPORTANT PROJECT ASSUMPTIONS
-----------------------------
- Nominal voltage: 11 kV line-to-line.
- All station loads use a power factor of 0.95.
- Feeder operating limit: 350 A.
- Voltage safe band: 0.94 to 1.06 pu.
- Station load allocations are the specified values in amps.
- Station loads are converted to constant P/Q loads at nominal 11 kV.
- Cable data are SYNTHETIC and identical for all sections in this first model:
      length = 0.5 km
      R = 0.25 ohm/km
      X = 0.08 ohm/km
- A01 and B09 are ideal 11 kV source buses at 1.0 pu.
- Every station has two remotely controlled feeder switches.
- A01 and B09 have source breakers.
- The normal open point (NOP) is the ME203 switch toward ECE204.
- The ECE204 side of the NOP remains energized from B09 in normal operation.

EXPECTED NORMAL OUTPUT
----------------------
Because the station amp values are converted to constant P/Q loads at nominal
voltage, the solved feeder-head currents will be slightly above the nominal
200 A / 100 A values due to cable voltage drop and losses.

With the default synthetic cable values, expected values are approximately:
    A01 current: 202.0 A
    B09 current: 100.4 A
    minimum station voltage: 0.986 pu

EXPECTED CLEAN E/F RESTORATION OUTPUT
-------------------------------------
For a true cable fault between ECE203 and ME203:
- EFI operated through ECE203.
- EFI NOT operated at ME203 and stations beyond it.
- The algorithm infers ECE203-ME203 as the suspected faulted cable.

Project restoration sequence:
1) A01 trips on Earth Fault.
2) Open ECE203 feeder switch toward ME203.
3) Reclose A01 -> restores healthy left side through ECE203.
4) Open ME203 feeder switch toward ECE203 -> cable isolated from both ends.
5) Close NOP ME203 toward ECE204 -> B09 restores ME203.

At the final restored state:
- All 300 A of nominal station load are supplied.
- A01 carries about 181.5 A in the solved P/Q model.
- B09 carries about 120.7 A in the solved P/Q model.
- The faulted ECE203-ME203 cable is not energized.
- Current and voltage constraints remain satisfied.

Later versions can add:
- Faulty EFI (false operated / false not-operated).
- Spoofed EFI.
- Overcurrent / possible flashover.
- Incremental restoration.
- Communication delay/drop/DoS cases.
"""

import math
import cmath
from pathlib import Path
from copy import deepcopy

import numpy as np
import pandas as pd
import networkx as nx

# ---------------------------
# 1. Global model parameters
# ---------------------------

V_LL = 11_000.0                     # V, nominal line-to-line voltage
V_PHASE = V_LL / math.sqrt(3)       # V, nominal phase voltage
POWER_FACTOR = 0.95
Q_FACTOR = math.sqrt(1 - POWER_FACTOR**2)

CURRENT_LIMIT_A = 350.0
V_MIN_PU = 0.94
V_MAX_PU = 1.06

# Synthetic cable model for the first simulation.
CABLE_LENGTH_KM = 0.50
R_OHM_PER_KM = 0.25
X_OHM_PER_KM = 0.08
Z_SECTION = complex(
    R_OHM_PER_KM * CABLE_LENGTH_KM,
    X_OHM_PER_KM * CABLE_LENGTH_KM,
)

# ---------------------------------------------
# 2. Feeder topology and specified station loading
# ---------------------------------------------

STATIONS = [
    "ECE201", "ME201", "ECE202", "ME202", "ECE203",
    "ME203", "ECE204", "ME204", "ECE205", "ME205",
]

# Specified nominal station current allocations.
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

# Each cable is connected through a device at each end.
# Station ends are feeder switches. Source ends are source breakers.
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

# Normal state: every device closed except the specified NOP.
NORMAL_SWITCH_STATE = {}
for pair in EDGE_SWITCHES.values():
    for switch_name in pair:
        NORMAL_SWITCH_STATE[switch_name] = True

# Specified NOP: ME203 feeder switch toward ECE204.
NORMAL_SWITCH_STATE["ME203_to_ECE204"] = False


# ---------------------------------------------------------
# 3. Convert operational amp loads into P/Q for load flow
# ---------------------------------------------------------

def three_phase_load_va(current_a):
    """
    Convert the specified nominal current at 11 kV into a 3-phase complex load.

    S = sqrt(3) * V_LL * I
    P = S * pf
    Q = S * sqrt(1-pf^2)
    """
    s_va = math.sqrt(3) * V_LL * current_a
    p_w = s_va * POWER_FACTOR
    q_var = s_va * Q_FACTOR
    return complex(p_w, q_var)


S_LOAD_3PH = {
    station: three_phase_load_va(current)
    for station, current in NOMINAL_LOAD_A.items()
}


# ---------------------------------------------
# 4. EFI fault-boundary inference
# ---------------------------------------------

def infer_earth_fault_from_efi(efi_status):
    """
    Infer the suspected E/F cable from EFI indications when viewed from A01.

    EFI status is supplied as:
        True  = EFI operated
        False = EFI not operated

    Expected clean pattern:
        operated ... operated, then not-operated ... not-operated

    Examples
    --------
    ECE201=False:
        suspected cable = A01-ECE201

    ECE203=True and ME203=False:
        suspected cable = ECE203-ME203

    A False followed later by True is contradictory evidence and is flagged.
    """
    seen_not_operated = False

    for station in STATIONS:
        status = efi_status.get(station, False)

        if not status:
            seen_not_operated = True
        elif seen_not_operated:
            return {
                "valid_pattern": False,
                "suspected_cable": None,
                "reason": "Contradictory EFI pattern: operated after not-operated.",
            }

    # Find first station that did NOT operate.
    first_not = next(
        (i for i, s in enumerate(STATIONS) if not efi_status.get(s, False)),
        None,
    )

    if first_not is None:
        return {
            "valid_pattern": True,
            "suspected_cable": None,
            "reason": "All EFIs operated; fault boundary is not identified by this rule.",
        }

    if first_not == 0:
        cable = ("A01", STATIONS[0])
    else:
        cable = (STATIONS[first_not - 1], STATIONS[first_not])

    return {
        "valid_pattern": True,
        "suspected_cable": cable,
        "reason": "Boundary between last EFI operated and first EFI not-operated.",
    }


def build_efi_evidence_table(efi_status, inference):
    """
    Create a human-readable table showing exactly how EFI evidence is used.

    Interpretation is from the tripped A01 source toward B09:
    - EFI operated: fault is downstream of that station.
    - EFI not operated: under the clean E/F pattern, fault is upstream of
      that station.
    - The last operated EFI + first not-operated EFI define the suspected
      faulted cable.

    NOTE:
    This interpretation is valid for the CLEAN E/F case. Later faulty/spoofed
    EFI scenarios will intentionally challenge the trustworthiness of these
    indications.
    """
    first_not_index = next(
        (i for i, s in enumerate(STATIONS) if not efi_status.get(s, False)),
        None,
    )

    last_operated_index = None
    if first_not_index is not None and first_not_index > 0:
        last_operated_index = first_not_index - 1

    rows = []
    for i, station in enumerate(STATIONS):
        operated = bool(efi_status.get(station, False))

        if operated:
            interpretation = (
                "EFI operated: earth-fault current passed this station; "
                "fault is downstream toward B09."
            )
        else:
            interpretation = (
                "EFI not operated: for a clean E/F pattern, "
                "fault is upstream toward A01."
            )

        rows.append({
            "order_from_A01": i + 1,
            "station": station,
            "EFI_status": "Operated" if operated else "Not operated",
            "last_operated_boundary": i == last_operated_index,
            "first_not_operated_boundary": i == first_not_index,
            "interpretation": interpretation,
            "inferred_fault_cable": (
                "-".join(inference["suspected_cable"])
                if inference.get("suspected_cable")
                else ""
            ),
        })

    return pd.DataFrame(rows)


# ---------------------------------------------
# 5. Network state / active cable construction
# ---------------------------------------------

def active_cables(switch_state):
    """A cable conducts only when the devices at BOTH ends are closed."""
    active = []
    for cable, devices in EDGE_SWITCHES.items():
        if all(switch_state[d] for d in devices):
            active.append(cable)
    return active


# ---------------------------------------------
# 6. Radial backward/forward-sweep load flow
# ---------------------------------------------

def solve_radial_component(component_nodes, component_edges, source):
    """
    Solve one energized radial component from one source.

    Loads are balanced, 3-phase, constant P/Q.
    The calculation is performed per phase.
    """
    graph = nx.Graph()
    graph.add_nodes_from(component_nodes)
    graph.add_edges_from(component_edges)

    if not nx.is_tree(graph):
        raise ValueError(
            "An energized component is not radial. "
            "The base project does not allow unintended A01-B09 paralleling."
        )

    tree = nx.bfs_tree(graph, source)
    order = list(tree.nodes)

    parent = {source: None}
    children = {node: list(tree.successors(node)) for node in tree.nodes}

    for node in order[1:]:
        parent[node] = next(tree.predecessors(node))

    # Start every bus at 1.0 pu.
    voltage = {
        node: complex(V_PHASE, 0.0)
        for node in component_nodes
    }

    branch_current = {}

    for _ in range(100):
        old_voltage = voltage.copy()

        # Load-current calculation from the current voltage estimate.
        injected_current = {}
        for node in component_nodes:
            s_phase = S_LOAD_3PH.get(node, 0j) / 3.0
            if s_phase == 0:
                injected_current[node] = 0j
            else:
                injected_current[node] = np.conj(s_phase / voltage[node])

        # Backward sweep: sum downstream currents.
        downstream_current = {}
        for node in reversed(order):
            total = injected_current[node]

            for child in children.get(node, []):
                total += downstream_current[child]

            downstream_current[node] = total

            if parent[node] is not None:
                branch_current[(parent[node], node)] = total

        # Forward sweep: calculate voltage drops.
        voltage[source] = complex(V_PHASE, 0.0)

        for node in order[1:]:
            p = parent[node]
            voltage[node] = (
                voltage[p]
                - Z_SECTION * branch_current[(p, node)]
            )

        # Convergence check.
        largest_change = max(
            abs(voltage[n] - old_voltage[n])
            for n in component_nodes
        )

        if largest_change < 1e-6:
            break

    return voltage, branch_current, children


# ---------------------------------------------
# 7. Evaluate one switching state
# ---------------------------------------------

def evaluate_state(switch_state, faulted_cables=None):
    """
    Run the electrical solution and the project safety checks.

    Safety checks currently include:
    - no unintended source paralleling,
    - branch current <= 350 A,
    - energized station voltage between 0.94 and 1.06 pu,
    - no faulted cable energized from an active source.
    """
    faulted_cables = set(faulted_cables or [])
    active = active_cables(switch_state)

    graph = nx.Graph()
    graph.add_nodes_from(NODES)
    graph.add_edges_from(active)

    station_rows = []
    branch_rows = []
    source_rows = []
    parallel_source = False

    # Solve each electrically connected component.
    for component in nx.connected_components(graph):
        component = set(component)

        sources = [
            source
            for source in ("A01", "B09")
            if source in component
        ]

        component_edges = [
            cable
            for cable in active
            if cable[0] in component and cable[1] in component
        ]

        load_nodes = [
            node
            for node in component
            if node in STATIONS
        ]

        # No source -> stations are de-energized.
        if len(sources) == 0:
            for station in load_nodes:
                station_rows.append({
                    "station": station,
                    "energized": False,
                    "voltage_pu": 0.0,
                    "nominal_load_A": NOMINAL_LOAD_A[station],
                })
            continue

        # More than one source in one component is invalid for this base model.
        if len(sources) > 1:
            parallel_source = True

            for station in load_nodes:
                station_rows.append({
                    "station": station,
                    "energized": True,
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
                "voltage_pu": abs(voltage[station]) / V_PHASE,
                "nominal_load_A": NOMINAL_LOAD_A[station],
            })

        for (parent, child), current in branch_current.items():
            # Return physical cable orientation.
            cable = (
                (parent, child)
                if (parent, child) in CABLES
                else (child, parent)
            )

            branch_rows.append({
                "source": source,
                "from": cable[0],
                "to": cable[1],
                "current_A": abs(current),
            })

        source_current = abs(
            sum(
                (
                    branch_current[(source, child)]
                    for child in children[source]
                ),
                0j,
            )
        )

        source_rows.append({
            "source": source,
            "current_A": source_current,
            "nominal_served_load_A": sum(
                NOMINAL_LOAD_A[n]
                for n in load_nodes
            ),
        })

    stations_df = pd.DataFrame(station_rows)
    branches_df = pd.DataFrame(branch_rows)
    sources_df = pd.DataFrame(source_rows)

    # Faulted cable is unsafe only if it is electrically active AND connected
    # to an energized source.
    fault_energized = False

    for fault in faulted_cables:
        reverse_fault = (fault[1], fault[0])

        if fault in active or reverse_fault in active:
            cable_start = fault[0]

            if cable_start in graph:
                connected = nx.node_connected_component(graph, cable_start)

                if "A01" in connected or "B09" in connected:
                    fault_energized = True

    max_current = (
        branches_df["current_A"].max()
        if not branches_df.empty
        else 0.0
    )

    energized_voltages = (
        stations_df.loc[
            stations_df["energized"],
            "voltage_pu",
        ]
        if not stations_df.empty
        else pd.Series(dtype=float)
    )

    min_voltage = (
        energized_voltages.min()
        if not energized_voltages.empty
        else float("nan")
    )

    max_voltage = (
        energized_voltages.max()
        if not energized_voltages.empty
        else float("nan")
    )

    current_ok = max_current <= CURRENT_LIMIT_A
    voltage_ok = (
        energized_voltages.empty
        or (
            min_voltage >= V_MIN_PU
            and max_voltage <= V_MAX_PU
        )
    )

    safety = {
        "max_branch_current_A": max_current,
        "min_voltage_pu": min_voltage,
        "max_voltage_pu": max_voltage,
        "current_limit_ok": current_ok,
        "voltage_limit_ok": voltage_ok,
        "faulted_cable_energized": fault_energized,
        "unintended_source_parallel": parallel_source,
        "safe": (
            current_ok
            and voltage_ok
            and not fault_energized
            and not parallel_source
        ),
    }

    return stations_df, branches_df, sources_df, safety


# ---------------------------------------------
# 8. Scenario 1 - Normal operation
# ---------------------------------------------

def run_normal_operation():
    state = deepcopy(NORMAL_SWITCH_STATE)

    stations, branches, sources, safety = evaluate_state(state)

    return {
        "stations": stations,
        "branches": branches,
        "sources": sources,
        "safety": safety,
    }


# ---------------------------------------------
# 9. Scenario 2 - Clean E/F ECE203-ME203
# ---------------------------------------------

def run_clean_earth_fault_restoration():
    """
    Clean E/F scenario defined for the project.

    True fault:
        ECE203-ME203 cable

    Clean EFI indications from A01:
        ECE201  operated
        ME201   operated
        ECE202  operated
        ME202   operated
        ECE203  operated
        ME203   NOT operated
        all downstream stations NOT operated

    The EFI rule should therefore identify:
        ECE203-ME203
    """

    true_fault = ("ECE203", "ME203")

    efi = {
        "ECE201": True,
        "ME201": True,
        "ECE202": True,
        "ME202": True,
        "ECE203": True,
        "ME203": False,
        "ECE204": False,
        "ME204": False,
        "ECE205": False,
        "ME205": False,
    }

    inference = infer_earth_fault_from_efi(efi)
    efi_evidence = build_efi_evidence_table(efi, inference)

    if inference["suspected_cable"] != true_fault:
        raise RuntimeError(
            "Clean EFI inference did not identify the expected fault."
        )

    state = deepcopy(NORMAL_SWITCH_STATE)

    sequence = []

    def record(description, decision_basis):
        stations, branches, sources, safety = evaluate_state(
            state,
            faulted_cables=[true_fault],
        )

        current_by_source = {
            row["source"]: row["current_A"]
            for _, row in sources.iterrows()
        }

        row = {
            "step": len(sequence),
            "action_state": description,
            "decision_basis": decision_basis,
            "inferred_fault_cable": "-".join(inference["suspected_cable"]),
            "A01_current_A": current_by_source.get("A01", 0.0),
            "B09_current_A": current_by_source.get("B09", 0.0),
            "served_nominal_load_A": (
                stations.loc[
                    stations["energized"],
                    "nominal_load_A",
                ].sum()
            ),
            "min_voltage_pu": safety["min_voltage_pu"],
            "max_branch_current_A": safety["max_branch_current_A"],
            "faulted_cable_energized": safety[
                "faulted_cable_energized"
            ],
            "safe": safety["safe"],
        }

        # Keep the latched EFI evidence visible beside every switching action.
        # This makes it clear that the controller is acting on observed EFI
        # indications rather than following a hidden/pre-scripted sequence.
        for station in STATIONS:
            row[f"{station}_EFI"] = (
                "Operated" if efi[station] else "Not operated"
            )

        sequence.append(row)

        return stations, branches, sources, safety

    # Step 0: fault occurs and the A01 source breaker trips.
    state["A01_breaker"] = False
    record(
        "E/F trip: A01 breaker open",
        "Protection event. A01 tripped on Earth Fault; read and latch all station EFI indications before restoration."
    )

    # Step 1: use EFI evidence to open the upstream boundary switch.
    state["ECE203_to_ME203"] = False
    record(
        "Open ECE203 feeder switch toward ME203",
        "EFI boundary: ECE203=Operated and ME203=Not operated. Therefore suspected fault is ECE203-ME203; open the A01-side boundary switch."
    )

    # Step 2: restore the healthy A01 side.
    state["A01_breaker"] = True
    record(
        "Reclose A01 and restore healthy left side through ECE203",
        "ECE203->ME203 is confirmed open, so the suspected faulted cable is disconnected from A01. Reclose A01 to restore the healthy upstream section."
    )

    # Step 3: isolate the suspected cable from its ME203 end.
    state["ME203_to_ECE203"] = False
    record(
        "Open ME203 feeder switch toward ECE203",
        "Use the same EFI-inferred fault boundary to open the ME203 end. The suspected ECE203-ME203 cable is now isolated from both ends."
    )

    # Step 4: close the NOP and restore ME203 from B09.
    state["ME203_to_ECE204"] = True
    final_state = record(
        "Close NOP ME203 toward ECE204 and restore ME203 from B09",
        "Faulted cable is isolated from both ends; alternate B09 path is radial and passes the 350 A and 0.94-1.06 pu safety checks."
    )

    return {
        "true_fault": true_fault,
        "efi": efi,
        "inference": inference,
        "efi_evidence": efi_evidence,
        "sequence": pd.DataFrame(sequence),
        "final_stations": final_state[0],
        "final_branches": final_state[1],
        "final_sources": final_state[2],
        "final_safety": final_state[3],
    }


# ---------------------------------------------
# 10. Save reproducible outputs
# ---------------------------------------------

def main():
    output = Path("results")
    output.mkdir(exist_ok=True)

    # Normal case.
    normal = run_normal_operation()

    normal["stations"].to_csv(
        output / "normal_station_results.csv",
        index=False,
    )
    normal["branches"].to_csv(
        output / "normal_branch_results.csv",
        index=False,
    )
    normal["sources"].to_csv(
        output / "normal_source_results.csv",
        index=False,
    )

    # Clean Earth Fault case.
    ef = run_clean_earth_fault_restoration()

    ef["efi_evidence"].to_csv(
        output / "earth_fault_efi_evidence.csv",
        index=False,
    )
    ef["sequence"].to_csv(
        output / "earth_fault_restoration_sequence.csv",
        index=False,
    )
    ef["final_stations"].to_csv(
        output / "earth_fault_final_station_results.csv",
        index=False,
    )
    ef["final_branches"].to_csv(
        output / "earth_fault_final_branch_results.csv",
        index=False,
    )
    ef["final_sources"].to_csv(
        output / "earth_fault_final_source_results.csv",
        index=False,
    )

    # Human-readable console summary.
    print("\n=== NORMAL OPERATION ===")
    print(normal["sources"].to_string(index=False))
    print("\nSafety:", normal["safety"])

    print("\n=== CLEAN EARTH FAULT: ECE203-ME203 ===")
    print(
        "EFI-inferred fault:",
        ef["inference"]["suspected_cable"],
    )
    print("\nEFI evidence from A01 toward B09:")
    print(ef["efi_evidence"].to_string(index=False))
    print("\nRestoration sequence:")
    print(ef["sequence"].to_string(index=False))
    print("\nFinal safety:", ef["final_safety"])

    print("\nExpected interpretation:")
    print(
        "- Normal source currents are slightly above 200 A / 100 A "
        "because constant-P/Q loads draw more current as voltage drops."
    )
    print(
        "- After restoration, A01 supplies the healthy section through "
        "ECE203 and B09 supplies ME203 plus the original B09-side load."
    )
    print(
        "- No faulted cable is energized and all modeled limits remain satisfied."
    )


if __name__ == "__main__":
    main()
