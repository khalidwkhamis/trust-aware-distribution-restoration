"""
ECEPRY Faulty/Spoofed EFI Simulation
Contradictory EFI Pattern Test
===================================

NEW TEST CASE
-------------
True physical Earth Fault: ME202-ECE203

Correct EFI pattern from A01 toward B09:
    ECE201  Operated
    ME201   Operated
    ECE202  Operated
    ME202   Operated   <- last true operated EFI
    ECE203  Not operated <- first true not-operated EFI
    ME203   Not operated
    ECE204  Not operated
    ME204   Not operated
    ECE205  Not operated
    ME205   Not operated

Injected indication error / spoof:
    ME201 is changed from Operated -> Not operated.

Observed EFI pattern becomes:
    ECE201  Operated
    ME201   Not operated
    ECE202  Operated
    ME202   Operated
    ECE203  Not operated
    ...

This is contradictory:
    Operated -> Not operated -> Operated

PROJECT LOGIC BEING TESTED
--------------------------
A clean E/F pattern should be monotonic from A01 toward B09:
    Operated ... Operated | Not operated ... Not operated

If the pattern returns to Operated after a Not operated indication, the EFI
evidence is internally inconsistent.

Therefore:
- A naive EFI-trusting baseline may use the FIRST apparent boundary
  ECE201-ME201 and isolate the wrong cable.
- The trust-aware controller detects the contradiction and blocks energizing
  restoration, entering Safe Hold.

The same observed pattern can represent:
1) non-malicious faulty EFI, or
2) malicious spoofing of the ME201 EFI field.

The physical response is identical; only the cause differs.

MODEL ASSUMPTIONS
-----------------
- 11 kV balanced three-phase steady-state model.
- PF = 0.95.
- Current limit = 350 A.
- Voltage band = 0.94-1.06 pu.
- Synthetic cable sections: 0.5 km, R=0.25 ohm/km, X=0.08 ohm/km.
- A01 and B09 are ideal 1.0 pu sources.
- NOP = ME203 feeder switch toward ECE204.
- All stations have two remotely controlled feeder switches.
"""

import math
from copy import deepcopy
from pathlib import Path
import numpy as np
import pandas as pd
import networkx as nx

# ============================================================
# 1. Parameters
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
# 2. Topology and loads
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
    for d in devices:
        NORMAL_SWITCH_STATE[d] = True

NORMAL_SWITCH_STATE["ME203_to_ECE204"] = False

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
# 4. True EFI generation and corruption
# ============================================================

def generate_true_efi_pattern(true_fault):
    """
    Generate the physically correct clean EFI pattern for an A01-fed E/F.

    For a station-to-station cable fault, the station on the A01 side of the
    fault is the last EFI that should operate.
    """
    if true_fault not in CABLES:
        raise ValueError("True fault must be a modeled cable.")

    left, right = true_fault

    if left == "A01":
        last_operated_index = -1
    else:
        last_operated_index = STATIONS.index(left)

    return {
        station: i <= last_operated_index
        for i, station in enumerate(STATIONS)
    }


def corrupt_false_negative(efi_pattern, station):
    """
    Change one true Operated indication to Not operated.
    This can represent either a faulty EFI or a spoofed EFI value.
    """
    observed = dict(efi_pattern)

    if not observed.get(station, False):
        raise ValueError(
            f"{station} is not Operated in the true pattern."
        )

    observed[station] = False
    return observed

# ============================================================
# 5. EFI logic
# ============================================================

def analyze_efi_pattern(efi_status):
    """
    Analyze whether the EFI pattern is a valid clean monotonic sequence.

    Valid clean sequence:
        Operated ... Operated | Not operated ... Not operated

    Contradiction:
        any Operated indication appearing AFTER a Not operated indication.
    """
    seen_not = False
    contradiction_stations = []

    for station in STATIONS:
        operated = bool(efi_status.get(station, False))

        if not operated:
            seen_not = True
        elif seen_not:
            contradiction_stations.append(station)

    is_consistent = len(contradiction_stations) == 0

    # First apparent boundary for a naive baseline.
    first_not_index = next(
        (i for i, s in enumerate(STATIONS) if not efi_status.get(s, False)),
        None,
    )

    naive_suspected = None
    naive_last_operated = None
    naive_first_not = None

    if first_not_index is not None:
        naive_first_not = STATIONS[first_not_index]

        if first_not_index == 0:
            naive_suspected = ("A01", STATIONS[0])
        else:
            naive_last_operated = STATIONS[first_not_index - 1]
            naive_suspected = (
                naive_last_operated,
                naive_first_not,
            )

    return {
        "consistent_clean_pattern": is_consistent,
        "contradiction_stations": contradiction_stations,
        "naive_suspected_cable": naive_suspected,
        "naive_last_operated": naive_last_operated,
        "naive_first_not_operated": naive_first_not,
    }


def assess_trust(analysis):
    """
    Trust-aware rule:
    Contradictory EFI evidence is Untrusted for energizing transitions.
    """
    if not analysis["consistent_clean_pattern"]:
        return {
            "trust_state": "Untrusted",
            "energizing_allowed": False,
            "reason": (
                "EFI sequence is contradictory: an Operated indication appears "
                "after a Not operated indication. Fault boundary is not trusted."
            ),
        }

    return {
        "trust_state": "Trusted",
        "energizing_allowed": True,
        "reason": "EFI pattern is monotonic and internally consistent.",
    }


def build_evidence_table(true_efi, observed_efi):
    rows = []
    seen_not = False

    for i, station in enumerate(STATIONS, start=1):
        true_state = bool(true_efi[station])
        observed_state = bool(observed_efi[station])

        contradiction_here = False
        if not observed_state:
            seen_not = True
        elif seen_not:
            contradiction_here = True

        if observed_state:
            interpretation = (
                "Observed EFI operated: fault current appears to have passed "
                "this station."
            )
        else:
            interpretation = (
                "Observed EFI not operated: a naive controller may interpret "
                "the fault as upstream of this station."
            )

        rows.append({
            "order_from_A01": i,
            "station": station,
            "true_EFI": "Operated" if true_state else "Not operated",
            "observed_EFI": "Operated" if observed_state else "Not operated",
            "indication_correct": true_state == observed_state,
            "corruption_type": (
                "" if true_state == observed_state else "False negative / spoof"
            ),
            "contradiction_after_not_operated": contradiction_here,
            "interpretation": interpretation,
        })

    return pd.DataFrame(rows)

# ============================================================
# 6. Electrical model
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
        raise ValueError("Energized component is not radial.")

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

        load_current = {}
        for node in component_nodes:
            s_phase = S_LOAD_3PH.get(node, 0j) / 3.0
            load_current[node] = (
                0j if s_phase == 0
                else np.conj(s_phase / voltage[node])
            )

        downstream = {}
        for node in reversed(order):
            total = load_current[node]

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


def evaluate_state(switch_state, true_fault=None):
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
        loads = [n for n in component if n in STATIONS]

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
                    "supplying_source": "INVALID PARALLEL",
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

    fault_energized = False
    if true_fault is not None:
        reverse = (true_fault[1], true_fault[0])

        if true_fault in active or reverse in active:
            start = true_fault[0]
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
        "true_fault_energized": bool(fault_energized),
        "unintended_source_parallel": bool(parallel),
        "actual_safe": bool(
            current_ok
            and voltage_ok
            and not fault_energized
            and not parallel
        ),
    }

    return stations_df, branches_df, sources_df, safety

# ============================================================
# 7. Naive EFI-trusting baseline
# ============================================================

def run_naive_baseline(true_fault, observed_efi):
    analysis = analyze_efi_pattern(observed_efi)
    suspected = analysis["naive_suspected_cable"]

    if suspected is None:
        raise RuntimeError("Naive baseline cannot find first boundary.")

    # IMPORTANT:
    # This baseline intentionally demonstrates the risk of blindly using the
    # first apparent Operated->Not operated boundary and ignoring later
    # contradictory indications.
    upstream_device, downstream_device = EDGE_SWITCHES[suspected]

    state = deepcopy(NORMAL_SWITCH_STATE)
    events = []

    def record(action, basis):
        stations, branches, sources, safety = evaluate_state(
            state,
            true_fault=true_fault,
        )

        current = {
            row["source"]: row["current_A"]
            for _, row in sources.iterrows()
        }

        events.append({
            "step": len(events),
            "controller": "Naive EFI-trusting baseline",
            "action": action,
            "decision_basis": basis,
            "first_apparent_fault_boundary": "-".join(suspected),
            "true_fault_cable": "-".join(true_fault),
            "A01_current_A": current.get("A01", 0.0),
            "B09_current_A": current.get("B09", 0.0),
            "served_nominal_load_A": float(
                stations.loc[
                    stations["energized"],
                    "nominal_load_A",
                ].sum()
            ),
            "true_fault_energized": safety["true_fault_energized"],
            "actual_safe": safety["actual_safe"],
            "unsafe_transition": not safety["actual_safe"],
        })

        return stations, branches, sources, safety

    # A01 trips.
    state["A01_breaker"] = False
    record(
        "A01 trips on Earth Fault",
        "Read observed EFI indications.",
    )

    # Blindly use first apparent boundary ECE201-ME201.
    state[upstream_device] = False
    record(
        f"Open {upstream_device}",
        (
            f"Baseline uses first apparent boundary "
            f"{analysis['naive_last_operated']}=Operated -> "
            f"{analysis['naive_first_not_operated']}=Not operated and "
            f"ignores later contradictory Operated indications."
        ),
    )

    # Reclose A01 because the baseline believes the suspected cable is isolated
    # from the A01 side.
    state["A01_breaker"] = True
    record(
        "Reclose A01 breaker",
        "Restore healthy upstream section according to the wrong inferred boundary.",
    )

    # Open downstream end of the wrong inferred cable.
    state[downstream_device] = False
    record(
        f"Open {downstream_device}",
        "Complete isolation of the wrong inferred cable.",
    )

    # Close NOP. This will backfeed through the TRUE ME202-ECE203 fault.
    state["ME203_to_ECE204"] = True
    _, _, _, after_close = record(
        "Close ME203_to_ECE204 NOP",
        (
            "Baseline assumes the wrong cable is isolated and authorizes B09 "
            "backfeed without resolving the contradictory EFI evidence."
        ),
    )

    # True physical consequence: B09 trips when backfeed reaches actual fault.
    if after_close["true_fault_energized"]:
        state["B09_breaker"] = False
        final = record(
            "B09 trips after backfeed energizes the true fault",
            "Actual ME202-ECE203 fault was energized from B09.",
        )
    else:
        final = evaluate_state(state, true_fault=true_fault)

    return {
        "analysis": analysis,
        "events": pd.DataFrame(events),
        "final_stations": final[0],
        "final_sources": final[2],
        "final_safety": final[3],
    }

# ============================================================
# 8. Trust-aware controller
# ============================================================

def run_trust_aware(true_fault, observed_efi):
    analysis = analyze_efi_pattern(observed_efi)
    trust = assess_trust(analysis)

    state = deepcopy(NORMAL_SWITCH_STATE)
    events = []

    def record(action, basis):
        stations, branches, sources, safety = evaluate_state(
            state,
            true_fault=true_fault,
        )

        current = {
            row["source"]: row["current_A"]
            for _, row in sources.iterrows()
        }

        events.append({
            "step": len(events),
            "controller": "Trust-aware",
            "action": action,
            "decision_basis": basis,
            "trust_state": trust["trust_state"],
            "energizing_allowed": trust["energizing_allowed"],
            "first_apparent_fault_boundary": (
                "-".join(analysis["naive_suspected_cable"])
                if analysis["naive_suspected_cable"] else ""
            ),
            "true_fault_cable": "-".join(true_fault),
            "A01_current_A": current.get("A01", 0.0),
            "B09_current_A": current.get("B09", 0.0),
            "served_nominal_load_A": float(
                stations.loc[
                    stations["energized"],
                    "nominal_load_A",
                ].sum()
            ),
            "true_fault_energized": safety["true_fault_energized"],
            "actual_safe": safety["actual_safe"],
            "unsafe_transition": not safety["actual_safe"],
        })

        return stations, branches, sources, safety

    state["A01_breaker"] = False
    record(
        "A01 trips on Earth Fault",
        "Collect all EFI indications before permitting restoration.",
    )

    # Because evidence is contradictory, do not trust any inferred boundary for
    # energizing restoration.
    record(
        "Enter Safe Hold; block reclosing and NOP closure",
        (
            trust["reason"]
            + " Non-energizing investigation/isolation may continue, but "
              "restoration energization is not authorized."
        ),
    )

    final = evaluate_state(state, true_fault=true_fault)

    return {
        "analysis": analysis,
        "trust": trust,
        "events": pd.DataFrame(events),
        "final_stations": final[0],
        "final_sources": final[2],
        "final_safety": final[3],
    }

# ============================================================
# 9. Main experiment
# ============================================================

def main():
    output = Path("results")
    output.mkdir(exist_ok=True)

    true_fault = ("ME202", "ECE203")

    true_efi = generate_true_efi_pattern(true_fault)

    # NEW corruption point: ME201 false negative.
    # Same observed value could also be maliciously spoofed.
    observed_efi = corrupt_false_negative(true_efi, "ME201")

    analysis = analyze_efi_pattern(observed_efi)
    trust = assess_trust(analysis)
    evidence = build_evidence_table(true_efi, observed_efi)

    baseline = run_naive_baseline(true_fault, observed_efi)
    trust_aware = run_trust_aware(true_fault, observed_efi)

    # Save detailed results.
    evidence.to_csv(
        output / "01_true_vs_observed_efi.csv",
        index=False,
    )
    baseline["events"].to_csv(
        output / "02_naive_baseline_sequence.csv",
        index=False,
    )
    trust_aware["events"].to_csv(
        output / "03_trust_aware_sequence.csv",
        index=False,
    )
    baseline["final_stations"].to_csv(
        output / "04_baseline_final_station_results.csv",
        index=False,
    )
    trust_aware["final_stations"].to_csv(
        output / "05_trust_aware_final_station_results.csv",
        index=False,
    )
    baseline["final_sources"].to_csv(
        output / "06_baseline_final_source_results.csv",
        index=False,
    )
    trust_aware["final_sources"].to_csv(
        output / "07_trust_aware_final_source_results.csv",
        index=False,
    )

    comparison = pd.DataFrame([
        {
            "controller": "Naive EFI-trusting baseline",
            "true_fault": "ME202-ECE203",
            "observed_pattern_consistent": analysis["consistent_clean_pattern"],
            "first_apparent_boundary": "-".join(
                analysis["naive_suspected_cable"]
            ),
            "trust_state": "Not evaluated",
            "energizing_allowed": True,
            "unsafe_transitions": int(
                baseline["events"]["unsafe_transition"].sum()
            ),
            "true_fault_reenergizations": int(
                baseline["events"]["true_fault_energized"].sum()
            ),
            "final_nominal_served_load_A": float(
                baseline["final_stations"].loc[
                    baseline["final_stations"]["energized"],
                    "nominal_load_A",
                ].sum()
            ),
            "outcome": (
                "Uses first false boundary, then B09 backfeed reaches true "
                "ME202-ECE203 fault and B09 trips."
            ),
        },
        {
            "controller": "Trust-aware",
            "true_fault": "ME202-ECE203",
            "observed_pattern_consistent": analysis["consistent_clean_pattern"],
            "first_apparent_boundary": "-".join(
                analysis["naive_suspected_cable"]
            ),
            "trust_state": trust["trust_state"],
            "energizing_allowed": trust["energizing_allowed"],
            "unsafe_transitions": int(
                trust_aware["events"]["unsafe_transition"].sum()
            ),
            "true_fault_reenergizations": int(
                trust_aware["events"]["true_fault_energized"].sum()
            ),
            "final_nominal_served_load_A": float(
                trust_aware["final_stations"].loc[
                    trust_aware["final_stations"]["energized"],
                    "nominal_load_A",
                ].sum()
            ),
            "outcome": (
                "Detects Operated->Not operated->Operated contradiction, "
                "blocks energizing restoration, remains in Safe Hold."
            ),
        },
    ])

    comparison.to_csv(
        output / "00_comparison_summary.csv",
        index=False,
    )

    # Console output.
    print("\n=== NEW FAULTY/SPOOFED EFI TEST ===")
    print("True physical fault: ME202-ECE203")
    print("Injected false negative/spoof at: ME201")
    print("\nTrue EFI pattern:")
    print({k: ("Operated" if v else "Not operated") for k, v in true_efi.items()})
    print("\nObserved EFI pattern:")
    print({k: ("Operated" if v else "Not operated") for k, v in observed_efi.items()})

    print("\nPattern consistent?", analysis["consistent_clean_pattern"])
    print("Contradiction stations:", analysis["contradiction_stations"])
    print("Naive first boundary:", analysis["naive_suspected_cable"])
    print("Trust assessment:", trust["trust_state"], "-", trust["reason"])

    print("\n=== TRUE VS OBSERVED EFI ===")
    print(evidence.to_string(index=False))

    print("\n=== NAIVE BASELINE SEQUENCE ===")
    print(
        baseline["events"][
            [
                "step",
                "action",
                "first_apparent_fault_boundary",
                "true_fault_cable",
                "served_nominal_load_A",
                "true_fault_energized",
                "actual_safe",
                "unsafe_transition",
            ]
        ].to_string(index=False)
    )

    print("\n=== TRUST-AWARE SEQUENCE ===")
    print(
        trust_aware["events"][
            [
                "step",
                "action",
                "trust_state",
                "energizing_allowed",
                "first_apparent_fault_boundary",
                "true_fault_cable",
                "served_nominal_load_A",
                "true_fault_energized",
                "actual_safe",
                "unsafe_transition",
            ]
        ].to_string(index=False)
    )

    print("\n=== COMPARISON ===")
    print(comparison.to_string(index=False))

    print("\n=== INTENDED INTERPRETATION ===")
    print(
        "1) The true ME202-ECE203 fault should produce EFI Operated through ME202, "
        "then Not operated from ECE203 onward."
    )
    print(
        "2) Corrupting ME201 to Not operated creates an impossible clean sequence: "
        "ECE201 Operated -> ME201 Not operated -> ECE202 Operated."
    )
    print(
        "3) A naive controller that uses only the first apparent boundary "
        "mislocalizes the fault as ECE201-ME201."
    )
    print(
        "4) Closing the NOP after isolating that wrong cable backfeeds the actual "
        "ME202-ECE203 fault from B09, creating an unsafe transition."
    )
    print(
        "5) The trust-aware controller recognizes the contradictory EFI evidence "
        "and blocks all energizing restoration transitions, leaving B09's normal "
        "100 A nominal side supplied in Safe Hold."
    )


if __name__ == "__main__":
    main()
