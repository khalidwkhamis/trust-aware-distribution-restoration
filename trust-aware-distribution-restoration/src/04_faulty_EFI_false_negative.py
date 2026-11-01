"""
ECEPRY Faulty-EFI Simulation
False Negative at ECE201 for a True ECE201-ME201 Cable Fault
=============================================================

PURPOSE
-------
This simulation compares two controllers under the SAME physical Earth Fault:

A) EFI-trusting baseline controller
B) Trust-aware controller

TRUE PHYSICAL FAULT
-------------------
ECE201-ME201

TRUE EFI PATTERN
----------------
ECE201 : Operated
ME201  : Not operated
ECE202 : Not operated
... all stations farther downstream: Not operated

Therefore, with correct EFI information, the fault boundary would be:
    ECE201 (Operated) | ME201 (Not operated)
and the true suspected cable would be ECE201-ME201.

FAULTY EFI
----------
ECE201 suffers a FALSE NEGATIVE:
    Actual/true EFI at ECE201   = Operated
    Reported/observed EFI       = Not operated

The observed controller data therefore become:
ECE201 : Not operated
ME201  : Not operated
ECE202 : Not operated
... all downstream: Not operated

If a controller blindly trusts that observed pattern, it infers:
    A01-ECE201

That is the wrong cable.

WHY THIS CASE MATTERS
---------------------
The baseline controller can believe it has isolated A01-ECE201 and can safely
close the normal open point (NOP). Its perceived electrical checks may pass.

However, the TRUE fault is ECE201-ME201. Closing the NOP from B09 then
backfeeds through the actual faulted cable.

The trust-aware controller uses an evidence-quality rule:
    An A01 Earth-Fault trip with NO positive EFI-operated indication on the
    A01-fed section is insufficient evidence for an energizing restoration
    transition.

Therefore:
- the observed localization may still be recorded,
- but trust state becomes UNTRUSTED for energizing restoration,
- closing the NOP is blocked,
- the system enters Safe Hold,
- B09 continues serving its normal 100 A side,
- the true fault is not re-energized.

This is intentionally conservative. It demonstrates the project's principle:
unsafe restoration must not be authorized when critical localization evidence
is insufficiently trustworthy.

MODEL ASSUMPTIONS
-----------------
- Balanced 11 kV, 3-phase steady-state model.
- Power factor = 0.95 for all station loads.
- Current limit = 350 A.
- Voltage band = 0.94-1.06 pu.
- Synthetic identical cable sections:
      length = 0.5 km
      R = 0.25 ohm/km
      X = 0.08 ohm/km
- A01 and B09 are ideal 1.0 pu sources.
- Every station has two remotely controlled feeder switches.
- NOP = ME203 feeder switch toward ECE204.
"""

import math
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import networkx as nx


# ============================================================
# 1. Global parameters
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
# 2. Feeder topology and station loads
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
# 4. Generate correct EFI pattern from a true fault
# ============================================================

def generate_true_efi_pattern(true_fault):
    """
    Generate the clean EFI pattern from the physical fault location.

    For an A01-fed Earth Fault:
    - Stations upstream of the fault boundary see earth-fault current -> Operated.
    - The first station downstream of the fault and stations beyond it -> Not operated.

    Example true fault ECE201-ME201:
        ECE201 = Operated
        ME201 onward = Not operated
    """
    left, right = true_fault

    if true_fault not in CABLES:
        raise ValueError("True fault must be one of the modeled feeder cables.")

    # For a station-to-station cable, the left station is the last operated EFI.
    # For A01-ECE201, no station EFI operates before the fault.
    if left == "A01":
        last_operated_index = -1
    else:
        last_operated_index = STATIONS.index(left)

    return {
        station: i <= last_operated_index
        for i, station in enumerate(STATIONS)
    }


def apply_false_negative(efi_pattern, station):
    """Return a copy with one EFI changed from Operated to Not operated."""
    corrupted = dict(efi_pattern)

    if not corrupted.get(station, False):
        raise ValueError(
            f"{station} is not Operated in the true pattern, so it cannot "
            "be used for this false-negative test."
        )

    corrupted[station] = False
    return corrupted


# ============================================================
# 5. EFI boundary inference
# ============================================================

def infer_earth_fault_from_efi(efi_status):
    """
    Infer a clean E/F boundary from the OBSERVED EFI states.

    This function deliberately does not know whether an indication is faulty.
    It only reports what a conventional EFI-trusting controller would infer.
    """
    seen_not = False

    for station in STATIONS:
        operated = bool(efi_status.get(station, False))

        if not operated:
            seen_not = True
        elif seen_not:
            return {
                "valid_pattern": False,
                "suspected_cable": None,
                "last_operated_station": None,
                "first_not_operated_station": None,
                "reason": "Contradictory EFI sequence.",
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
            "reason": "All EFIs operated; no downstream boundary identified.",
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
            "Observed boundary between last EFI Operated and first EFI "
            "Not operated."
        ),
    }


# ============================================================
# 6. Trust assessment for Earth-Fault localization
# ============================================================

def assess_efi_trust(observed_efi, inference):
    """
    Project-specific trust rule used in this simulation.

    If A01 trips on Earth Fault but NONE of the station EFIs report Operated,
    the controller has no positive EFI evidence that fault current passed a
    station.

    The observed pattern could represent a genuine first-cable fault, but it
    could also result from a false-negative/spoofed upstream EFI.

    Therefore the localization is not trusted for an ENERGIZING transition.

    Non-energizing actions and Safe Hold remain allowed.
    """
    operated_stations = [
        station
        for station in STATIONS
        if observed_efi.get(station, False)
    ]

    if not inference["valid_pattern"]:
        return {
            "trust_state": "Untrusted",
            "energizing_allowed": False,
            "reason": "Contradictory EFI pattern.",
        }

    if len(operated_stations) == 0:
        return {
            "trust_state": "Untrusted",
            "energizing_allowed": False,
            "reason": (
                "A01 Earth-Fault trip has no positive EFI-operated evidence. "
                "Observed A01-ECE201 localization depends entirely on "
                "not-operated indications, so energizing restoration is blocked."
            ),
        }

    return {
        "trust_state": "Trusted",
        "energizing_allowed": True,
        "reason": "Positive EFI-operated evidence supports the fault boundary.",
    }


# ============================================================
# 7. Evidence table
# ============================================================

def build_true_vs_observed_table(true_efi, observed_efi):
    rows = []

    for i, station in enumerate(STATIONS, start=1):
        true_state = bool(true_efi[station])
        observed_state = bool(observed_efi[station])

        rows.append({
            "order_from_A01": i,
            "station": station,
            "true_EFI": "Operated" if true_state else "Not operated",
            "observed_EFI": "Operated" if observed_state else "Not operated",
            "indication_correct": true_state == observed_state,
            "fault_type_if_incorrect": (
                "" if true_state == observed_state else "False negative"
            ),
        })

    return pd.DataFrame(rows)


# ============================================================
# 8. Electrical model
# ============================================================

def active_cables(switch_state):
    active = []

    for cable, devices in EDGE_SWITCHES.items():
        if all(switch_state[d] for d in devices):
            active.append(cable)

    return active


def solve_radial_component(component_nodes, component_edges, source):
    graph = nx.Graph()
    graph.add_nodes_from(component_nodes)
    graph.add_edges_from(component_edges)

    if not nx.is_tree(graph):
        raise ValueError("Energized component is not radial.")

    tree = nx.bfs_tree(graph, source)
    order = list(tree.nodes)

    parent = {source: None}
    children = {node: list(tree.successors(node)) for node in tree.nodes}

    for node in order[1:]:
        parent[node] = next(tree.predecessors(node))

    voltage = {node: complex(V_PHASE, 0.0) for node in component_nodes}
    branch_current = {}

    for _ in range(100):
        old = voltage.copy()

        injected = {}
        for node in component_nodes:
            s_phase = S_LOAD_3PH.get(node, 0j) / 3.0

            if s_phase == 0:
                injected[node] = 0j
            else:
                injected[node] = np.conj(s_phase / voltage[node])

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


def evaluate_state(switch_state, physical_fault=None):
    """
    Evaluate the actual electrical state.

    physical_fault is the TRUE cable fault used by the experiment evaluator.
    The controller may believe a different cable is faulted.
    """
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
            parallel_source = True

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
            component_edges,
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

    actual_fault_energized = False

    if physical_fault is not None:
        reverse = (physical_fault[1], physical_fault[0])

        if physical_fault in active or reverse in active:
            start = physical_fault[0]

            if start in graph:
                connected = nx.node_connected_component(graph, start)

                if "A01" in connected or "B09" in connected:
                    actual_fault_energized = True

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
        "actual_faulted_cable_energized": bool(actual_fault_energized),
        "unintended_source_parallel": bool(parallel_source),
        "actual_safe": bool(
            current_ok
            and voltage_ok
            and not actual_fault_energized
            and not parallel_source
        ),
    }

    return stations_df, branches_df, sources_df, safety


def controller_perceived_check(switch_state, perceived_fault):
    """
    What a controller believes from its inferred fault location.

    This uses the same electrical/current/voltage/radial checks, but checks
    fault isolation against the PERCEIVED fault instead of the true fault.
    """
    _, _, _, perceived = evaluate_state(
        switch_state,
        physical_fault=perceived_fault,
    )

    return {
        "perceived_max_branch_current_A": perceived["max_branch_current_A"],
        "perceived_min_voltage_pu": perceived["min_voltage_pu"],
        "perceived_current_ok": perceived["current_limit_ok"],
        "perceived_voltage_ok": perceived["voltage_limit_ok"],
        "perceived_fault_energized": perceived[
            "actual_faulted_cable_energized"
        ],
        "perceived_parallel": perceived["unintended_source_parallel"],
        "perceived_safe": perceived["actual_safe"],
    }


# ============================================================
# 9. Baseline controller
# ============================================================

def run_baseline(true_fault, observed_efi):
    inference = infer_earth_fault_from_efi(observed_efi)
    suspected = inference["suspected_cable"]

    if suspected is None:
        raise RuntimeError("Baseline could not infer a cable.")

    state = deepcopy(NORMAL_SWITCH_STATE)
    events = []

    def record(action, decision_basis, perceived_check=None):
        stations, branches, sources, actual = evaluate_state(
            state,
            physical_fault=true_fault,
        )

        source_current = {
            row["source"]: row["current_A"]
            for _, row in sources.iterrows()
        }

        row = {
            "step": len(events),
            "controller": "EFI-trusting baseline",
            "action": action,
            "decision_basis": decision_basis,
            "trust_state": "Assumed trusted",
            "inferred_fault_cable": "-".join(suspected),
            "true_fault_cable": "-".join(true_fault),
            "A01_current_A": source_current.get("A01", 0.0),
            "B09_current_A": source_current.get("B09", 0.0),
            "served_nominal_load_A": float(
                stations.loc[
                    stations["energized"],
                    "nominal_load_A",
                ].sum()
            ),
            "actual_faulted_cable_energized": actual[
                "actual_faulted_cable_energized"
            ],
            "actual_safe": actual["actual_safe"],
            "unsafe_transition": not actual["actual_safe"],
        }

        if perceived_check:
            row.update(perceived_check)

        events.append(row)
        return stations, branches, sources, actual

    # A01 protection trips.
    state["A01_breaker"] = False
    record(
        "A01 trips on Earth Fault",
        "Protection trip. Baseline accepts observed EFI indications as valid.",
    )

    # Baseline uses observed all-not-operated EFI and mislocalizes A01-ECE201.
    upstream_device, downstream_device = EDGE_SWITCHES[suspected]

    # A01 breaker is already open for this first-cable inferred fault.
    # Open the ECE201 side of the perceived cable.
    state[downstream_device] = False
    record(
        f"Open {downstream_device}",
        (
            "Observed ECE201 EFI is Not operated, so baseline infers "
            "A01-ECE201 and isolates the ECE201 end of that perceived cable."
        ),
    )

    # Candidate NOP close: baseline checks only what it believes.
    candidate = deepcopy(state)
    candidate["ME203_to_ECE204"] = True
    perceived = controller_perceived_check(candidate, suspected)

    # Because perceived electrical/safety checks pass, baseline authorizes close.
    if perceived["perceived_safe"]:
        state["ME203_to_ECE204"] = True

    _, _, _, after_close = record(
        "Close ME203_to_ECE204 NOP",
        (
            "Baseline believes A01-ECE201 is isolated. Its perceived current, "
            "voltage, radiality, and perceived-fault checks pass, so it "
            "authorizes B09 backfeed."
        ),
        perceived_check=perceived,
    )

    # Physical consequence: true ECE201-ME201 fault is now energized from B09.
    # Simulate protective trip of B09 after this unsafe transition.
    if after_close["actual_faulted_cable_energized"]:
        state["B09_breaker"] = False
        final = record(
            "B09 trips on Earth Fault after backfeed reaches true fault",
            (
                "The NOP closure energized the actual ECE201-ME201 fault. "
                "B09 protection trips, ending the unsafe backfeed."
            ),
        )
    else:
        final = evaluate_state(state, physical_fault=true_fault)

    return {
        "inference": inference,
        "events": pd.DataFrame(events),
        "final_stations": final[0],
        "final_branches": final[1],
        "final_sources": final[2],
        "final_safety": final[3],
    }


# ============================================================
# 10. Trust-aware controller
# ============================================================

def run_trust_aware(true_fault, observed_efi):
    inference = infer_earth_fault_from_efi(observed_efi)
    trust = assess_efi_trust(observed_efi, inference)

    state = deepcopy(NORMAL_SWITCH_STATE)
    events = []

    def record(action, decision_basis):
        stations, branches, sources, actual = evaluate_state(
            state,
            physical_fault=true_fault,
        )

        source_current = {
            row["source"]: row["current_A"]
            for _, row in sources.iterrows()
        }

        events.append({
            "step": len(events),
            "controller": "Trust-aware",
            "action": action,
            "decision_basis": decision_basis,
            "trust_state": trust["trust_state"],
            "energizing_allowed": trust["energizing_allowed"],
            "inferred_fault_cable": (
                "-".join(inference["suspected_cable"])
                if inference["suspected_cable"]
                else ""
            ),
            "true_fault_cable": "-".join(true_fault),
            "A01_current_A": source_current.get("A01", 0.0),
            "B09_current_A": source_current.get("B09", 0.0),
            "served_nominal_load_A": float(
                stations.loc[
                    stations["energized"],
                    "nominal_load_A",
                ].sum()
            ),
            "actual_faulted_cable_energized": actual[
                "actual_faulted_cable_energized"
            ],
            "actual_safe": actual["actual_safe"],
            "unsafe_transition": not actual["actual_safe"],
        })

        return stations, branches, sources, actual

    # Protection trip.
    state["A01_breaker"] = False
    record(
        "A01 trips on Earth Fault",
        "Protection trip triggers EFI collection and trust assessment.",
    )

    # The localization is recorded, but not trusted for energization.
    record(
        "Enter Safe Hold; block NOP closure",
        (
            trust["reason"]
            + " The controller keeps A01 open and keeps the NOP open. "
              "B09 continues supplying its normal healthy side."
        ),
    )

    final = evaluate_state(state, physical_fault=true_fault)

    return {
        "inference": inference,
        "trust": trust,
        "events": pd.DataFrame(events),
        "final_stations": final[0],
        "final_branches": final[1],
        "final_sources": final[2],
        "final_safety": final[3],
    }


# ============================================================
# 11. Main experiment
# ============================================================

def main():
    output = Path("results")
    output.mkdir(exist_ok=True)

    true_fault = ("ECE201", "ME201")

    # Generate the physically correct EFI pattern from the actual cable fault.
    true_efi = generate_true_efi_pattern(true_fault)

    # Inject one non-malicious sensor/protection indication failure:
    # ECE201 should be Operated but falsely reports Not operated.
    observed_efi = apply_false_negative(true_efi, "ECE201")

    inference = infer_earth_fault_from_efi(observed_efi)
    trust = assess_efi_trust(observed_efi, inference)

    evidence = build_true_vs_observed_table(true_efi, observed_efi)

    baseline = run_baseline(true_fault, observed_efi)
    trust_aware = run_trust_aware(true_fault, observed_efi)

    # ---------------------------
    # Save detailed CSV outputs
    # ---------------------------

    evidence.to_csv(
        output / "01_true_vs_observed_efi.csv",
        index=False,
    )

    baseline["events"].to_csv(
        output / "02_baseline_sequence.csv",
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

    # ---------------------------
    # Comparison summary
    # ---------------------------

    baseline_unsafe = int(
        baseline["events"]["unsafe_transition"].sum()
    )
    trust_unsafe = int(
        trust_aware["events"]["unsafe_transition"].sum()
    )

    baseline_reenerg = int(
        baseline["events"]["actual_faulted_cable_energized"].sum()
    )
    trust_reenerg = int(
        trust_aware["events"]["actual_faulted_cable_energized"].sum()
    )

    def final_served(stations):
        return float(
            stations.loc[
                stations["energized"],
                "nominal_load_A",
            ].sum()
        )

    comparison = pd.DataFrame([
        {
            "controller": "EFI-trusting baseline",
            "true_fault": "ECE201-ME201",
            "observed_inferred_fault": "-".join(
                baseline["inference"]["suspected_cable"]
            ),
            "trust_result": "Observed EFI blindly accepted",
            "energizing_restoration_allowed": True,
            "unsafe_transitions": baseline_unsafe,
            "faulted_section_reenergizations": baseline_reenerg,
            "final_nominal_served_load_A": final_served(
                baseline["final_stations"]
            ),
            "final_actual_safe": baseline["final_safety"]["actual_safe"],
            "outcome": (
                "Wrong localization -> NOP close backfeeds true fault -> "
                "B09 trips."
            ),
        },
        {
            "controller": "Trust-aware",
            "true_fault": "ECE201-ME201",
            "observed_inferred_fault": "-".join(
                trust_aware["inference"]["suspected_cable"]
            ),
            "trust_result": trust_aware["trust"]["trust_state"],
            "energizing_restoration_allowed": trust_aware["trust"][
                "energizing_allowed"
            ],
            "unsafe_transitions": trust_unsafe,
            "faulted_section_reenergizations": trust_reenerg,
            "final_nominal_served_load_A": final_served(
                trust_aware["final_stations"]
            ),
            "final_actual_safe": trust_aware["final_safety"]["actual_safe"],
            "outcome": (
                "Insufficient EFI evidence -> energizing transition blocked -> "
                "Safe Hold; B09 normal side remains supplied."
            ),
        },
    ])

    comparison.to_csv(
        output / "00_comparison_summary.csv",
        index=False,
    )

    # ---------------------------
    # Console output
    # ---------------------------

    print("\n=== FAULTY EFI FALSE-NEGATIVE TEST ===")
    print("True physical fault: ECE201-ME201")
    print(
        "Correct EFI boundary would be:",
        "ECE201=Operated, ME201=Not operated",
    )
    print(
        "Injected fault:",
        "ECE201 falsely reports Not operated",
    )
    print(
        "Observed EFI inference:",
        inference["suspected_cable"],
    )
    print(
        "Trust-aware assessment:",
        trust["trust_state"],
        "-",
        trust["reason"],
    )

    print("\n=== TRUE VS OBSERVED EFI ===")
    print(evidence.to_string(index=False))

    print("\n=== BASELINE SEQUENCE ===")
    print(
        baseline["events"][
            [
                "step",
                "action",
                "inferred_fault_cable",
                "true_fault_cable",
                "served_nominal_load_A",
                "actual_faulted_cable_energized",
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
                "inferred_fault_cable",
                "true_fault_cable",
                "served_nominal_load_A",
                "actual_faulted_cable_energized",
                "actual_safe",
                "unsafe_transition",
            ]
        ].to_string(index=False)
    )

    print("\n=== COMPARISON ===")
    print(comparison.to_string(index=False))

    print("\n=== INTENDED INTERPRETATION ===")
    print(
        "1) The physical fault is ECE201-ME201, so ECE201 should report EFI Operated."
    )
    print(
        "2) A false-negative at ECE201 makes the observed pattern look like "
        "a first-cable A01-ECE201 fault."
    )
    print(
        "3) The baseline trusts the bad EFI, isolates the wrong cable, and "
        "its perceived safety checks approve NOP closure."
    )
    print(
        "4) That NOP closure actually energizes ECE201-ME201 from B09, "
        "creating one unsafe transition and one true-fault re-energization."
    )
    print(
        "5) B09 then trips on Earth Fault, so the baseline loses the restored "
        "path and ends with no load supplied in this simplified event sequence."
    )
    print(
        "6) The trust-aware controller sees an A01 E/F trip but no positive "
        "EFI-operated evidence. It blocks energizing restoration and enters "
        "Safe Hold, leaving the original B09-side 100 A nominal load supplied."
    )


if __name__ == "__main__":
    main()
