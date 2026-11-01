"""
ECEPRY Replay / Stale EFI Cyberattack Simulation
================================================

PURPOSE
-------
This scenario compares:

1) a conventional EFI-trusting baseline controller, and
2) the proposed trust-aware controller

under a replay/stale-information cyberattack.

TRUE CURRENT PHYSICAL FAULT
---------------------------
ECE203-ME203

Correct current EFI pattern from A01 toward the NOP:
    ECE201 = Operated
    ME201  = Operated
    ECE202 = Operated
    ME202  = Operated
    ECE203 = Operated
    ME203  = Not operated

Therefore, fresh/current EFI evidence would correctly identify:
    ECE203-ME203

REPLAYED STALE EFI SNAPSHOT
---------------------------
The attacker replays an older EFI snapshot from a different earlier event:

    ECE201 = Operated
    ME201  = Not operated
    ECE202 = Not operated
    ME202  = Not operated
    ECE203 = Not operated
    ME203  = Not operated

That stale snapshot falsely implies:
    ECE201-ME201

TIMING
------
Current event time             = 100 s
Replayed snapshot timestamp    = 88 s
Snapshot age                   = 12 s

Project freshness threshold:
    T_fresh = 5 s

Thus:
    12 s > 5 s

TRUST LOGIC
-----------
Baseline:
- ignores freshness/age,
- accepts the stale EFI snapshot,
- isolates the wrong cable,
- closes the NOP,
- backfeeds the TRUE ECE203-ME203 fault from B09,
- causes an unsafe transition and B09 trip.

Trust-aware:
- checks age of the critical EFI snapshot,
- marks it Untrusted because age > T_fresh,
- blocks energizing restoration,
- enters Safe Hold,
- retains the original healthy B09-side load,
- produces zero unsafe energizing transitions.

MODEL ASSUMPTIONS
-----------------
- Balanced 11 kV three-phase steady-state model.
- PF = 0.95.
- Current limit = 350 A.
- Voltage safe band = 0.94-1.06 pu.
- Synthetic identical cable sections:
      length = 0.5 km
      R = 0.25 ohm/km
      X = 0.08 ohm/km
- A01 and B09 are ideal 1.0 pu sources.
- All station switches are remotely controlled.
- NOP = ME203 feeder switch toward ECE204.
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

T_FRESH_S = 5.0

CURRENT_EVENT_TIME_S = 100.0
REPLAY_SNAPSHOT_TIME_S = 88.0
REPLAY_AGE_S = CURRENT_EVENT_TIME_S - REPLAY_SNAPSHOT_TIME_S

CABLE_LENGTH_KM = 0.50
R_OHM_PER_KM = 0.25
X_OHM_PER_KM = 0.08
Z_SECTION = complex(
    R_OHM_PER_KM * CABLE_LENGTH_KM,
    X_OHM_PER_KM * CABLE_LENGTH_KM,
)


# ============================================================
# 2. Feeder topology and loads
# ============================================================

STATIONS = [
    "ECE201", "ME201", "ECE202", "ME202", "ECE203",
    "ME203", "ECE204", "ME204", "ECE205", "ME205",
]

A01_LOCALIZATION_CHAIN = [
    "ECE201", "ME201", "ECE202", "ME202", "ECE203", "ME203"
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
    for device in devices:
        NORMAL_SWITCH_STATE[device] = True

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
# 4. EFI helpers
# ============================================================

def infer_clean_earth_fault_from_efi(efi_status):
    """
    Infer a clean A01-side Earth-Fault boundary using the relevant
    pre-fault A01 localization chain.
    """
    seen_not = False

    for station in A01_LOCALIZATION_CHAIN:
        operated = bool(efi_status[station])

        if not operated:
            seen_not = True
        elif seen_not:
            return {
                "valid": False,
                "suspected_cable": None,
                "last_operated": None,
                "first_not_operated": None,
                "reason": "Contradictory EFI pattern.",
            }

    first_not_index = next(
        (
            i
            for i, station in enumerate(A01_LOCALIZATION_CHAIN)
            if not efi_status[station]
        ),
        None,
    )

    if first_not_index is None:
        return {
            "valid": True,
            "suspected_cable": None,
            "last_operated": A01_LOCALIZATION_CHAIN[-1],
            "first_not_operated": None,
            "reason": "All relevant EFIs operated; no boundary identified.",
        }

    first_not = A01_LOCALIZATION_CHAIN[first_not_index]

    if first_not_index == 0:
        suspected = ("A01", first_not)
        last_operated = None
    else:
        last_operated = A01_LOCALIZATION_CHAIN[first_not_index - 1]
        suspected = (last_operated, first_not)

    return {
        "valid": True,
        "suspected_cable": suspected,
        "last_operated": last_operated,
        "first_not_operated": first_not,
        "reason": "Boundary between last operated and first not-operated EFI.",
    }


def assess_freshness(snapshot_age_s):
    """
    Project timing rule:
        fresh if age <= T_fresh
        untrusted if age > T_fresh
    """
    fresh = snapshot_age_s <= T_FRESH_S

    return {
        "fresh": fresh,
        "trust_state": "Trusted" if fresh else "Untrusted",
        "energizing_allowed": fresh,
        "reason": (
            f"EFI snapshot age = {snapshot_age_s:.1f} s; "
            f"T_fresh = {T_FRESH_S:.1f} s. "
            + (
                "Critical EFI data are fresh."
                if fresh
                else "Critical EFI data are stale/replayed; energizing restoration is blocked."
            )
        ),
    }


def build_true_vs_received_table(true_efi, replay_efi):
    rows = []

    for station in STATIONS:
        true_state = bool(true_efi[station])
        received_state = bool(replay_efi[station])

        rows.append({
            "station": station,
            "current_true_EFI": (
                "Operated" if true_state else "Not operated"
            ),
            "received_replayed_EFI": (
                "Operated" if received_state else "Not operated"
            ),
            "value_matches_current_truth": true_state == received_state,
            "received_snapshot_timestamp_s": REPLAY_SNAPSHOT_TIME_S,
            "current_event_time_s": CURRENT_EVENT_TIME_S,
            "received_snapshot_age_s": REPLAY_AGE_S,
            "T_fresh_s": T_FRESH_S,
            "freshness_status": (
                "Fresh" if REPLAY_AGE_S <= T_FRESH_S else "Stale"
            ),
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

    true_fault_energized = False

    if true_fault is not None:
        reverse = (true_fault[1], true_fault[0])

        if true_fault in active or reverse in active:
            start = true_fault[0]

            if start in graph:
                connected = nx.node_connected_component(graph, start)

                if "A01" in connected or "B09" in connected:
                    true_fault_energized = True

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
        "true_fault_energized": bool(true_fault_energized),
        "unintended_source_parallel": bool(parallel),
        "actual_safe": bool(
            current_ok
            and voltage_ok
            and not true_fault_energized
            and not parallel
        ),
    }


# ============================================================
# 6. Baseline controller
# ============================================================

def run_baseline(true_fault, replayed_efi):
    inference = infer_clean_earth_fault_from_efi(replayed_efi)

    if not inference["valid"] or inference["suspected_cable"] is None:
        raise RuntimeError("Baseline cannot infer a replayed fault boundary.")

    suspected = inference["suspected_cable"]
    upstream_device, downstream_device = EDGE_SWITCHES[suspected]

    state = deepcopy(NORMAL_SWITCH_STATE)
    events = []

    def record(action, basis):
        stations, branches, sources, safety = evaluate_state(
            state,
            true_fault=true_fault,
        )

        currents = {
            row["source"]: row["current_A"]
            for _, row in sources.iterrows()
        }

        events.append({
            "step": len(events),
            "controller": "EFI-trusting baseline",
            "action": action,
            "decision_basis": basis,
            "snapshot_age_s": REPLAY_AGE_S,
            "T_fresh_s": T_FRESH_S,
            "freshness_checked": False,
            "received_inferred_fault": "-".join(suspected),
            "true_fault": "-".join(true_fault),
            "A01_current_A": currents.get("A01", 0.0),
            "B09_current_A": currents.get("B09", 0.0),
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
        (
            "Baseline receives EFI data but does not verify freshness. "
            "Replayed snapshot is accepted as current."
        ),
    )

    state[upstream_device] = False
    record(
        f"Open {upstream_device}",
        (
            f"Replayed EFI falsely implies {suspected[0]}-{suspected[1]}. "
            "Open the upstream side of that wrong inferred cable."
        ),
    )

    state["A01_breaker"] = True
    record(
        "Reclose A01 breaker",
        (
            "Baseline believes the replayed fault boundary has been safely "
            "disconnected from A01."
        ),
    )

    state[downstream_device] = False
    record(
        f"Open {downstream_device}",
        "Complete isolation of the wrong replay-inferred cable.",
    )

    state["ME203_to_ECE204"] = True
    _, _, _, after_close = record(
        "Close ME203_to_ECE204 NOP",
        (
            "Baseline closes the alternate path based on stale EFI evidence. "
            "This backfeeds the actual ECE203-ME203 fault from B09."
        ),
    )

    if after_close["true_fault_energized"]:
        state["B09_breaker"] = False
        final = record(
            "B09 trips after true fault is backfed",
            (
                "The replay-induced wrong restoration energized the actual "
                "faulted cable from B09."
            ),
        )
    else:
        final = evaluate_state(state, true_fault=true_fault)

    return {
        "inference": inference,
        "events": pd.DataFrame(events),
        "final_stations": final[0],
        "final_sources": final[2],
        "final_safety": final[3],
    }


# ============================================================
# 7. Trust-aware controller
# ============================================================

def run_trust_aware(true_fault, replayed_efi):
    inference = infer_clean_earth_fault_from_efi(replayed_efi)
    freshness = assess_freshness(REPLAY_AGE_S)

    state = deepcopy(NORMAL_SWITCH_STATE)
    events = []

    def record(action, basis):
        stations, branches, sources, safety = evaluate_state(
            state,
            true_fault=true_fault,
        )

        currents = {
            row["source"]: row["current_A"]
            for _, row in sources.iterrows()
        }

        events.append({
            "step": len(events),
            "controller": "Trust-aware",
            "action": action,
            "decision_basis": basis,
            "snapshot_age_s": REPLAY_AGE_S,
            "T_fresh_s": T_FRESH_S,
            "freshness_checked": True,
            "freshness_status": "Fresh" if freshness["fresh"] else "Stale",
            "trust_state": freshness["trust_state"],
            "energizing_allowed": freshness["energizing_allowed"],
            "received_inferred_fault": (
                "-".join(inference["suspected_cable"])
                if inference["suspected_cable"]
                else ""
            ),
            "true_fault": "-".join(true_fault),
            "A01_current_A": currents.get("A01", 0.0),
            "B09_current_A": currents.get("B09", 0.0),
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
        (
            "Protection trip triggers EFI collection plus timestamp/freshness "
            "validation before any restoration action."
        ),
    )

    record(
        "Enter Safe Hold; block restoration energization",
        (
            freshness["reason"]
            + " The replayed EFI boundary may be recorded for diagnostics, "
              "but it is not trusted for reclosing A01 or closing the NOP."
        ),
    )

    final = evaluate_state(state, true_fault=true_fault)

    return {
        "inference": inference,
        "freshness": freshness,
        "events": pd.DataFrame(events),
        "final_stations": final[0],
        "final_sources": final[2],
        "final_safety": final[3],
    }


# ============================================================
# 8. Main experiment
# ============================================================

def main():
    output = Path("results")
    output.mkdir(exist_ok=True)

    true_fault = ("ECE203", "ME203")

    # Current correct EFI evidence for the true fault.
    true_current_efi = {
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

    # Replayed stale EFI snapshot from an older/different event.
    replayed_efi = {
        "ECE201": True,
        "ME201": False,
        "ECE202": False,
        "ME202": False,
        "ECE203": False,
        "ME203": False,
        "ECE204": False,
        "ME204": False,
        "ECE205": False,
        "ME205": False,
    }

    true_inference = infer_clean_earth_fault_from_efi(true_current_efi)
    replay_inference = infer_clean_earth_fault_from_efi(replayed_efi)

    evidence = build_true_vs_received_table(
        true_current_efi,
        replayed_efi,
    )

    baseline = run_baseline(
        true_fault,
        replayed_efi,
    )

    trust_aware = run_trust_aware(
        true_fault,
        replayed_efi,
    )

    evidence.to_csv(
        output / "01_true_vs_replayed_efi.csv",
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

    comparison = pd.DataFrame([
        {
            "controller": "EFI-trusting baseline",
            "true_fault": "ECE203-ME203",
            "fresh_current_EFI_would_infer": "-".join(
                true_inference["suspected_cable"]
            ),
            "replayed_EFI_infers": "-".join(
                replay_inference["suspected_cable"]
            ),
            "snapshot_age_s": REPLAY_AGE_S,
            "T_fresh_s": T_FRESH_S,
            "freshness_check": "Ignored",
            "trust_state": "Assumed trusted",
            "energizing_restoration_allowed": True,
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
                "Stale EFI accepted -> wrong isolation -> NOP close backfeeds "
                "true ECE203-ME203 fault -> B09 trips."
            ),
        },
        {
            "controller": "Trust-aware",
            "true_fault": "ECE203-ME203",
            "fresh_current_EFI_would_infer": "-".join(
                true_inference["suspected_cable"]
            ),
            "replayed_EFI_infers": "-".join(
                replay_inference["suspected_cable"]
            ),
            "snapshot_age_s": REPLAY_AGE_S,
            "T_fresh_s": T_FRESH_S,
            "freshness_check": "Applied",
            "trust_state": trust_aware["freshness"]["trust_state"],
            "energizing_restoration_allowed": trust_aware["freshness"][
                "energizing_allowed"
            ],
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
                "12 s-old EFI exceeds 5 s freshness threshold -> Untrusted -> "
                "energizing restoration blocked -> Safe Hold."
            ),
        },
    ])

    comparison.to_csv(
        output / "00_comparison_summary.csv",
        index=False,
    )

    print("\n=== REPLAY / STALE EFI ATTACK ===")
    print("True physical fault: ECE203-ME203")
    print(
        "Fresh/current EFI would infer:",
        true_inference["suspected_cable"],
    )
    print(
        "Replayed stale EFI infers:",
        replay_inference["suspected_cable"],
    )
    print(
        f"Replay age = {REPLAY_AGE_S:.1f} s; "
        f"T_fresh = {T_FRESH_S:.1f} s"
    )

    print("\n=== TRUE VS REPLAYED EFI ===")
    print(evidence.to_string(index=False))

    print("\n=== BASELINE SEQUENCE ===")
    print(
        baseline["events"][
            [
                "step",
                "action",
                "snapshot_age_s",
                "freshness_checked",
                "received_inferred_fault",
                "true_fault",
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
                "snapshot_age_s",
                "T_fresh_s",
                "freshness_status",
                "trust_state",
                "energizing_allowed",
                "received_inferred_fault",
                "true_fault",
                "served_nominal_load_A",
                "true_fault_energized",
                "unsafe_transition",
            ]
        ].to_string(index=False)
    )

    print("\n=== COMPARISON ===")
    print(comparison.to_string(index=False))

    print("\n=== INTENDED INTERPRETATION ===")
    print(
        "1) The real current EFI pattern identifies ECE203-ME203 correctly."
    )
    print(
        "2) A replayed 12-second-old snapshot falsely identifies ECE201-ME201."
    )
    print(
        "3) The baseline ignores timestamp freshness and acts on the stale "
        "boundary."
    )
    print(
        "4) After wrong isolation, its NOP closure backfeeds the actual "
        "ECE203-ME203 fault from B09."
    )
    print(
        "5) The trust-aware controller applies T_fresh=5 s, detects that "
        "12 s > 5 s, marks the evidence Untrusted, and blocks energizing "
        "restoration."
    )
    print(
        "6) Trust-aware result: zero unsafe transitions and zero true-fault "
        "re-energizations; B09 keeps its normal healthy side supplied."
    )


if __name__ == "__main__":
    main()
