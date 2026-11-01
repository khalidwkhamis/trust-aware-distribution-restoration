"""
ECEPRY Overcurrent / Possible Station Flashover
Scenario B: ECE202 remains de-energized until field inspection
================================================================

PURPOSE
-------
This is the second O/C progressive-restoration scenario.

Unlike Scenario A, where a cable section was suspected and the station on the
opposite side could later be restored safely, this scenario assumes the newly
energized STATION itself may have flashed over.

True hidden unsafe asset:
    ECE202 station

Initial event:
    A01 trips on Overcurrent (O/C)

EFI:
    Inconclusive / not used for deterministic localization.

CONTROL PRINCIPLE
-----------------
The controller restores one incremental target at a time.

When energizing ECE202 causes an immediate O/C re-trip:
- ECE202 becomes the suspected unsafe asset.
- BOTH local feeder switches at ECE202 are opened/held open:
      ECE202_to_ME201
      ECE202_to_ME202
- ECE202 is NOT energized again from A01.
- ECE202 is NOT energized from B09.
- Healthy stations upstream are restored from A01.
- Healthy stations downstream are restored progressively from B09.
- ECE202 remains de-energized until field inspection.

This is deliberately different from Scenario A:
    Scenario A suspected a cable section and later restored ECE203 from
    the opposite side.
    Scenario B suspects the station itself, so that station remains isolated.

EXPECTED FINAL HEALTHY LOAD
---------------------------
All station loads except ECE202 (45 A) are restored:
    300 A - 45 A = 255 A nominal served load.

MODEL ASSUMPTIONS
-----------------
- Balanced 11 kV three-phase steady-state model.
- PF = 0.95.
- Current limit = 350 A.
- Voltage safe band = 0.94-1.06 pu.
- Synthetic 0.5 km cable sections.
- R = 0.25 ohm/km.
- X = 0.08 ohm/km.
- A01 and B09 are ideal 1.0 pu sources.
- All station switches are remotely controlled.
- NOP = ME203 feeder switch toward ECE204.
- Detailed arc/flashover physics are outside the minimum model.
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

SUSPECT_FLASHED_STATION = "ECE202"


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
# 4. Electrical helper functions
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
            "Energized component is not radial; source parallel or loop detected."
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


def station_is_energized(switch_state, station):
    stations, _, _, _ = evaluate_state(switch_state)
    row = stations.loc[stations["station"] == station]

    if row.empty:
        return False

    return bool(row.iloc[0]["energized"])


# ============================================================
# 5. Scenario B controller
# ============================================================

def run_station_flashover_scenario():
    state = deepcopy(NORMAL_SWITCH_STATE)
    events = []

    suspected_station = None
    retrip_count = 0
    reenergizations_after_suspicion = 0

    def record(
        action,
        decision_basis,
        protection_state="No new trip",
        incremental_target="",
        candidate_result="",
    ):
        nonlocal reenergizations_after_suspicion

        stations, branches, sources, electrical = evaluate_state(state)

        source_current = {
            row["source"]: row["current_A"]
            for _, row in sources.iterrows()
        }

        flashed_station_energized = bool(
            stations.loc[
                stations["station"] == SUSPECT_FLASHED_STATION,
                "energized",
            ].iloc[0]
        )

        if (
            suspected_station == SUSPECT_FLASHED_STATION
            and flashed_station_energized
        ):
            reenergizations_after_suspicion += 1

        events.append({
            "step": len(events),
            "action": action,
            "decision_basis": decision_basis,
            "incremental_target": incremental_target,
            "protection_state": protection_state,
            "candidate_result": candidate_result,
            "suspected_flashed_station": suspected_station or "",
            "ECE202_left_switch_state": (
                "Closed" if state["ECE202_to_ME201"] else "Open"
            ),
            "ECE202_right_switch_state": (
                "Closed" if state["ECE202_to_ME202"] else "Open"
            ),
            "ECE202_energized": flashed_station_energized,
            "A01_current_A": source_current.get("A01", 0.0),
            "B09_current_A": source_current.get("B09", 0.0),
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
            "reenergizations_after_station_suspected": (
                reenergizations_after_suspicion
            ),
        })

        return stations, branches, sources, electrical

    # --------------------------------------------------------
    # Initial O/C event.
    # --------------------------------------------------------
    state["A01_breaker"] = False
    record(
        "A01 trips on Overcurrent",
        (
            "O/C trip detected. EFI is inconclusive, so select progressive "
            "one-station-at-a-time restoration."
        ),
        protection_state="A01 O/C trip",
    )

    # --------------------------------------------------------
    # Sectionalize the A01-fed path.
    #
    # Important for this station-flashover test:
    # ECE202's own right-hand switch ECE202_to_ME202 is NOT opened here.
    # Instead ME202_to_ECE202 is opened. This lets the later re-trip cause
    # the controller to explicitly open BOTH local ECE202 switches.
    # --------------------------------------------------------
    sectionalizing = [
        "ECE201_to_ME201",
        "ME201_to_ECE202",
        "ME202_to_ECE202",
        "ME202_to_ECE203",
        "ECE203_to_ME203",
    ]

    for sw in sectionalizing:
        state[sw] = False
        record(
            f"Open {sw}",
            (
                "Non-energizing sectionalization action to prepare "
                "incremental restoration."
            ),
            candidate_result="Non-energizing action",
        )

    # Restore ECE201.
    state["A01_breaker"] = True
    record(
        "Reclose A01 breaker",
        "Restore first incremental target ECE201.",
        protection_state="No re-trip",
        incremental_target="ECE201",
        candidate_result="Successful incremental restoration",
    )

    # Restore ME201.
    state["ECE201_to_ME201"] = True
    _, _, _, electrical = evaluate_state(state)
    if not electrical["electrically_admissible"]:
        raise RuntimeError("ME201 restoration failed electrical checks.")

    record(
        "Close ECE201_to_ME201",
        "Add ME201. No O/C re-trip occurs.",
        protection_state="No re-trip",
        incremental_target="ME201",
        candidate_result="Successful incremental restoration",
    )

    # --------------------------------------------------------
    # Attempt ECE202.
    # This is the flashed station, so protection re-trips.
    # --------------------------------------------------------
    state["ME201_to_ECE202"] = True

    record(
        "Close ME201_to_ECE202",
        (
            "Attempt to energize the next station, ECE202. "
            "The station is energized from A01 and O/C protection re-trips."
        ),
        protection_state="O/C re-trip triggered",
        incremental_target="ECE202",
        candidate_result="FAILED: suspected station flashover",
    )

    retrip_count += 1

    # Protection opens A01.
    state["A01_breaker"] = False
    record(
        "A01 breaker trips again",
        (
            "The re-trip occurred immediately when ECE202 was energized. "
            "Classify ECE202 as the suspected flashed station."
        ),
        protection_state="A01 O/C re-trip",
        incremental_target="ECE202",
        candidate_result="Suspected station identified by re-trip",
    )

    suspected_station = SUSPECT_FLASHED_STATION

    # --------------------------------------------------------
    # Isolate the STATION itself using both local feeder switches.
    # --------------------------------------------------------
    state["ECE202_to_ME201"] = False
    record(
        "Open ECE202 feeder switch toward ME201",
        (
            "Isolate the suspected station from the A01 side. "
            "Do not energize ECE202 again remotely."
        ),
        candidate_result="Station-side isolation",
    )

    state["ECE202_to_ME202"] = False
    record(
        "Open ECE202 feeder switch toward ME202",
        (
            "Isolate the suspected station from the opposite side. "
            "ECE202 is now disconnected from both feeder directions."
        ),
        candidate_result="Suspected station isolated on both sides",
    )

    # --------------------------------------------------------
    # Restore healthy upstream A01 load.
    # ECE202 remains isolated because its left local switch is open.
    # --------------------------------------------------------
    state["A01_breaker"] = True
    _, _, _, electrical = evaluate_state(state)

    if not electrical["electrically_admissible"]:
        raise RuntimeError("Healthy A01-side restoration failed checks.")

    record(
        "Reclose A01 after ECE202 isolation",
        (
            "Restore ECE201 and ME201 only. "
            "ECE202_to_ME201 remains open, so ECE202 is not energized."
        ),
        protection_state="No re-trip",
        candidate_result="Healthy upstream area restored",
    )

    # --------------------------------------------------------
    # Restore downstream healthy stations from B09 progressively.
    # --------------------------------------------------------
    state["ME203_to_ECE204"] = True
    _, _, _, electrical = evaluate_state(state)
    if not electrical["electrically_admissible"]:
        raise RuntimeError("NOP closure failed electrical checks.")

    record(
        "Close ME203_to_ECE204 NOP",
        (
            "Restore ME203 from B09. ECE203_to_ME203 remains open, "
            "so only one new target is added."
        ),
        protection_state="No re-trip",
        incremental_target="ME203",
        candidate_result="Successful B09-side restoration",
    )

    state["ECE203_to_ME203"] = True
    _, _, _, electrical = evaluate_state(state)
    if not electrical["electrically_admissible"]:
        raise RuntimeError("ECE203 restoration failed electrical checks.")

    record(
        "Close ECE203_to_ME203",
        "Restore ECE203 from B09.",
        protection_state="No re-trip",
        incremental_target="ECE203",
        candidate_result="Successful B09-side restoration",
    )

    state["ME202_to_ECE203"] = True
    _, _, _, electrical = evaluate_state(state)
    if not electrical["electrically_admissible"]:
        raise RuntimeError("ME202 restoration failed electrical checks.")

    final_state = record(
        "Close ME202_to_ECE203",
        (
            "Restore ME202 from B09. "
            "ME202_to_ECE202 remains open and ECE202_to_ME202 remains open, "
            "so the suspected flashed ECE202 station cannot be energized."
        ),
        protection_state="No re-trip",
        incremental_target="ME202",
        candidate_result="Successful B09-side restoration",
    )

    return {
        "events": pd.DataFrame(events),
        "final_stations": final_state[0],
        "final_branches": final_state[1],
        "final_sources": final_state[2],
        "final_electrical": final_state[3],
        "retrip_count": retrip_count,
        "suspected_station": suspected_station,
        "reenergizations_after_suspicion": reenergizations_after_suspicion,
        "final_switch_state": state,
    }


# ============================================================
# 6. Save results
# ============================================================

def main():
    output = Path("results")
    output.mkdir(exist_ok=True)

    result = run_station_flashover_scenario()

    result["events"].to_csv(
        output / "01_progressive_restoration_sequence.csv",
        index=False,
    )
    result["final_stations"].to_csv(
        output / "02_final_station_results.csv",
        index=False,
    )
    result["final_branches"].to_csv(
        output / "03_final_branch_results.csv",
        index=False,
    )
    result["final_sources"].to_csv(
        output / "04_final_source_results.csv",
        index=False,
    )

    final_sources = {
        row["source"]: row["current_A"]
        for _, row in result["final_sources"].iterrows()
    }

    final_station_rows = result["final_stations"]
    final_ece202 = final_station_rows.loc[
        final_station_rows["station"] == "ECE202"
    ].iloc[0]

    final_served = float(
        final_station_rows.loc[
            final_station_rows["energized"],
            "nominal_load_A",
        ].sum()
    )

    summary = pd.DataFrame([{
        "initial_trip": "A01 Overcurrent",
        "EFI_localization": "Inconclusive / not used",
        "restoration_method": "Progressive one-station-at-a-time",
        "suspected_flashed_station": result["suspected_station"],
        "OC_retrip_count": result["retrip_count"],
        "reenergizations_after_station_suspected": result[
            "reenergizations_after_suspicion"
        ],
        "ECE202_final_energized": bool(final_ece202["energized"]),
        "ECE202_left_local_switch": (
            "Closed"
            if result["final_switch_state"]["ECE202_to_ME201"]
            else "Open"
        ),
        "ECE202_right_local_switch": (
            "Closed"
            if result["final_switch_state"]["ECE202_to_ME202"]
            else "Open"
        ),
        "final_A01_current_A": final_sources.get("A01", 0.0),
        "final_B09_current_A": final_sources.get("B09", 0.0),
        "final_nominal_served_load_A": final_served,
        "unserved_suspect_station_load_A": NOMINAL_LOAD_A["ECE202"],
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
        "field_inspection_required_before_ECE202_reenergization": True,
    }])

    summary.to_csv(
        output / "00_scenario_summary.csv",
        index=False,
    )

    print("\n=== SCENARIO B: SUSPECTED STATION FLASHOVER ===")
    print("Initial trip: A01 Overcurrent")
    print("Suspected flashed station after re-trip: ECE202")
    print("EFI localization: inconclusive / not used")

    print("\n=== PROGRESSIVE RESTORATION SEQUENCE ===")
    print(
        result["events"][
            [
                "step",
                "action",
                "incremental_target",
                "protection_state",
                "candidate_result",
                "suspected_flashed_station",
                "ECE202_left_switch_state",
                "ECE202_right_switch_state",
                "ECE202_energized",
                "A01_current_A",
                "B09_current_A",
                "served_nominal_load_A",
                "reenergizations_after_station_suspected",
            ]
        ].to_string(index=False)
    )

    print("\n=== FINAL SOURCE LOADING ===")
    print(result["final_sources"].to_string(index=False))

    print("\n=== SUMMARY ===")
    print(summary.to_string(index=False))

    print("\n=== INTENDED INTERPRETATION ===")
    print(
        "1) Progressive restoration successfully restores ECE201 and ME201."
    )
    print(
        "2) Energizing ECE202 causes the O/C re-trip, so the station itself "
        "becomes the suspected flashed asset."
    )
    print(
        "3) Both local ECE202 feeder switches are opened: toward ME201 and "
        "toward ME202."
    )
    print(
        "4) ECE202 is never energized again after being classified as suspect."
    )
    print(
        "5) A01 restores ECE201 and ME201; B09 restores ME203, ECE203, and "
        "ME202 progressively from the opposite side."
    )
    print(
        "6) ECE202 remains de-energized with its 45 A nominal load unserved "
        "until field inspection confirms it is safe."
    )
    print(
        "7) Final healthy served load is therefore 255 A, not 300 A."
    )


if __name__ == "__main__":
    main()
