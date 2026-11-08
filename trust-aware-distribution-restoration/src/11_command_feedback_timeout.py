"""
ECEPRY Command-Feedback Timeout / Missing Feedback Simulation
=============================================================

PURPOSE
-------
Evaluate the safety effect of requiring positive command-execution feedback
before any subsequent topology-changing/energizing action.

The physical event is a clean A01-side Earth Fault on ECE203-ME203.
Correct EFI evidence localizes that cable. The first restoration command is to
open ECE203_to_ME203 (the source-side boundary switch).

Two feedback/actuation cases are evaluated under identical feeder conditions:

A) DELAYED FEEDBACK
   - The open command physically executes at t=1 s.
   - Matching SCADA feedback arrives at t=13 s (12 s after command issue).
   - This exceeds T_cmd = 10 s.

B) MISSING FEEDBACK + FAILED EXECUTION
   - The open command is issued at t=1 s.
   - The switch does not physically open.
   - No matching feedback is received.

Controllers:
1) Command-issuing baseline:
   proceeds after command dispatch without requiring positive execution
   confirmation. This deliberately weak matched baseline is used only to
   isolate the value of the command-feedback interlock.

2) Trust-aware controller:
   allows only one topology-changing command at a time. No subsequent
   energizing action is authorized until matching feedback is received.
   If no matching feedback is available by T_cmd = 10 s, information trust
   becomes Untrusted and the controller enters Safe Hold.

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
- NOP = ME203 feeder switch toward ECE204.
- Positive command feedback is safety-critical evidence for topology changes.
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

T_CMD_S = 10.0
COMMAND_ISSUE_TIME_S = 1.0
BASELINE_PROCEED_TIME_S = 2.0
DELAYED_FEEDBACK_TIME_S = 13.0  # 12 s after command issue

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

TRUE_FAULT = ("ECE203", "ME203")
UPSTREAM_ISOLATION_DEVICE = "ECE203_to_ME203"
DOWNSTREAM_ISOLATION_DEVICE = "ME203_to_ECE203"
NOP_DEVICE = "ME203_to_ECE204"


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
# 4. Electrical / topology evaluation
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
            "Energized component is not radial; source 