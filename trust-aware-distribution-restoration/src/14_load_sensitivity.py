"""
PR-4 Scenario 14: Uniform +/-10% load-sensitivity study.

Tests normal operation and three validated clean-E/F restored topologies at
90%, 100%, and 110% of every station load, reusing script 01's feeder solver.
"""
from copy import deepcopy
from importlib.util import spec_from_file_location, module_from_spec
from pathlib import Path
import pandas as pd

HERE = Path(__file__).resolve().parent
spec = spec_from_file_location("base", HERE / "01_feeder_simulation_v1.py")
base = module_from_spec(spec)
spec.loader.exec_module(base)

ORIGINAL = dict(base.NOMINAL_LOAD_A)
SCALES = (0.9, 1.0, 1.1)
NAMES = (
    "Normal",
    "Clean E/F ECE203-ME203",
    "Clean E/F ME201-ECE202",
    "Clean E/F ME205-B09",
)

def topology(name):
    state = deepcopy(base.NORMAL_SWITCH_STATE)
    if name == "Clean E/F ECE203-ME203":
        state["ECE203_to_ME203"] = False
        state["ME203_to_ECE203"] = False
        state["ME203_to_ECE204"] = True
    elif name == "Clean E/F ME201-ECE202":
        state["ME201_to_ECE202"] = False
        state["ECE202_to_ME201"] = False
        state["ME203_to_ECE204"] = True
    elif name == "Clean E/F ME205-B09":
        state["ME205_to_B09"] = False
        state["B09_breaker"] = False
        state["ME203_to_ECE204"] = True
    return state

def apply_scale(scale):
    base.NOMINAL_LOAD_A.clear()
    base.NOMINAL_LOAD_A.update({
        station: value * scale for station, value in ORIGINAL.items()
    })
    base.S_LOAD_3PH.clear()
    base.S_LOAD_3PH.update({
        station: base.three_phase_load_va(value)
        for station, value in base.NOMINAL_LOAD_A.items()
    })

def evaluate(name, scale):
    apply_scale(scale)
    stations, branches, sources, safety = base.evaluate_state(topology(name))
    served = float(
        stations.loc[stations["energized"], "nominal_load_A"].sum()
    )
    max_source = (
        float(sources["current_A"].max()) if not sources.empty else 0.0
    )
    max_i = float(safety["max_branch_current_A"])
    min_v = float(safety["min_voltage_pu"])
    max_v = float(safety["max_voltage_pu"])
    return {
        "scenario": name,
        "load_scale": scale,
        "total_system_load_A": sum(base.NOMINAL_LOAD_A.values()),
        "served_scaled_load_A": served,
        "max_source_current_A": max_source,
        "max_branch_current_A": max_i,
        "current_margin_A": base.CURRENT_LIMIT_A - max_i,
        "min_voltage_pu": min_v,
        "max_voltage_pu": max_v,
        "voltage_margin_pu": min(
            min_v - base.V_MIN_PU, base.V_MAX_PU - max_v
        ),
        "unintended_source_parallel": bool(
            safety["unintended_source_parallel"]
        ),
        "electrical_safe": bool(safety["safe"]),
    }

def main():
    rows = []
    try:
        for name in NAMES:
            for scale in SCALES:
                rows.append(evaluate(name, scale))
    finally:
        apply_scale(1.0)

    out = Path("results") / "14_load_sensitivity"
    out.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(out / "00_load_sensitivity_summary.csv", index=False)
    print(df.to_string(index=False))

if __name__ == "__main__":
    main()
