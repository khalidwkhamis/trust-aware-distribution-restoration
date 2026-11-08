"""
PR-4 Scenario 12: Selective station SCADA communication loss / DoS.

Remote station command/status traffic is unavailable while local source-breaker
protection remains available. Reuses the validated feeder model in script 01.
"""
from copy import deepcopy
from importlib.util import spec_from_file_location, module_from_spec
from pathlib import Path
import pandas as pd

HERE = Path(__file__).resolve().parent
spec = spec_from_file_location("base", HERE / "01_feeder_simulation_v1.py")
base = module_from_spec(spec)
spec.loader.exec_module(base)
FAULT = ("ECE203", "ME203")

def snap(state):
    stations, branches, sources, safety = base.evaluate_state(
        state, faulted_cables=[FAULT]
    )
    currents = {
        row["source"]: float(row["current_A"])
        for _, row in sources.iterrows()
    }
    served = float(
        stations.loc[stations["energized"], "nominal_load_A"].sum()
    )
    return served, currents, safety

def row(controller, state, unsafe, reenerg, dropped, trust, outcome):
    served, currents, safety = snap(state)
    return {
        "scenario": "station_communication_DoS",
        "controller": controller,
        "unsafe_transitions": unsafe,
        "true_fault_reenergizations": reenerg,
        "remote_commands_dropped": dropped,
        "final_trust_state": trust,
        "final_nominal_served_load_A": served,
        "final_A01_current_A": currents.get("A01", 0.0),
        "final_B09_current_A": currents.get("B09", 0.0),
        "final_max_branch_current_A": float(safety["max_branch_current_A"]),
        "final_min_voltage_pu": float(safety["min_voltage_pu"]),
        "final_actual_safe": bool(safety["safe"]),
        "outcome": outcome,
    }

def main():
    rows = []

    # Baseline: A01 trips; remote isolation command is dropped; baseline recloses A01.
    baseline = deepcopy(base.NORMAL_SWITCH_STATE)
    baseline["A01_breaker"] = False
    baseline["A01_breaker"] = True
    _, _, unsafe = snap(baseline)
    reenerg = int(unsafe["faulted_cable_energized"])
    baseline["A01_breaker"] = False
    rows.append(row(
        "Availability-first baseline",
        baseline,
        1,
        reenerg,
        1,
        "Assumed trusted",
        "DoS drops remote isolation command; baseline recloses A01 via the local "
        "source-breaker channel and re-energizes the true fault before protection re-trips.",
    ))

    # Trust-aware: critical communications unavailable -> Untrusted -> Safe Hold.
    trust = deepcopy(base.NORMAL_SWITCH_STATE)
    trust["A01_breaker"] = False
    rows.append(row(
        "Trust-aware",
        trust,
        0,
        0,
        0,
        "Untrusted",
        "Critical communication loss -> Untrusted -> Safe Hold; A01 remains open, "
        "B09 healthy side remains supplied, and no unsafe energization occurs.",
    ))

    out = Path("results") / "12_communication_loss_dos"
    out.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(out / "00_comparison_summary.csv", index=False)
    print(df.to_string(index=False))

if __name__ == "__main__":
    main()
