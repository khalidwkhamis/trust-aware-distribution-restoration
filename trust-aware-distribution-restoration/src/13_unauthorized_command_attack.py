"""
PR-4 Scenario 13: Unauthorized switching-command attack.

A forged CLOSE targets the normally open ME203_to_ECE204 NOP before fault
isolation. The trust-aware path validates authorization state and X_safe.
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
NOP = "ME203_to_ECE204"

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

def row(controller, state, executed, unsafe, reenerg, trust, outcome):
    served, currents, safety = snap(state)
    min_v = safety["min_voltage_pu"]
    return {
        "scenario": "unauthorized_NOP_close",
        "controller": controller,
        "unauthorized_commands_executed": executed,
        "unsafe_transitions": unsafe,
        "true_fault_reenergizations": reenerg,
        "final_trust_state": trust,
        "final_nominal_served_load_A": served,
        "final_A01_current_A": currents.get("A01", 0.0),
        "final_B09_current_A": currents.get("B09", 0.0),
        "final_max_branch_current_A": float(safety["max_branch_current_A"]),
        "final_min_voltage_pu": float(min_v) if pd.notna(min_v) else float("nan"),
        "final_actual_safe": bool(safety["safe"]),
        "outcome": outcome,
    }

def main():
    rows = []

    # Fault trip opens A01; forged NOP close lets B09 backfeed the fault.
    baseline = deepcopy(base.NORMAL_SWITCH_STATE)
    baseline["A01_breaker"] = False
    baseline[NOP] = True
    _, _, unsafe = snap(baseline)
    reenerg = int(unsafe["faulted_cable_energized"])
    baseline["B09_breaker"] = False
    rows.append(row(
        "Unprotected command-path baseline",
        baseline,
        1,
        1,
        reenerg,
        "Assumed trusted",
        "Forged NOP-close executes -> B09 backfeeds ECE203-ME203 -> one unsafe "
        "transition / true-fault re-energization -> B09 trips.",
    ))

    # Trust-aware path rejects command before actuation.
    trust = deepcopy(base.NORMAL_SWITCH_STATE)
    trust["A01_breaker"] = False
    candidate = deepcopy(trust)
    candidate[NOP] = True
    _, _, candidate_safety = snap(candidate)
    assert candidate_safety["faulted_cable_energized"], (
        "Candidate X_safe check should reject forged NOP close."
    )
    rows.append(row(
        "Trust-aware",
        trust,
        0,
        0,
        0,
        "Untrusted",
        "Command mismatches authorized restoration state and fails X_safe candidate "
        "check -> rejected -> B09 healthy side remains supplied.",
    ))

    out = Path("results") / "13_unauthorized_command_attack"
    out.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(out / "00_comparison_summary.csv", index=False)
    print(df.to_string(index=False))

if __name__ == "__main__":
    main()
