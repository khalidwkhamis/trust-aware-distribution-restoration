"""
PR-4 Scenario 11: Command-feedback timeout / missing feedback.

Reuses the validated feeder model in 01_feeder_simulation_v1.py.
T_cmd = 10 s. The physical fault is ECE203-ME203.
"""
from copy import deepcopy
from importlib.util import spec_from_file_location, module_from_spec
from pathlib import Path
import pandas as pd

HERE = Path(__file__).resolve().parent
spec = spec_from_file_location("base", HERE / "01_feeder_simulation_v1.py")
base = module_from_spec(spec)
spec.loader.exec_module(base)

T_CMD_S = 10.0
FAULT = ("ECE203", "ME203")
UP = "ECE203_to_ME203"
DOWN = "ME203_to_ECE203"
NOP = "ME203_to_ECE204"

def snapshot(state):
    stations, branches, sources, safety = base.evaluate_state(
        state, faulted_cables=[FAULT]
    )
    currents = {
        row["source"]: float(row["current_A"])
        for _, row in sources.iterrows()
    }
    return {
        "served": float(
            stations.loc[stations["energized"], "nominal_load_A"].sum()
        ),
        "A01": currents.get("A01", 0.0),
        "B09": currents.get("B09", 0.0),
        "maxI": float(safety["max_branch_current_A"]),
        "minV": float(safety["min_voltage_pu"]),
        "safe": bool(safety["safe"]),
        "fault_on": bool(safety["faulted_cable_energized"]),
    }

def final_row(case, controller, state, trust, timeout, unsafe, reenerg, outcome):
    x = snapshot(state)
    return {
        "case": case,
        "controller": controller,
        "T_cmd_s": T_CMD_S,
        "feedback_timeout_reached": timeout,
        "final_trust_state": trust,
        "unsafe_transitions": unsafe,
        "true_fault_reenergizations": reenerg,
        "final_nominal_served_load_A": x["served"],
        "final_A01_current_A": x["A01"],
        "final_B09_current_A": x["B09"],
        "final_max_branch_current_A": x["maxI"],
        "final_min_voltage_pu": x["minV"],
        "final_actual_safe": x["safe"],
        "outcome": outcome,
    }

def main():
    rows = []

    # A) Delayed feedback: command physically executes, but feedback arrives after T_cmd.
    baseline = deepcopy(base.NORMAL_SWITCH_STATE)
    baseline["A01_breaker"] = False
    baseline[UP] = False
    baseline["A01_breaker"] = True
    baseline[DOWN] = False
    baseline[NOP] = True
    rows.append(final_row(
        "delayed_feedback",
        "Command-issuing baseline",
        baseline,
        "Assumed trusted",
        True,
        0,
        0,
        "Command physically opened, so baseline completes restoration safely, "
        "but it changes topology before positive feedback arrives.",
    ))

    trust = deepcopy(base.NORMAL_SWITCH_STATE)
    trust["A01_breaker"] = False
    trust[UP] = False
    rows.append(final_row(
        "delayed_feedback",
        "Trust-aware",
        trust,
        "Untrusted",
        True,
        0,
        0,
        "Feedback exceeds T_cmd; controller enters Safe Hold and preserves safety "
        "at the cost of restoration availability.",
    ))

    # B) Missing feedback + failed execution.
    baseline = deepcopy(base.NORMAL_SWITCH_STATE)
    baseline["A01_breaker"] = False
    baseline["A01_breaker"] = True
    unsafe_state = snapshot(baseline)
    reenerg = int(unsafe_state["fault_on"])
    baseline["A01_breaker"] = False
    rows.append(final_row(
        "missing_feedback_failed_execution",
        "Command-issuing baseline",
        baseline,
        "Assumed trusted",
        False,
        1,
        reenerg,
        "Open command failed; baseline assumes success and recloses A01, "
        "re-energizing the true fault before protection re-trips.",
    ))

    trust = deepcopy(base.NORMAL_SWITCH_STATE)
    trust["A01_breaker"] = False
    rows.append(final_row(
        "missing_feedback_failed_execution",
        "Trust-aware",
        trust,
        "Untrusted",
        True,
        0,
        0,
        "No positive feedback by T_cmd; controller enters Safe Hold and blocks "
        "A01 reclose, preventing true-fault re-energization.",
    ))

    out = Path("results") / "11_command_feedback_timeout"
    out.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(out / "00_comparison_summary.csv", index=False)
    print(df.to_string(index=False))

if __name__ == "__main__":
    main()
