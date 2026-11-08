"""
PR-4 Scenario 15: Terminal EFI case -- fault on the NOP cable.

Normal topology:
    A01 -> ... -> ME203 [ME203_to_ECE204 OPEN] -- cable -- [ECE204_to_ME203 CLOSED]
         ECE204 -> ... -> B09

Therefore the ME203-ECE204 cable is energized only from the B09 side in the
normal radial state. A fault on this cable causes B09 to trip. If every EFI in
the B09-directed chain operates (ME205, ECE205, ME204, ECE204), the fault lies
beyond the final B09-side EFI. Because the next boundary is the already-open
NOP at ME203, the terminal section is uniquely the ME203-ECE204 cable.

Restoration:
1) B09 trips on Earth Fault.
2) Infer the terminal NOP cable from the all-operated B09 EFI chain.
3) Open ECE204_to_ME203. The cable is now isolated at both ends because
   ME203_to_ECE204 was already open as the normal open point.
4) Reclose B09. No NOP transfer is required; the normal 200 A / 100 A source
   split is restored.

The short-circuit transient is represented as a discrete protection event, as
in the other supervisory-level simulations. The steady-state solver is used
for the post-trip and restored switching states.
"""
from copy import deepcopy
from importlib.util import spec_from_file_location, module_from_spec
from pathlib import Path
import pandas as pd

HERE = Path(__file__).resolve().parent
spec = spec_from_file_location("base", HERE / "01_feeder_simulation_v1.py")
base = module_from_spec(spec)
spec.loader.exec_module(base)

TRUE_FAULT = ("ME203", "ECE204")
B09_SCAN = ["ME205", "ECE205", "ME204", "ECE204"]
NOP_ME203_SIDE = "ME203_to_ECE204"
B09_SIDE_SWITCH = "ECE204_to_ME203"

def infer_terminal_b09_fault(efi_status):
    seen_not_operated = False
    for station in B09_SCAN:
        status = bool(efi_status[station])
        if not status:
            seen_not_operated = True
        elif seen_not_operated:
            return {
                "valid": False,
                "suspected": None,
                "reason": "Contradictory B09-side EFI pattern.",
            }

    first_not = next(
        (i for i, s in enumerate(B09_SCAN) if not efi_status[s]),
        None,
    )

    if first_not is None:
        return {
            "valid": True,
            "suspected": TRUE_FAULT,
            "reason": (
                "All B09-side EFIs operated; fault is beyond ECE204. "
                "The next boundary is the already-open ME203 NOP, so the "
                "terminal ME203-ECE204 cable is inferred."
            ),
        }

    if first_not == 0:
        suspected = ("ME205", "B09")
    else:
        left = B09_SCAN[first_not - 1]
        right = B09_SCAN[first_not]
        suspected = (
            (left, right)
            if (left, right) in base.CABLES
            else (right, left)
        )

    return {
        "valid": True,
        "suspected": suspected,
        "reason": "Boundary between last operated and first not-operated EFI.",
    }

def snapshot(state):
    stations, branches, sources, safety = base.evaluate_state(state)
    currents = {
        row["source"]: float(row["current_A"])
        for _, row in sources.iterrows()
    }
    served = float(
        stations.loc[stations["energized"], "nominal_load_A"].sum()
    )
    return stations, branches, sources, safety, currents, served

def main():
    efi = {station: True for station in B09_SCAN}
    inference = infer_terminal_b09_fault(efi)
    assert inference["valid"]
    assert inference["suspected"] == TRUE_FAULT
    assert base.NORMAL_SWITCH_STATE[NOP_ME203_SIDE] is False
    assert base.NORMAL_SWITCH_STATE[B09_SIDE_SWITCH] is True

    state = deepcopy(base.NORMAL_SWITCH_STATE)
    sequence = []

    def record(action, decision_basis):
        stations, branches, sources, safety, currents, served = snapshot(state)
        sequence.append({
            "step": len(sequence),
            "action": action,
            "decision_basis": decision_basis,
            "B09_EFI_chain": "B09->ME205->ECE205->ME204->ECE204->NOP",
            "all_B09_side_EFIs_operated": True,
            "inferred_fault_cable": "ME203-ECE204",
            "ME203_to_ECE204_closed": state[NOP_ME203_SIDE],
            "ECE204_to_ME203_closed": state[B09_SIDE_SWITCH],
            "fault_cable_isolated_both_ends": (
                not state[NOP_ME203_SIDE]
                and not state[B09_SIDE_SWITCH]
            ),
            "A01_current_A": currents.get("A01", 0.0),
            "B09_current_A": currents.get("B09", 0.0),
            "served_nominal_load_A": served,
            "min_voltage_pu": float(safety["min_voltage_pu"]),
            "max_branch_current_A": float(safety["max_branch_current_A"]),
            "safe_steady_state": bool(safety["safe"]),
        })
        return stations, branches, sources, safety

    # Step 0: terminal cable E/F causes B09 breaker trip.
    state["B09_breaker"] = False
    record(
        "B09 trips on Earth Fault",
        "The ME203-side NOP is already open, so the NOP cable is energized "
        "from B09 through ECE204. All B09-side EFIs operated.",
    )

    # Step 1: isolate the only still-connected end of the terminal cable.
    state[B09_SIDE_SWITCH] = False
    record(
        "Open ECE204 feeder switch toward ME203/NOP cable",
        "All B09-side EFIs operated -> terminal NOP cable inferred. "
        "ME203_to_ECE204 is already open, so this action isolates the cable "
        "from both ends.",
    )

    if state[NOP_ME203_SIDE] or state[B09_SIDE_SWITCH]:
        raise RuntimeError("Terminal fault cable is not isolated at both ends.")

    # Step 2: reclose B09; no transfer/NOP closing is necessary.
    candidate = deepcopy(state)
    candidate["B09_breaker"] = True
    _, _, _, candidate_safety, _, _ = snapshot(candidate)
    if not candidate_safety["safe"]:
        raise RuntimeError(
            f"B09 reclose candidate failed steady-state safety checks: "
            f"{candidate_safety}"
        )
    state = candidate
    final = record(
        "Reclose B09",
        "Terminal cable is isolated at both ends; the normal radial source "
        "split is restored without closing the NOP.",
    )

    out = Path("results") / "15_terminal_EFI_NOP_cable_fault"
    out.mkdir(parents=True, exist_ok=True)

    seq = pd.DataFrame(sequence)
    seq.to_csv(out / "01_restoration_sequence.csv", index=False)

    stations, branches, sources, safety = final
    stations.to_csv(out / "02_final_station_results.csv", index=False)
    branches.to_csv(out / "03_final_branch_results.csv", index=False)
    sources.to_csv(out / "04_final_source_results.csv", index=False)

    currents = {
        row["source"]: float(row["current_A"])
        for _, row in sources.iterrows()
    }
    summary = pd.DataFrame([{
        "true_fault": "ME203-ECE204",
        "tripped_source": "B09",
        "EFI_scan_direction": "B09->ME205->ECE205->ME204->ECE204->NOP",
        "all_relevant_EFIs_operated": True,
        "inferred_fault": "ME203-ECE204",
        "ME203_side_already_open": True,
        "ECE204_side_opened_for_isolation": True,
        "B09_reclosed": True,
        "NOP_closed_for_transfer": False,
        "A01_final_current_A": currents.get("A01", 0.0),
        "B09_final_current_A": currents.get("B09", 0.0),
        "total_nominal_restored_A": float(
            stations.loc[stations["energized"], "nominal_load_A"].sum()
        ),
        "final_min_voltage_pu": float(safety["min_voltage_pu"]),
        "final_max_branch_current_A": float(safety["max_branch_current_A"]),
        "final_safe": bool(safety["safe"]),
    }])
    summary.to_csv(out / "00_scenario_summary.csv", index=False)

    print("\n=== TERMINAL EFI / NOP CABLE FAULT ===")
    print(summary.to_string(index=False))
    print("\n=== RESTORATION SEQUENCE ===")
    print(seq.to_string(index=False))

if __name__ == "__main__":
    main()
