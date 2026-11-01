# Trust-Aware Post-Fault Service Restoration

This repository contains the Python simulation scripts and reported numerical outputs for an academic study of **post-fault service restoration in an 11 kV looped distribution feeder operated radially**.

The project evaluates a supervisory, trust-aware safe-switching layer that combines feeder topology, protection and earth-fault indicator (EFI) evidence, SCADA status/feedback, electrical feasibility, information trustworthiness, and Safe Hold behavior.

> **Data note:** The network, loading, cable parameters, event data, and cyber anomalies in this repository are synthetic/generic and are intended only for reproducible academic evaluation.

## Repository contents

```text
trust-aware-distribution-restoration/
├── README.md
├── requirements.txt
├── .gitignore
├── src/
│   ├── 01_feeder_simulation_v1.py
│   ├── 02_feeder_simulation_v1_1.py
│   ├── 03_clean_EF_ME201_ECE202.py
│   ├── 04_faulty_EFI_false_negative.py
│   ├── 05_faulty_spoofed_EFI_contradictory.py
│   ├── 06_clean_EF_B09_ME205.py
│   ├── 07_OC_progressive_section_retrip.py
│   ├── 08_OC_station_flashover_hold.py
│   ├── 09_OC_B09_with_EFI_station_hold.py
│   └── 10_replay_stale_EFI_attack.py
└── reported_results/
    └── scenario-specific CSV outputs used for report traceability
```

## Model summary

The executable study uses one representative A01-B09 feeder pair from a generic two-source 11 kV distribution architecture.

Key project assumptions include:

- The feeder is physically looped but operated radially.
- A01 normally supplies 200 A nominal station load.
- B09 normally supplies 100 A nominal station load.
- The normal open point is the ME203 feeder switch toward ECE204.
- The total nominal station load is 300 A.
- Each station load uses a nominal power factor of 0.95.
- The project feeder-current limit is 350 A.
- Energized modeled station voltages must remain within 0.94-1.06 pu.
- Synthetic cable sections are 0.5 km with R = 0.25 ohm/km and X = 0.08 ohm/km.
- Unintended A01/B09 paralleling is prohibited.
- The 11 kV bus coupler remains open in the modeled restoration cases.

The electrical model is balanced, three-phase, steady-state, and uses a radial backward/forward-sweep solution. NetworkX is used for connectivity/radiality checks, NumPy for electrical calculations, and pandas for auditable scenario/result tables.

## Safety and security objective

A candidate switching transition is accepted only when the resulting state satisfies the modeled safe-set conditions, including:

- suspected/confirmed unsafe equipment remains de-energized and appropriately isolated;
- the energized feeder topology remains radial;
- no unintended A01/B09 source parallel is created;
- feeder/branch current remains at or below 350 A;
- energized modeled station voltage remains within 0.94-1.06 pu; and
- required supervisory evidence is sufficiently fresh, consistent, and available for an energizing action.

The trust-aware controller classifies critical supervisory evidence as **Trusted**, **Degraded**, or **Untrusted**. Energizing restoration is blocked and the controller enters **Safe Hold** when critical evidence is stale, contradictory, or unavailable.

## Included scenarios

| Script | Scenario | Purpose |
|---|---|---|
| `01_feeder_simulation_v1.py` | Normal operation + clean E/F at ECE203-ME203 | Base feeder model and original restoration sequence |
| `02_feeder_simulation_v1_1.py` | Clean E/F with decision traceability | Adds EFI evidence and restoration decision traceability |
| `03_clean_EF_ME201_ECE202.py` | Independent clean E/F at ME201-ECE202 | Tests generalization away from the original fault location |
| `04_faulty_EFI_false_negative.py` | False-negative EFI | Compares an EFI-trusting baseline with trust-aware Safe Hold |
| `05_faulty_spoofed_EFI_contradictory.py` | Contradictory/spoofed EFI | Tests inconsistent Operated/Not-Operated evidence |
| `06_clean_EF_B09_ME205.py` | Reverse-direction B09-side E/F | Validates B09-to-NOP source-directed EFI interpretation |
| `07_OC_progressive_section_retrip.py` | O/C with unsafe cable section | Tests progressive restoration and re-trip handling |
| `08_OC_station_flashover_hold.py` | O/C with suspected station flashover | Tests station isolation and intentional partial restoration |
| `09_OC_B09_with_EFI_station_hold.py` | B09 O/C with useful EFI | Uses EFI to narrow a suspect region without treating it as deterministic cable proof |
| `10_replay_stale_EFI_attack.py` | Replay/stale EFI attack | Tests timestamp freshness and Safe Hold under plausible but stale evidence |

## Software requirements

- Python 3
- NumPy
- pandas
- NetworkX

Install the Python dependencies with:

```bash
pip install -r requirements.txt
```

## Running a scenario

From the repository root, run a script with Python, for example:

```bash
python src/03_clean_EF_ME201_ECE202.py
```

The scripts create CSV outputs in a local `results/` directory. Because the validated scripts were originally executed independently, running multiple scripts from the same working directory may reuse generic output filenames. The `reported_results/` directory therefore preserves the scenario-specific outputs corresponding to the reported evaluation.

## Reported evaluation metrics

The simulations record or evaluate, as applicable:

- unsafe switching-state transitions;
- true-fault re-energization count;
- fault/suspect localization outcome;
- healthy nominal load served;
- source and maximum branch current;
- minimum energized-node voltage; and
- information trust state / Safe Hold outcome.

## Current project scope

The executable model is intentionally supervisory-level. Detailed electromagnetic transients, relay coordination studies, arc/flashover physics, upstream 33 kV faults, transformer thermal dynamics, intentional live transfer, and field proving-dead/earthing procedures are outside the executable model.

The validated scripts are retained individually for traceability. A later refactoring may consolidate the common feeder model, controller logic, scenario definitions, and master runner while preserving these original validated copies.

## Reproducibility

`reported_results/` contains the scenario-specific CSV outputs retained from the validated runs used in the academic report. The scripts in `src/` are the sanitized source listings corresponding to those simulations.
