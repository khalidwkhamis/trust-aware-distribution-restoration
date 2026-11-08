# Reported Results

This directory preserves scenario-specific CSV outputs from the validated simulation runs used for report traceability.

The source scripts generally write to a local directory named `results/`. Because some scenarios use the same generic output filenames, retained outputs are separated here by scenario number.

Scenario folders `01`-`10` correspond to the PR-3 evaluation. Folders `11`-`14` add the PR-4 final-evaluation cases:

- `11_command_feedback_timeout`: delayed feedback and missing-feedback/failed-execution variants using T_cmd = 10 s;
- `12_communication_loss_dos`: selective station SCADA communication loss / DoS;
- `13_unauthorized_command_attack`: forged NOP-close command with authorization and X_safe validation;
- `14_load_sensitivity`: 90%, 100%, and 110% uniform load scaling across four representative topologies.

For scenarios 11-14, the retained repository files are the final comparison/sensitivity summary CSVs. Running the corresponding scripts regenerates those summaries locally.
