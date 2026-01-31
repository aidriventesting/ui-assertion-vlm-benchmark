# Metrics

The benchmark uses standard ML metrics where **POSITIVE = BUG (Expected FAIL)**.
Many metrics can be included, since the output file of experiment save all information.
But these are the most important metrics

## Core Metrics
- **FNR (False Negative Rate)**: Miss Rate. Percentage of bugs that escaped the model. (Goal: < 5%)
- **FPR (False Positive Rate)**: False Alarm Rate. Percentage of correct UI screens marked as bugs. (Causes flakiness).
- **Accuracy**: Percentage of correct PASS/FAIL decisions.
- **Coverage**: Percentage of tests where the model did not abstain (for `ternary` mode).

## Confusion Matrix Convention
| Metric | Description |
|--------|-------------|
| **TP** | Correctly detected bug (Predicted FAIL, Expected FAIL). |
| **FP** | False alarm (Predicted FAIL, Expected PASS). |
| **TN** | Correct pass (Predicted PASS, Expected PASS). |
| **FN** | Missed bug (Predicted PASS, Expected FAIL). |
