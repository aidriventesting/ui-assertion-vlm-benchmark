# UI Assertion via Screenshot — VLM QA Benchmark

UI test automation mostly relies on structured signals (DOM, selectors, accessibility tree), but:

- Classic coded verifications are heavy and slow to build and maintain.
- Code-based verifications are prone to false positives, since the verification must check intended behavior but also avoid letting potential anomalies slip by (which is hard).
- Most UI tests are based on representation, not perception: elements not accessible in the DOM or hard to identify are problematic for automated QA.
- Many real bugs are visual (e.g., overlap, wrong color), and thus not detected by code-based checks.

Existing multimodal benchmarks evaluate UI understanding, but they typically do not report deployment-oriented reliability metrics (false-positive rate, risk/coverage under abstention, calibration, consistency, run-to-run stability) for screenshot-based UI assertions.

This project is a lightweight experiment to evaluate Vision-Language Models (VLMs) for screenshot-based UI assertions, with some goals:
- Understanding how AI confidence can be used to meet pass/fail thresholds.
- Studying which variables (via ablations) improve reliability.
- Understanding how prompts can influence the results.
- Understanding what verifications a VLM can reliably perform from a screenshot.
- Make and provide some metrics (False positive rate, calibration, consistency, run-to-run stability) that help this understanding.


## Quick Start

### 1. Add an Experiment
Create a `.yaml` file in `experiments/`. See [experiments/README.md](ui-assertion-vlm-benchmark/experiments/README.md) for the full schema.

### 2. Run Evaluation
```bash
env/bin/python3 scripts/run_eval.py --config experiments/gpt4o-mini_personas.yaml
```

### 3. Compute Metrics
```bash
env/bin/python3 scripts/compute_metrics.py --input_file results/raw_latest.jsonl
```


## Documentation
- [Dataset](file:///Users/abdelkader/Documents/ui-assertion-vlm-benchmark/docs/dataset.md): Directory structure and `tests.json`.
- [Tags & Taxonomy](file:///Users/abdelkader/Documents/ui-assertion-vlm-benchmark/docs/tags.md): Classification of UI verifications.
- [Experiments](file:///Users/abdelkader/Documents/ui-assertion-vlm-benchmark/docs/experiments.md): Configuring and running evaluations.
- [Prompts](file:///Users/abdelkader/Documents/ui-assertion-vlm-benchmark/docs/prompts.md): Factorial design of personas and policies.
- [Metrics](file:///Users/abdelkader/Documents/ui-assertion-vlm-benchmark/docs/metrics.md): ML reporting (FNR, FPR, Accuracy).