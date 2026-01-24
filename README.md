# UI Assertion via Screenshot — VLM QA Benchmark

UI test automation mostly relies on structured signals (DOM, selectors, accessibility tree), but:

- Classic coded verifications are heavy and slow to build and maintain.
- Code-based verifications are prone to false positives, since the verification must check intended behavior but also avoid letting potential anomalies slip by (which is hard).
- Most UI tests are based on representation, not perception: elements not accessible in the DOM or hard to identify are problematic for automated QA.
- Many real bugs are visual (e.g., overlap, wrong color), and thus not detected by code-based checks.

Existing multimodal benchmarks evaluate UI understanding, but they typically do not report deployment-oriented reliability metrics (false-positive rate, risk/coverage under abstention, calibration, consistency, run-to-run stability) for screenshot-based UI assertions.

This project introduces a lightweight benchmark to evaluate Vision-Language Models (VLMs) for screenshot-based UI assertions, with these goals:
- Calibrate AI confidence to meet pass/fail thresholds.
- Study which variables (via ablations) improve reliability.
- Understand what verifications a VLM can reliably perform from a screenshot.
