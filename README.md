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

## Goals

1. Measure **false positive rate** (FPR) — avoiding missed bugs
2. Test **calibration** — is confidence score useful?
3. Study **what works** — which prompts, decomposition, grounding help? what type of assertions can a VLM reliably perform from a screenshot? a reference screenshot help? 

## Dataset

Each screenshot lives in `dataset/screenshots/<shot_id>/` with:
- `screenshot.png` — the image
- `tests.json` — assertions to verify

**Test format:**
```json
{
  "test_id": "shot001_01",
  "assertion": "Vérifier que le texte exact 'ÉTAPE SUIVANTE' est visible.",
  "expected": "PASS",
}
```

**Tag structure:**

Tags will help understand the type of assertions, the difficulty, and the type of bugs that VLM can reliably perform from a screenshot.
---
