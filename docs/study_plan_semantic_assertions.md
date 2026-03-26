# Study Plan: Semantic UI Assertion Benchmark
## VLM-as-Test-Oracle beyond Perception-Level Verification

### Problem Statement

Our current ScreenQA-derived benchmark operates entirely at the **perception level**: assertions like *"Verify that 'Settings' is visible"* test OCR, not understanding. A `ctrl+F` could do the same.

Real QA test cases require **semantic comprehension** — the VLM must:
1. Understand what the assertion *means* functionally
2. Identify the relevant UI elements (not by exact text, but by role/concept)
3. Judge whether the screen state satisfies the assertion

This study designs a benchmark with **three cognitive tiers** (Perception / Understanding / Reasoning), grounded in the literature (LENS 2025, MMBench-GUI 2025, BloomVQA 2023), using real mobile task data.

---

## 1. Cognitive Tier Taxonomy

Based on convergent findings from 15+ VLM benchmarks:

| Tier | Name | What the VLM does | Bloom's equiv. | Example |
|------|------|-------------------|----------------|---------|
| **P** | Perception | Read/detect an atomic element (text, icon, color) | Remember | *"Verify that the text 'OK' is visible"* |
| **U** | Understanding | Comprehend function/state/role of a UI element or group | Understand + Apply | *"Verify that the active tab is 'Home'"* |
| **R** | Reasoning | Judge a functional state by cross-referencing multiple visual cues | Analyze + Evaluate | *"Verify that the user can proceed to checkout"* |

**Key references**:
- **LENS (2025)**: P→U→R shows 85%→69%→54% accuracy drop
- **MMBench-GUI (2025)**: Content→Grounding→Automation shows 79%→74%→27%
- **Cognitive Mismatch (2026)**: Some models score BETTER on reasoning than perception (linguistic priors bypass visual grounding) — we want to detect this

**Expected finding**: VLMs will show a steep drop from P→R. The interesting question is whether different prompting strategies (CoT, evidence_first) can close the gap.

---

## 2. Dataset Selection

### Primary: AMEX (Android Multi-annotation Expo)

**Why AMEX over AITW/MoTIF**:

| Criterion | AMEX | AITW | MoTIF |
|-----------|------|------|-------|
| Apps | 110 | 159+ | 125 |
| Screenshots | 103K | 5.7M | ~30K |
| Task instructions | 2,946 | 30K | 6.1K |
| Success/fail labels | TASK_COMPLETE / TASK_IMPOSSIBLE | No (action match only) | feasible / infeasible |
| Element annotations | 1.6M with functionality descriptions | Bounding boxes + OCR | View hierarchy XML |
| Screen descriptions | GPT-4o generated + human verified | No | No |
| Device | Pixel 7 Pro, Samsung S10 | 8 Pixel types | Emulator |
| License | CC BY 4.0 | CC BY 4.0 | CC BY 4.0 (paper) |
| Download | HuggingFace | GCS TFRecord (complex) | Google Drive |

**AMEX wins because**:
1. **Element functionality descriptions** → we can generate U-level assertions about *what elements do*, not just what they say
2. **Screen-level descriptions** → we can generate R-level assertions about *what the screen enables*
3. **TASK_COMPLETE/IMPOSSIBLE** → natural ground truth for success/failure assertions
4. **HuggingFace availability** → clean integration with our pipeline

### Secondary (for cross-validation): AitZ

- 2,504 instructions, 70+ apps, manually verified
- `coat_screen_desc` (80-120 word screen descriptions) → rich source for U/R assertions
- Apache 2.0 license
- Smaller = faster to iterate on

---

## 3. Assertion Generation Strategy

### Tier P — Perception (automated, ~200 assertions)

Same approach as ScreenQA: extract element text from annotations, generate exact-match assertions.

```
Source: AMEX element_anno → element.text
Assertion: "Verify that the text '{element_text}' is visible on the screen."
Ground truth: PASS (element exists in annotation)
```

For FAIL: perturb text (char swap, off-by-one, case change).

### Tier U — Understanding (LLM-generated + human-validated, ~200 assertions)

Use AMEX's **element functionality descriptions** to generate assertions about element *role*, not text.

**Generation prompt** (applied to each screenshot + its element annotations):
```
You are a QA engineer writing functional test assertions for a mobile app screen.

Given this screenshot and its UI element annotations:
{element_annotations_with_functionality}

Write 3 assertions that test UNDERSTANDING of the screen — what elements do,
not what they literally say. Do NOT mention exact text. Use functional language.

Format: {"assertion": "...", "expected": "PASS"}

Examples of good U-level assertions:
- "Verify that there is a way to navigate back to the previous screen"
- "Verify that the user's profile picture is displayed in the header"
- "Verify that a search functionality is available on this screen"
```

**FAIL generation for U-tier**:
- Semantic inversion: *"Verify that there is NO way to navigate back"* (when back button exists → FAIL)
- Role confusion: *"Verify that the search bar is a text input field"* (when it's actually a button → FAIL)
- State mismatch: *"Verify that the shopping cart is empty"* (when items are visible → FAIL)

**Validation**: Human review of each assertion (estimate: 2-3h for 200 assertions). Check:
- Is the assertion unambiguous?
- Can a human determine PASS/FAIL from the screenshot alone?
- Is the expected label correct?

### Tier R — Reasoning (LLM-generated + human-validated, ~150 assertions)

Use AMEX's **task instructions** + **final screenshot** to generate assertions about task completion.

**Generation prompt**:
```
You are a QA engineer. A tester was asked to perform this task on a mobile app:
  Task: "{task_instruction}"

Here is the screenshot after the task was attempted.
The task was marked as: {TASK_COMPLETE | TASK_IMPOSSIBLE}

Write 2 assertions that test whether the task goal was achieved. These should
require REASONING — the VLM must cross-reference multiple visual cues to judge.

Examples of good R-level assertions:
- "Verify that the flight booking to Paris has been confirmed"
  (requires seeing confirmation message + flight details + no error banners)
- "Verify that dark mode has been successfully enabled"
  (requires seeing dark background + correct toggle state + theme applied)
- "Verify that the item has been added to the shopping cart"
  (requires seeing cart badge increment + confirmation toast/animation)
```

**Ground truth**:
- TASK_COMPLETE + assertion about success → expected PASS
- TASK_COMPLETE + assertion about failure → expected FAIL
- TASK_IMPOSSIBLE + assertion about success → expected FAIL

---

## 4. Tag Taxonomy (v5)

### Operation Tags (what cognitive skill is tested)

```yaml
# Tier P — Perception
text_match_exact: "Character-for-character string match"
presence: "Element exists in viewport"
absence: "Element does NOT exist"
icon_recognition: "Identify an icon by its visual appearance"

# Tier U — Understanding
element_role: "Identify what a UI element does (not what it says)"
screen_type: "Classify the screen's purpose (login, settings, checkout)"
widget_state: "Understand interactive state (enabled, selected, expanded)"
navigation: "Identify navigation affordances (back, menu, tabs)"

# Tier R — Reasoning
task_completion: "Judge whether a multi-step task goal was achieved"
error_detection: "Identify error states or failure conditions"
data_consistency: "Cross-reference multiple UI values for coherence"
workflow_state: "Determine where in a workflow the user currently is"
```

### Difficulty Tags (unchanged from v4)

```yaml
near_miss, small_text, low_contrast, cluttered, occluded, confusable, truncated
```

### Cognitive Tier (derived, NOT ad-hoc)

```
tier = "R" if any R-tag in tags
      else "U" if any U-tag in tags
      else "P"
```

This is now **meaningful** because the tiers are defined by what cognitive operation the assertion *requires*, not by how many tags it has.

---

## 5. Independent Variables

| IV | Values | Rationale |
|----|--------|-----------|
| **cognitive_tier** | P, U, R | Primary research question: where does the VLM break? |
| **system_prompt** | minimal, evidence_first, strict_oracle | Carried from ScreenQA study |
| **model** | gpt-4o-mini, gpt-4o, gemini-2.0-flash, claude-3.5-sonnet | Cross-model comparison |
| **output_format** | json (CoT), abc (logprobs) | CoT may help R-tier more than P-tier |
| **few_shot** | 0-shot, 1-shot (per tier) | Exemplars may anchor correct reasoning strategy |

### Experiment Matrix (minimum viable)

| Experiment | Model | Prompt | Format | Tiers | Est. API calls |
|------------|-------|--------|--------|-------|----------------|
| E1: Baseline | gpt-4o-mini | 3 personas | json | P+U+R | 550 × 3 = 1,650 |
| E2: Strong model | gpt-4o | evidence_first | json | P+U+R | 550 |
| E3: Calibration | gpt-4o-mini | evidence_first | abc | P+U+R | 550 |
| E4: Few-shot | gpt-4o-mini | evidence_first+1shot | json | P+U+R | 550 |
| **Total** | | | | | ~3,300 |

---

## 6. Metrics Framework

### Primary DV (same convention: POSITIVE = BUG)

| Metric | What it measures | Target |
|--------|-----------------|--------|
| FNR (Miss Rate) | Bugs that escape | < 5% |
| FPR (False Alarm) | Test flakiness | < 15% |
| Balanced Accuracy | Robust overall | > 85% |
| MCC | Best single metric | > 0.7 |

### Stratified by Cognitive Tier (the key analysis)

```
For each tier T in {P, U, R}:
  - FNR_T, FPR_T, Acc_T, MCC_T
  - Bootstrap 95% CI (1000 iter)
  - McNemar's test: P vs U, U vs R, P vs R (within same prompt)
  - Effect size: Cohen's d for tier differences
```

**Expected insight**: FNR_R >> FNR_P (models miss more bugs when reasoning is needed). If FNR_R ≈ FNR_P, the model uses linguistic shortcuts.

### Calibration (per tier)

```
For each tier:
  - Brier score
  - ECE (10 bins) + reliability diagram
  - Overconfidence ratio
```

**Expected insight**: Models are likely overconfident on R-tier (high confidence but low accuracy), matching findings from "Overconfidence in LLM-as-a-Judge" (2025).

### Prompt × Tier Interaction

```
2-way analysis: prompt_strategy × cognitive_tier
- Does evidence_first help more on R-tier than P-tier?
- Does strict_oracle's conservatism reduce FNR_R at the cost of FPR_R?
```

---

## 7. Annotation Protocol

### Phase 1: Assertion Generation (automated)

1. Download AMEX screenshots + annotations from HuggingFace
2. Sample 100 diverse screens (stratified by app category)
3. Generate P-tier assertions automatically from element annotations
4. Generate U-tier and R-tier assertions via GPT-4o (with element + screen context)
5. Target: 200 P + 200 U + 150 R = 550 assertions

### Phase 2: Human Validation (manual, ~4h)

For each U/R assertion:
- [ ] Is the assertion unambiguous? (can a human answer PASS/FAIL from screenshot alone?)
- [ ] Is the expected label correct?
- [ ] Is the tier assignment correct? (does it genuinely require understanding/reasoning?)
- [ ] Rate difficulty: easy / medium / hard

Reject assertions that are ambiguous or where the tier is wrong.

### Phase 3: Balance Check

Target distribution:
- PASS/FAIL ratio: 55/45 to 60/40 (slight PASS majority, realistic)
- Per tier: minimum 100 assertions after validation
- Per app category: minimum 3 apps represented
- Difficulty: at least 20% "hard" per tier

---

## 8. Expected Conclusions

Based on the literature, we expect to show:

1. **Steep cognitive tier gradient**: Accuracy drops 15-25% from P→R, confirming VLMs are visual grounding tools, not reasoning engines (consistent with LENS, MMBench-GUI)

2. **Prompt strategy × tier interaction**: CoT/evidence_first helps on R-tier (forces step-by-step reasoning) but adds noise on P-tier (overthinks simple checks)

3. **Calibration gap widens with complexity**: Models are well-calibrated on P-tier but overconfident on R-tier (Brier_R >> Brier_P)

4. **FNR varies by tier**: FNR_P < 5% (acceptable for deployment) but FNR_R > 15% (not deployable for complex assertions without human review)

5. **Cross-model divergence increases with tier**: gpt-4o-mini ≈ gpt-4o on P-tier but gpt-4o >> gpt-4o-mini on R-tier (reasoning capability matters more at higher tiers)

### Practical Implications for Test Automation

- **P-tier assertions**: VLMs are production-ready. Can replace manual visual checks.
- **U-tier assertions**: VLMs are useful but need confidence thresholds. Route low-confidence to human review.
- **R-tier assertions**: VLMs are assistive, not autonomous. Best used as a "second pair of eyes" with human final decision.

---

## 9. Implementation Roadmap

| Step | Task | Effort | Dependency |
|------|------|--------|------------|
| 1 | Write `convert_amex.py` — download + convert AMEX to benchmark format | 1 day | HuggingFace access |
| 2 | Generate P-tier assertions from element annotations | 2h | Step 1 |
| 3 | Generate U/R-tier assertions via GPT-4o | 3h + ~$5 API | Step 1 |
| 4 | Human validation of U/R assertions | 4h manual | Step 3 |
| 5 | Run E1 (baseline, 1,650 calls) | 1h + ~$3 API | Step 4 |
| 6 | Run E2-E4 (comparisons) | 2h + ~$10 API | Step 5 |
| 7 | Compute metrics + generate report | 30min | Step 6 |
| 8 | Write findings | 1 day | Step 7 |

**Total estimated effort**: 3-4 days + ~$18 API costs

---

## 10. References

- LENS (2025): Multi-level Evaluation of Multimodal Reasoning — 3-tier P/U/R hierarchy
- MMBench-GUI (2025): 4-level GUI hierarchy, "step-wise collapse" finding
- BloomVQA (2023): Bloom's taxonomy applied to VQA, 38% drop L1→L6
- Cognitive Mismatch (2026): Recognition-reasoning inversion in MLLMs
- Visual Room 2.0 (2025): 6-level P/C hierarchy, cognition degrades faster
- AMEX (2024): 110 apps, 103K screenshots, element functionality annotations
- AitZ (EMNLP 2024): Chain-of-Action-Thought, screen descriptions
- MLLM-as-a-Judge (ICML 2024): Position bias, calibration gaps
- Overconfidence in LLM-as-a-Judge (2025): ECE 0.11-0.43, format-dependent
- Lee & Zeng (2025): Bias-corrected CIs for LLM-judge evaluations
