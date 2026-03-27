# Study Plan: Semantic UI Assertion Benchmark
## Can VLMs replace QA engineers for UI test verification?

### Objective

Measure where Vision Language Models break down as test oracles — not just "can they read text?" but "can they understand a screen and judge if a test passes?"

**Target audience**: QA/test automation teams evaluating VLM adoption.
**Deliverable**: Mini research paper / blog post with actionable recommendations.

---

## 1. Research Questions

| # | Question | IV | Insight for test auto |
|---|---------|-----|----------------------|
| **RQ1** | At what cognitive level do VLMs stop being reliable test oracles? | `cognitive_tier` (P/U/R) | "Use VLMs for X, not for Y" |
| **RQ2** | Does how you phrase the assertion matter? | `assertion_phrasing` | "Write your test assertions like THIS" |
| **RQ3** | Are some app categories harder to test with VLMs? | `app_category` | "Be careful with category X" |
| **RQ4** | Does screen complexity degrade VLM accuracy? | `screen_complexity` | "Simplify screens before testing" or "doesn't matter" |

---

## 2. Variables

### Independent Variables (what we control)

| IV | Type | Values | How it's set |
|----|------|--------|-------------|
| **cognitive_tier** | per-assertion | **P** (Perception), **U** (Understanding), **R** (Reasoning) | assertion tag |
| **assertion_phrasing** | per-experiment | **imperative** ("Verify that..."), **interrogative** ("Is...?"), **declarative** ("The cart is empty") | prompt rewriting |
| **app_category** | per-screenshot (observed) | `shopping`, `social`, `productivity`, `finance`, `media`, `travel`, `settings`, `other` | AMEX metadata |
| **screen_complexity** | per-screenshot (observed, continuous) | `n_elements` (5-100+), binned as `sparse`/`moderate`/`dense`/`cluttered` | AMEX element count |

### Held Constant (E1 baseline)

| Variable | Fixed value | Why |
|----------|------------|-----|
| Model | gpt-4o-mini | Cheap, fast, representative of "practical" deployment |
| System prompt | evidence_first | Best performer from ScreenQA study |
| Output format | json (CoT) | Provides reasoning trace for error analysis |
| Few-shot | 0-shot | Simplest setup |
| Temperature | 0.0 | Deterministic |
| Image | Original resolution | No preprocessing |

### Dependent Variables (what we measure)

| DV | What it means for test auto |
|----|-----------------------------|
| **FNR** (Miss Rate) | Bugs that escape → production incidents |
| **FPR** (False Alarm) | False test failures → dev productivity loss |
| **MCC** | Overall oracle quality (single number) |
| **Balanced Accuracy** | Robust to class imbalance |

---

## 3. Dataset Design

### Sample Size Justification

For a mini paper, each stratification cell needs **≥30 observations** for meaningful bootstrap CIs (per CLT guidelines for LLM evaluation, arXiv:2503.01747).

```
3 tiers × 3 phrasings = 9 cells
9 × 40 observations/cell = 360 assertions minimum
360 assertions / 6 per screenshot = 60 screenshots
```

**Final target: 60 screenshots, 360 assertions, 1,080 API calls (360 × 3 phrasings)**

### Screenshot Sampling (60 from AMEX)

Stratified sample to ensure coverage:

| Dimension | Bins | Target per bin |
|-----------|------|---------------|
| app_category | 8 categories | 7-8 screenshots each |
| screen_complexity | sparse (≤10 elem), moderate (11-25), dense (26-50), cluttered (>50) | 15 each |

Selection algorithm:
1. Group AMEX screenshots by app category
2. Within each category, sort by n_elements
3. Pick screenshots to fill complexity bins evenly
4. Deduplicate visually similar screens from same app
5. Ensure each screenshot has ≥3 AMEX element annotations with functionality descriptions (needed for U/R generation)

### Assertion Generation (6 per screenshot)

Each screenshot gets **exactly 6 assertions**: 2 per tier.

| Tier | 2 assertions per screenshot | PASS/FAIL split | Total |
|------|---------------------------|----------------|-------|
| **P** (Perception) | 1 PASS + 1 FAIL | 60/60 | 120 |
| **U** (Understanding) | 1 PASS + 1 FAIL | 60/60 | 120 |
| **R** (Reasoning) | 1 PASS + 1 FAIL | 60/60 | 120 |
| **Total** | | 180/180 (50/50) | **360** |

#### Tier P — Perception (automated)

```
PASS: "Verify that the text '{element_text}' is visible on the screen."
FAIL: "Verify that the text '{perturbed_text}' is visible on the screen."
```
- Source: AMEX element text annotations
- Perturbation: char_swap, case_change, or number_off_by_one
- Tags: `text_match_exact`, `presence`

#### Tier U — Understanding (LLM-generated + human-validated)

```
PASS: "Verify that there is a way to navigate back to the previous screen."
      (back button exists in annotations)
FAIL: "Verify that the screen has a voice input feature."
      (no microphone element exists)
```
- Source: AMEX element functionality descriptions + GPT-4o generation
- Tags: `element_role`, `widget_state`, `navigation`, `screen_type`
- Constraint: NO exact text mentioned — must require understanding

#### Tier R — Reasoning (LLM-generated + human-validated)

```
PASS: "Verify that the user can complete a purchase from this screen."
      (cart has items + checkout button is enabled + no error messages)
FAIL: "Verify that the user has successfully logged in."
      (screen shows login form, not a logged-in state)
```
- Source: AMEX task instructions + TASK_COMPLETE/IMPOSSIBLE labels
- Tags: `task_completion`, `error_detection`, `workflow_state`
- Constraint: Must require cross-referencing ≥2 visual cues

### Phrasing Variants (3 per assertion)

Each of the 360 assertions is rewritten in 3 phrasings:

| Style | Template | Example |
|-------|----------|---------|
| **Imperative** | "Verify that {X}" | "Verify that the cart shows 3 items" |
| **Interrogative** | "Does the screen show {X}?" / "Is {X} true?" | "Does the cart show 3 items?" |
| **Declarative** | "{X}." | "The cart shows 3 items." |

Rewriting is deterministic (template-based), not LLM-generated — ensures the semantic content is identical, only phrasing changes.

---

## 4. Annotation Schema

### 4.1. Per-Screenshot (60 annotations, ~80% auto)

| Field | Type | Auto/Manual | Example |
|-------|------|-------------|---------|
| `screen_id` | str | auto | `amex_amazon_042` |
| `app_name` | str | auto (AMEX) | `Amazon Shopping` |
| `app_category` | categorical | auto (AMEX) or manual | `shopping` |
| `screen_type` | categorical | manual | `product_detail` |
| `n_elements` | int | auto (AMEX count) | `34` |
| `n_text_elements` | int | auto | `18` |
| `n_interactive` | int | auto | `7` |
| `visual_density` | ordinal | derived | `dense` |
| `color_scheme` | categorical | manual | `light` |
| `resolution` | str | auto | `1080x2400` |

**Manual effort**: screen_type + color_scheme → ~20 sec/screenshot = **20 min total**.

### 4.2. Per-Assertion (360 annotations)

| Field | Type | Auto/Manual | Example |
|-------|------|-------------|---------|
| `test_id` | str | auto | `amex_amazon_042_U_01` |
| `assertion` | str | generated | (the test text) |
| `expected` | str | generated + validated | `PASS` |
| `cognitive_tier` | P/U/R | auto (from generation method) | `U` |
| `operation` | categorical | auto | `element_role` |
| `specificity` | ordinal | manual | `moderate` |
| `n_visual_cues` | int | manual | `2` |
| `requires_domain_knowledge` | bool | manual | `false` |
| `generation_method` | categorical | auto | `llm_generated` |
| `perturbation_type` | categorical | auto | `none` |
| `validated_by` | str | manual | `human` |
| `validation_confidence` | ordinal | manual | `certain` |

**Manual effort**: Only for U/R assertions (240): specificity + n_visual_cues + domain_knowledge + validation → ~40 sec each = **2.5h**.

### 4.3. Human Validation Checklist (for U/R assertions)

For each of the 240 U/R assertions:

```
1. Can PASS/FAIL be determined from screenshot alone?     [ Yes / No → REJECT ]
2. Is the expected label correct?                         [ Yes / Fix / Reject ]
3. Does it genuinely require the tagged tier (U or R)?    [ Yes / Re-tag ]
4. Specificity?                [ vague / moderate / precise ]
5. Visual cues needed?         [ 1 / 2 / 3+ ]
6. Needs domain knowledge?     [ no / yes ]
7. Your confidence?            [ certain / probable / ambiguous ]
```

**Quality gate**: Reject if ambiguous or wrong tier. Target ≤15% rejection → regenerate to fill.

---

## 5. Experiment Plan

### E1: Baseline (RQ1 + RQ3 + RQ4)

| Parameter | Value |
|-----------|-------|
| Assertions | 360 (imperative phrasing only) |
| Model | gpt-4o-mini |
| Prompt | evidence_first |
| Format | json (CoT) |
| API calls | **360** |
| Est. cost | ~$1.50 |

**Analysis**:
- FNR, FPR, MCC stratified by cognitive_tier → **RQ1**
- Same metrics stratified by app_category → **RQ3**
- Spearman ρ between n_elements and accuracy → **RQ4**
- Bootstrap 95% CIs for all metrics

### E2: Phrasing Ablation (RQ2)

| Parameter | Value |
|-----------|-------|
| Assertions | 360 × 3 phrasings = 1,080 |
| Model | gpt-4o-mini |
| Prompt | evidence_first |
| Format | json (CoT) |
| API calls | **1,080** |
| Est. cost | ~$4.50 |

**Analysis**:
- FNR, FPR by phrasing × tier (2-way)
- McNemar's test: imperative vs interrogative, imperative vs declarative
- Key question: does interrogative reduce FPR on R-tier?

### Total

| | Calls | Cost |
|-|-------|------|
| E1 | 360 | ~$1.50 |
| E2 | 1,080 | ~$4.50 |
| **Total** | **1,440** | **~$6** |

---

## 6. Analysis Plan

### 6.1. RQ1 — Cognitive Tier Gradient

```
For each tier T ∈ {P, U, R}:
  FNR_T ± 95% CI (bootstrap, 1000 iter)
  FPR_T ± 95% CI
  MCC_T ± 95% CI
  Balanced_Acc_T ± 95% CI

Pairwise: McNemar P↔U, U↔R, P↔R
Effect size: Cohen's d for significant pairs
```

**Expected result table** (for the paper):

| Tier | FNR | FPR | MCC | Balanced Acc |
|------|-----|-----|-----|-------------|
| P | 3-5% | 10-15% | 0.75-0.85 | 88-92% |
| U | 8-15% | 15-25% | 0.55-0.70 | 75-85% |
| R | 15-30% | 20-35% | 0.35-0.55 | 60-75% |

**Insight for test auto**: "VLMs are reliable for perception-level checks (text visible, element present). For understanding-level (widget state, screen type), use with confidence thresholds. For reasoning-level (task completion, workflow state), always pair with human review."

### 6.2. RQ2 — Phrasing Effect

```
For each (phrasing P, tier T):
  FNR_{P,T}, FPR_{P,T}

Interaction: is phrasing effect larger for R than P?
```

**Expected**: Interrogative reduces FPR on U/R (invites doubt → fewer false alarms), but may increase FNR (more hesitation → misses bugs).

**Insight for test auto**: "Write test assertions as questions ('Is the checkout button enabled?') rather than commands ('Verify that the checkout button is enabled') — reduces false positives by X%."

### 6.3. RQ3 — App Category

```
For each category C:
  FNR_C, FPR_C, MCC_C, N_C

Rank categories by MCC (best → worst)
```

**Expected**: Settings/forms > shopping/productivity > social/media > custom UI.

**Insight for test auto**: "VLMs are most reliable on standardized UIs (settings, forms) and least reliable on media-heavy or custom-designed screens. Allocate human QA time to category X."

### 6.4. RQ4 — Screen Complexity

```
Spearman ρ(n_elements, accuracy) — overall and per tier
Box plot: accuracy by visual_density bins, colored by tier
```

**Expected**: Negative correlation for P-tier (more elements = more OCR confusion), weak/no correlation for R-tier (reasoning is about understanding, not scanning).

**Insight for test auto**: "Screen complexity hurts text-matching assertions but doesn't affect functional verification. Don't avoid VLMs on complex screens — just avoid OCR-level assertions on them."

---

## 7. Paper Outline

### Title
*"Where VLMs Break as Test Oracles: A Cognitive Tier Analysis of Vision-Language Models for UI Assertion Verification"*

### Structure

1. **Introduction** (1 page)
   - VLMs are being adopted for visual testing
   - Current benchmarks only test perception (OCR-level)
   - We need to know: at what cognitive level do they fail?

2. **Related Work** (0.5 page)
   - VLM benchmarks (LENS, MMBench-GUI, VisualWebBench)
   - VLM-as-judge (MLLM-as-a-Judge, calibration studies)
   - UI testing with VLMs (WebTestPilot, VisionDroid)

3. **Method** (1.5 pages)
   - P/U/R tier taxonomy (with Bloom's mapping)
   - Dataset: 60 AMEX screenshots, 360 assertions
   - Phrasing ablation design
   - Annotation protocol

4. **Results** (2 pages)
   - RQ1: Tier gradient (the main figure — bar chart with CIs)
   - RQ2: Phrasing × tier interaction (heatmap)
   - RQ3: App category ranking (horizontal bar chart)
   - RQ4: Complexity correlation (scatter plot)

5. **Discussion** (1 page)
   - Deployment recommendations per tier
   - When to trust, when to verify
   - Limitations (single model, English only, AMEX apps)

6. **Conclusion** (0.5 page)
   - VLMs are P-ready, U-cautious, R-not-ready
   - Test teams should match assertion level to VLM capability

### Key Figures

| Fig | Type | Shows |
|-----|------|-------|
| Fig 1 | Bar chart + CI whiskers | FNR by tier (P < U < R) — the money shot |
| Fig 2 | Grouped bar chart | FNR × phrasing × tier interaction |
| Fig 3 | Horizontal bar chart | MCC by app category, sorted |
| Fig 4 | Scatter + regression line | n_elements vs accuracy, colored by tier |
| Table 1 | Full metrics | All DVs × tier, with bootstrap CIs |
| Table 2 | McNemar results | Pairwise significance tests |

---

## 8. Implementation Roadmap

| Step | Task | Effort | Output |
|------|------|--------|--------|
| 1 | `convert_amex.py` — download, sample 60 screens, extract metadata | 4h | `dataset_amex/screenshots/`, `images.json` |
| 2 | Generate P-tier assertions (auto from element text) | 1h | 120 P assertions |
| 3 | Generate U/R-tier assertions (GPT-4o + templates) | 2h + ~$3 | 240 U/R assertions |
| 4 | Human validation of U/R + annotation | 3h | validated `tests.json` per screen |
| 5 | Generate phrasing variants (template-based) | 30min | 3× assertion variants |
| 6 | Run E1 (360 calls) | 30min + ~$1.50 | `results/raw_E1.jsonl` |
| 7 | Run E2 (1,080 calls) | 1h + ~$4.50 | `results/raw_E2.jsonl` |
| 8 | Compute metrics + generate figures | 1h | `results/metrics_*.json`, plots |
| 9 | Write paper/blog | 4h | `docs/paper.md` |
| **Total** | | **~2 days + ~$9** | |

---

## 9. References

- LENS (2025): P→U→R hierarchy, 85%→69%→54% accuracy drop
- MMBench-GUI (2025): Content→Grounding→Automation, 79%→74%→27%
- BloomVQA (2023): Bloom's taxonomy for VQA, 38% drop across levels
- Cognitive Mismatch (2026): Recognition-reasoning inversion in MLLMs
- AMEX (2024): 110 apps, 103K screenshots, element functionality annotations
- MLLM-as-a-Judge (ICML 2024): Position bias, calibration gaps
- VisualWebBench (2024): Resolution sensitivity, granularity levels
- arXiv:2503.01747: Don't use CLT for LLM evaluation with small N — use bootstrap
- Lee & Zeng (2025): Bias-corrected CIs for LLM-judge evaluation
