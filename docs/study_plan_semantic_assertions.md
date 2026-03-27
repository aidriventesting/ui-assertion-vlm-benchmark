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
| **RQ5** | At what confidence threshold should you trust the VLM vs route to a human? | `logprobs` (confidence) | "Set threshold at X% — below that, send to human review" |
| **RQ6** | Is the model's confidence well-calibrated, and does it vary by tier? | `calibration × tier` | "Trust confidence scores on P-tier, ignore them on R-tier" |
| **RQ7** | Does allowing the model to say "UNCERTAIN" improve reliability on decided cases? | `verdict_options` (binary vs ternary) | "Let the model abstain — it routes X% to human but FNR drops by Y%" |

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
| Verdict options | binary (PASS/FAIL) | Baseline — no abstention allowed |
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

#### Calibration DVs (from logprobs — E3 only)

| DV | What it means for test auto |
|----|-----------------------------|
| **Brier Score** | Overall quality of confidence estimates (lower = better, 0.25 = random) |
| **ECE** (10 bins) | Expected Calibration Error — are confidence scores trustworthy? |
| **Overconfidence Ratio** | % of bins where model is more confident than accurate |
| **Optimal Threshold** | Confidence cutoff that maximizes F1 (the deployment setting) |
| **Reliability Diagram** | Per-bin (confidence, accuracy, count) — visual calibration check |
| **FNR@threshold** | Miss rate when filtering below confidence threshold |
| **Human Routing Rate** | % of assertions routed to human at optimal threshold |

### Provider Constraint

**Logprobs are OpenAI-only.** Anthropic and Google do not expose token-level log-probabilities on vision API calls. This means:
- E1 (JSON/CoT) and E2 (phrasing): any provider works, but we use OpenAI for consistency
- E3 (ABC/logprobs): **must be OpenAI** (gpt-4o-mini with `logprobs=true, top_logprobs=5`)
- The ABC format sends 3 tokens (A=PASS, B=FAIL, C=UNCLEAR) and reads the log-probability of each — no chain-of-thought, just a calibrated probability

This is a real constraint for deployment: **confidence-based routing only works with OpenAI today.**

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

### E3: Logprobs & Calibration (RQ5 + RQ6)

| Parameter | Value |
|-----------|-------|
| Assertions | 360 (imperative phrasing only) |
| Model | gpt-4o-mini |
| Prompt | evidence_first |
| Format | **abc** (1 token, logprobs=true, top_logprobs=5) |
| API calls | **360** |
| Est. cost | ~$0.50 (1 token output per call) |

**Why a separate experiment**: JSON format gives reasoning traces (useful for debugging) but no calibrated confidence. ABC format gives `p_pass`, `p_fail`, `p_unclear` from logprobs — needed for threshold analysis.

**Analysis**:
- Reliability diagram: 10-bin plot of (confidence vs accuracy) → **visual calibration check**
- ECE and Brier score, overall and per tier → **RQ6**
- Sweep confidence thresholds 0.5→0.99:
  - At each threshold t: compute FNR, FPR, and human_routing_rate for decisions where max(p_pass, p_fail) ≥ t
  - Find **optimal t** that maximizes F1 on decided cases → **RQ5**
  - Find **safe t** where FNR < 5% → the deployment threshold
- Key output: **routing curve** (threshold vs FNR vs human_routing_rate per tier)

**Expected findings**:

| Tier | Optimal threshold | FNR@optimal | Human routing rate | Interpretation |
|------|------------------|-------------|-------------------|---------------|
| P | ~0.75 | ~3% | ~10% | Mostly autonomous, few edge cases |
| U | ~0.85 | ~8% | ~25% | Viable with routing |
| R | ~0.90 | ~15% | ~45% | Half the assertions need human review |

**Insight for test auto**: "For P-tier assertions, auto-decide when confidence > 75%. For U-tier, require 85%. For R-tier, even at 90% threshold you're still routing 45% to humans — consider if VLM adds value vs direct human review."

### E4: Binary vs Ternary — Does abstention improve reliability? (RQ7)

| Parameter | Value |
|-----------|-------|
| Assertions | 360 (imperative phrasing only) |
| Model | gpt-4o-mini |
| Prompt | evidence_first, **modified to allow UNCERTAIN** |
| Format | json (CoT) |
| API calls | **360** |
| Est. cost | ~$1.50 |

**Design**: Run the same 360 assertions with a modified prompt that adds UNCERTAIN as a third option:

```
Binary (E1):    {"result": "PASS"} or {"result": "FAIL"}
Ternary (E4):   {"result": "PASS"} or {"result": "FAIL"} or {"result": "UNCERTAIN"}
```

The prompt for E4 adds:
```
If you cannot determine with reasonable confidence whether the assertion passes
or fails based on the screenshot, respond with {"result": "UNCERTAIN"}.
Only use UNCERTAIN when genuinely unsure — not as a default.
```

**Analysis**:

```
From E4:
  abstain_rate = |UNCERTAIN| / |total|         → how often does it abstain?
  abstain_rate_P, abstain_rate_U, abstain_rate_R → does it abstain more on R?

On decided cases only (PASS/FAIL):
  FNR_decided, FPR_decided, MCC_decided

Compare E1 (binary) vs E4 (ternary, decided only):
  Δ_FNR = FNR_E1 - FNR_E4_decided   → did abstention improve miss rate?
  Δ_FPR = FPR_E1 - FPR_E4_decided   → did abstention reduce false alarms?
  McNemar on decided pairs

Key question per tier:
  Is routing via UNCERTAIN better than routing via confidence threshold (RQ5)?
```

**Two routing strategies compared**:

| Strategy | How it routes to human | Metric |
|----------|----------------------|--------|
| **Confidence routing** (E3) | `max(p_pass, p_fail) < threshold` | continuous, tunable |
| **Abstention routing** (E4) | Model says UNCERTAIN | binary, model-decided |

If abstention routing ≈ confidence routing in FNR/routing rate → simpler to deploy (no logprobs needed, works with any provider).
If confidence routing >> abstention routing → logprobs are worth the OpenAI lock-in.

**Expected findings**:

| Tier | Abstain rate | FNR (binary E1) | FNR (ternary decided) | Improvement |
|------|-------------|-----------------|----------------------|-------------|
| P | ~5% | ~4% | ~2% | Small — model is already confident |
| U | ~15% | ~10% | ~6% | Moderate — UNCERTAIN catches edge cases |
| R | ~30% | ~20% | ~12% | Large — model knows when it's guessing |

**Insight for test auto**: "Adding UNCERTAIN as an option reduces FNR by ~40% on decided cases, at the cost of routing 15-30% to human review. This is the cheapest way to improve reliability — no logprobs, no multi-sample, works with any provider."

### Total

| | Calls | Cost |
|-|-------|------|
| E1 (JSON, binary baseline) | 360 | ~$1.50 |
| E2 (JSON, phrasing ×3) | 1,080 | ~$4.50 |
| E3 (ABC, logprobs binary) | 360 | ~$0.50 |
| E4 (JSON, ternary UNCERTAIN) | 360 | ~$1.50 |
| **Total** | **2,160** | **~$8** |

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

### 6.5. RQ5 — Optimal Confidence Threshold

```
From E3 (ABC logprobs):
For each threshold t ∈ {0.50, 0.55, 0.60, ..., 0.95, 0.99}:
  decided = results where max(p_pass, p_fail) ≥ t
  routed  = results where max(p_pass, p_fail) < t  → sent to human
  FNR_t   = FNR on decided subset
  FPR_t   = FPR on decided subset
  F1_t    = F1 on decided subset
  routing_rate_t = |routed| / |total|

Find:
  t_optimal = argmax_t F1_t
  t_safe    = min t such that FNR_t < 0.05

Stratify by tier: t_optimal_P, t_optimal_U, t_optimal_R
```

This produces the **routing curve** — the key deployment artifact:

```
X-axis: threshold (0.5 → 1.0)
Y-axis (left): FNR (want low)
Y-axis (right): human routing rate (want low)
Lines: one per tier (P=green, U=orange, R=red)
Annotation: mark t_optimal and t_safe on each line
```

**Expected**: P-tier has a wide "sweet spot" (low FNR + low routing from 0.7-0.9). R-tier curve is flat — no threshold gives both low FNR and low routing.

**Insight for test auto**: Concrete deployment config:
```yaml
# Recommended VLM test oracle settings
confidence_thresholds:
  perception_assertions: 0.75    # auto-decide above this
  understanding_assertions: 0.85
  reasoning_assertions: 0.90     # but expect 40%+ human routing

routing:
  below_threshold: "send_to_human_review"
  above_threshold: "auto_decide"
```

### 6.6. RQ6 — Calibration by Tier

```
From E3 (ABC logprobs):
For each tier T:
  Brier_T = mean((confidence - correct)^2)
  ECE_T   = expected calibration error (10 bins)
  overconfidence_ratio_T = |bins where conf > acc| / |nonempty bins|

Compare: Brier_P vs Brier_U vs Brier_R
```

Reliability diagram per tier (3 subplots):
```
X-axis: predicted confidence (10 bins)
Y-axis: observed accuracy
Diagonal = perfect calibration
Bars = histogram of prediction count per bin
```

**Expected**:
- P-tier: well-calibrated (points close to diagonal), Brier ~0.10
- U-tier: slightly overconfident (points below diagonal at high confidence), Brier ~0.18
- R-tier: significantly overconfident (says 90% confident but only 65% accurate), Brier ~0.25

**Insight for test auto**: "Don't trust raw confidence scores on reasoning assertions — the model thinks it knows but it doesn't. Apply tier-specific calibration or use the routing thresholds from RQ5 instead."

### 6.7. RQ7 — Binary vs Ternary (Abstention)

```
From E1 (binary) and E4 (ternary):

Overall:
  abstain_rate_E4 = |UNCERTAIN| / 360
  FNR_E1 vs FNR_E4_decided (on PASS/FAIL only)
  FPR_E1 vs FPR_E4_decided
  McNemar on common decided pairs

Per tier:
  abstain_rate_T → does the model abstain more on R-tier? (expected: yes)
  FNR improvement per tier

Compare routing strategies:
  confidence_routing (E3, threshold t*) vs abstention_routing (E4):
  - At matched routing rates: which has lower FNR?
  - At matched FNR: which routes fewer to human?
```

**Expected**: Abstention is a cheap approximation of confidence routing. It catches ~70% of what threshold-based routing catches, with zero infrastructure (no logprobs, any provider).

**Insight**: "If you can't use logprobs (Anthropic, Gemini, self-hosted models), add UNCERTAIN to your prompt. You'll get ~70% of the benefit of confidence-based routing, at zero extra cost."

### 6.8. Combined Deployment Recommendation (the takeaway figure)

A single summary figure for the paper/blog:

```
┌────────────────────────────────────────────────────────────────────────┐
│              VLM Test Oracle Deployment Guide                          │
├─────────┬──────────┬──────────────────────┬───────────────────────────┤
│  Tier   │ FNR      │ With logprobs (OpenAI)│ Without logprobs (any)   │
├─────────┼──────────┼──────────────────────┼───────────────────────────┤
│ P       │ ~3%  ✅  │ conf > 0.75 → auto   │ Binary prompt → auto     │
│ U       │ ~10% ⚠️  │ conf > 0.85 → auto   │ Ternary → route UNCERTAIN│
│ R       │ ~20% ❌  │ conf > 0.90 → 45% human │ Ternary → ~30% human  │
└─────────┴──────────┴──────────────────────┴───────────────────────────┘

"Match assertion complexity to VLM capability.
 Use confidence routing (logprobs) or abstention routing (UNCERTAIN)
 to handle the gray zone. Both beat forcing a binary decision."
```

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

4. **Results** (3 pages)
   - RQ1: Tier gradient (bar chart with CIs)
   - RQ2: Phrasing × tier interaction (heatmap)
   - RQ3: App category ranking (horizontal bar chart)
   - RQ4: Complexity correlation (scatter plot)
   - RQ5: Routing curve — threshold vs FNR vs human rate (the deployment figure)
   - RQ6: Reliability diagrams per tier (calibration)

5. **Discussion** (1 page)
   - Deployment guide: threshold config per tier
   - When to trust, when to route, when to skip VLM
   - Cost-benefit: VLM + routing vs pure human QA
   - Limitations (single model, OpenAI-only for logprobs, English, AMEX apps)

6. **Conclusion** (0.5 page)
   - VLMs are P-ready, U-routable, R-assistive
   - Confidence routing is the key enabler — not just accuracy
   - Test teams should match assertion level to VLM capability AND set confidence thresholds

### Key Figures

| Fig | Type | Shows | RQ |
|-----|------|-------|----|
| Fig 1 | Bar chart + CI whiskers | FNR by tier (P < U < R) | RQ1 |
| Fig 2 | Grouped bar chart | FNR × phrasing × tier | RQ2 |
| Fig 3 | Horizontal bar chart | MCC by app category, sorted | RQ3 |
| Fig 4 | Scatter + regression | n_elements vs accuracy, colored by tier | RQ4 |
| **Fig 5** | **Routing curve** | **Threshold vs FNR vs human_rate, per tier** | **RQ5** |
| **Fig 6** | **Reliability diagram** | **3 subplots: P/U/R calibration** | **RQ6** |
| **Fig 7** | **Routing comparison** | **Confidence routing vs abstention routing: FNR vs human rate** | **RQ5 vs RQ7** |
| **Fig 8** | **Deployment guide** | **Summary table: tier → strategy → action** | All |
| Table 1 | Full metrics | All DVs × tier, with bootstrap CIs | RQ1 |
| Table 2 | McNemar results | Pairwise significance tests | RQ1-2 |
| **Table 3** | **Threshold sweep** | **t, FNR@t, FPR@t, F1@t, routing_rate per tier** | **RQ5** |
| **Table 4** | **Abstention analysis** | **abstain_rate, FNR_decided, FPR_decided per tier** | **RQ7** |

---

## 8. Implementation Roadmap

| Step | Task | Effort | Output |
|------|------|--------|--------|
| 1 | `convert_amex.py` — download, sample 60 screens, extract metadata | 4h | `dataset_amex/screenshots/`, `images.json` |
| 2 | Generate P-tier assertions (auto from element text) | 1h | 120 P assertions |
| 3 | Generate U/R-tier assertions (GPT-4o + templates) | 2h + ~$3 | 240 U/R assertions |
| 4 | Human validation of U/R + annotation | 3h | validated `tests.json` per screen |
| 5 | Generate phrasing variants (template-based) | 30min | 3× assertion variants |
| 6 | Run E1 — binary baseline (360 calls) | 30min + ~$1.50 | `results/raw_E1.jsonl` |
| 7 | Run E2 — phrasing ablation (1,080 calls) | 1h + ~$4.50 | `results/raw_E2.jsonl` |
| 8 | Run E3 — ABC logprobs (360 calls) | 20min + ~$0.50 | `results/raw_E3.jsonl` |
| 9 | Run E4 — ternary UNCERTAIN (360 calls) | 30min + ~$1.50 | `results/raw_E4.jsonl` |
| 10 | Compute metrics + calibration + threshold sweep + abstention analysis | 1h | `results/metrics_*.json` |
| 11 | Generate figures (8 figs + 4 tables) | 1.5h | `docs/figures/` |
| 12 | Write paper/blog | 4h | `docs/paper.md` |
| **Total** | | **~2.5 days + ~$11** | |

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
- Overconfidence in LLM-as-a-Judge (2025): ECE 0.11-0.43, float repr best calibration
- Calibrating LLM Judges (2025): Brier score loss for training calibrated uncertainty probes
