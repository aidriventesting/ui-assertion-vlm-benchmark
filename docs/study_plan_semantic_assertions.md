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

## 4. Annotation Schema

Every data point in the benchmark carries **three levels of annotation**: per-screenshot, per-assertion, and per-evaluation-result. This is critical for meaningful stratified analysis.

### 4.1. Screenshot-Level Annotations

Annotated **once per screenshot** (100 screenshots = 100 annotations). These are covariates — they don't change between experiments but may explain variance.

| Field | Type | Values | Source | Why it matters |
|-------|------|--------|--------|----------------|
| `screen_id` | str | `amex_{app}_{idx}` | auto | Primary key |
| `app_name` | str | e.g. "Amazon", "Spotify" | AMEX metadata | App-level analysis |
| `app_category` | categorical | `shopping`, `social`, `finance`, `productivity`, `media`, `travel`, `health`, `settings`, `communication`, `food`, `other` | AMEX metadata or manual | Performance may vary by domain (shopping UIs are more standardized than games) |
| `screen_type` | categorical | `login`, `home`, `list`, `detail`, `settings`, `form`, `search`, `confirmation`, `error`, `modal`, `navigation`, `other` | Manual or LLM-assisted | Screen type may correlate with difficulty — forms are easier than modals |
| `n_elements` | int | 5-100+ | AMEX element_anno count | **Visual complexity proxy** — more elements = harder to find the right one |
| `n_text_elements` | int | 0-50+ | count elements with text | OCR load — many text elements = more confusables |
| `n_interactive_elements` | int | 0-30+ | count tappable elements | State/role assertions are harder with many interactive elements |
| `visual_density` | ordinal | `sparse` (≤10 elem), `moderate` (11-25), `dense` (26-50), `cluttered` (>50) | derived from n_elements | Binned version for stratification |
| `has_scrollable_content` | bool | true/false | AMEX view hierarchy or manual | Partial visibility affects absence assertions |
| `dominant_color_scheme` | categorical | `light`, `dark`, `mixed` | image analysis or manual | Dark mode may affect OCR/element detection |
| `device` | str | "Pixel 7 Pro", "Samsung S10" | AMEX metadata | Device variation (resolution, density) |
| `resolution` | str | "1080x2400" etc. | AMEX image metadata | Covariate for resolution sensitivity analysis |
| `language` | str | "en", "es", "mixed" | manual | Multilingual screens may confuse models |

**Annotation effort**: ~80% auto-extractable from AMEX metadata. Manual: `screen_type`, `has_scrollable_content`, `dominant_color_scheme`, `language` (~30 sec/screenshot = ~50 min total).

### 4.2. Assertion-Level Annotations

Annotated **once per test assertion** (550 assertions). These define what we're measuring.

| Field | Type | Values | Source | Why it matters |
|-------|------|--------|--------|----------------|
| `test_id` | str | `amex_{screen}_{tier}_{idx}` | auto | Primary key |
| `assertion` | str | Natural language | generated | The test input |
| `expected` | str | `PASS`, `FAIL` | generated + validated | Ground truth |
| `cognitive_tier` | categorical | `P`, `U`, `R` | derived from tags | **Primary IV** |
| **Operation tags** | | | | |
| `operation` | categorical | see §4.3 | manual/auto | What cognitive skill is required |
| **Difficulty tags** | | | | |
| `difficulty_tags` | list[str] | 0..N from set | manual | Visual/perceptual challenge modifiers |
| **Assertion characteristics** | | | | |
| `assertion_length` | int | word count | auto | Longer assertions may be harder to parse |
| `assertion_specificity` | ordinal | `vague`, `moderate`, `precise` | manual | *"Check the header"* vs *"Verify the header says 'Cart (3)'"* |
| `assertion_phrasing` | categorical | `imperative`, `interrogative`, `declarative` | auto (regex) | Phrasing style as potential confound |
| `n_visual_cues_required` | int | 1, 2, 3+ | manual | How many distinct screen regions must be checked |
| `requires_domain_knowledge` | bool | true/false | manual | *"Verify the price includes tax"* needs shopping domain knowledge |
| **Provenance** | | | | |
| `generation_method` | categorical | `auto_from_element`, `llm_generated`, `llm_perturbed`, `manual` | auto | Track how the assertion was created |
| `perturbation_type` | categorical | `none`, `char_swap`, `case`, `word_drop`, `number`, `antonym`, `semantic_inversion`, `absence_inversion` | auto | For FAIL assertions: what was changed |
| `source_element_id` | str | AMEX element ID | auto | Traceability to source annotation |
| `validated_by` | str | `human`, `auto` | manual | Was ground truth human-verified? |
| `validation_confidence` | ordinal | `certain`, `probable`, `ambiguous` | manual | Annotator's confidence in the label |

**Annotation effort**: Auto fields (~60%) are computed. Manual fields for U/R tier assertions (~350): `specificity`, `n_visual_cues`, `requires_domain_knowledge`, `validation_confidence` (~45 sec/assertion = ~4.5h).

### 4.3. Operation Tags (v5 — tier-aligned)

```yaml
# Tier P — Perception
text_match_exact: "Character-for-character string match"
text_match_normalized: "Match after case/whitespace normalization"
presence: "Element exists in viewport"
absence: "Element does NOT exist"
icon_recognition: "Identify an icon by its visual appearance"
color_check: "Verify an element's color/style"

# Tier U — Understanding
element_role: "Identify what a UI element does (not what it says)"
screen_type: "Classify the screen's purpose (login, settings, checkout)"
widget_state: "Understand interactive state (enabled, selected, expanded)"
navigation: "Identify navigation affordances (back, menu, tabs)"
grouping: "Understand that elements form a logical group (e.g., a form)"
content_type: "Classify what kind of content is displayed (image, map, video)"

# Tier R — Reasoning
task_completion: "Judge whether a multi-step task goal was achieved"
error_detection: "Identify error states or failure conditions"
data_consistency: "Cross-reference multiple UI values for coherence"
workflow_state: "Determine where in a workflow the user currently is"
conditional_logic: "If X is true, then Y should be visible/enabled"
constraint_check: "Verify a business rule (e.g., can't checkout with empty cart)"
```

### 4.4. Difficulty Tags

```yaml
# Visual challenges
near_miss: "Minimal visual difference from correct state"
small_text: "Text < ~10px rendered, requires fine OCR"
low_contrast: "Poor foreground/background distinction"
cluttered: "Dense layout with competing elements"
occluded: "Element partially hidden, cropped, or overlapping"
confusable: "Multiple visually similar elements"
truncated: "Text cut off with '...' or overflow hidden"
# Semantic challenges
ambiguous_element: "Element could be interpreted multiple ways"
multi_region: "Answer requires scanning multiple screen areas"
implicit_state: "State is implied, not explicitly shown (e.g., no error = success)"
```

### 4.5. Evaluation-Result-Level Fields

These are recorded **per API call** (each assertion × prompt × model = 1 result). They come from `run_eval.py` output.

| Field | Source | Notes |
|-------|--------|-------|
| `result` | model output | PASS/FAIL/UNCLEAR |
| `confidence` | model output (json) or logprobs (abc) | 0.0-1.0 |
| `evidence` | model CoT output | What the model cited as proof |
| `reasoning` | model CoT output | How it reached the verdict |
| `latency_ms` | measured | Response time |
| `input_tokens` | API response | Token count for cost analysis |
| `output_tokens` | API response | Token count |
| `cost` | computed | USD cost per call |

---

## 5. Independent Variables — Full Matrix

### 5.1. Test-Level IVs (vary per assertion, within same experiment)

| IV | Type | Values | Controlled how |
|----|------|--------|---------------|
| `cognitive_tier` | categorical | P, U, R | Assertion tag |
| `operation` | categorical | 16 operation tags | Assertion tag |
| `difficulty` | categorical | 0..N difficulty tags | Assertion tag |
| `assertion_specificity` | ordinal | vague, moderate, precise | Assertion annotation |
| `n_visual_cues_required` | int | 1, 2, 3+ | Assertion annotation |
| `screen_complexity` (n_elements) | continuous | 5-100+ | Screenshot annotation |
| `app_category` | categorical | 11 categories | Screenshot annotation |
| `expected_polarity` | binary | PASS, FAIL | Assertion label |

These are **observed** — we don't control them experimentally but stratify results by them.

### 5.2. Experiment-Level IVs (vary between experiments)

| IV | Values | Rationale | Priority |
|----|--------|-----------|----------|
| **system_prompt** | `minimal`, `evidence_first`, `strict_oracle` | Prompt strategy affects reasoning | HIGH — primary ablation |
| **model** | `gpt-4o-mini`, `gpt-4o`, `gemini-2.0-flash`, `claude-sonnet-4` | Model capability ceiling | HIGH — cross-model comparison |
| **output_format** | `json` (CoT, 500 tokens), `abc` (1 token, logprobs) | CoT vs direct; enables calibration | HIGH — unlocks Brier/ECE |
| **assertion_phrasing** | `imperative` ("Verify that..."), `interrogative` ("Is the...?"), `declarative` ("The cart is empty") | Phrasing may shift model behavior | MEDIUM — cheap to test |
| **few_shot** | `0-shot`, `1-shot` (tier-matched exemplar), `3-shot` | Exemplars may anchor reasoning | MEDIUM — especially for R-tier |
| **image_preprocessing** | `original`, `resized_50%`, `element_highlighted` (bbox overlay) | Resolution sensitivity; highlighting may help R-tier | MEDIUM — VisualWebBench finding |
| **multi_sample** | `single`, `majority_vote_3`, `majority_vote_5` | Trading cost for reliability | LOW — expensive, but key for deployment |
| **temperature** | `0.0`, `0.7` | Only relevant for multi_sample | LOW — near-null effect in literature |

### 5.3. Experiment Matrix (expanded)

| ID | What it tests | Model | Prompt | Format | Phrasing | Few-shot | Image | Multi | Calls |
|----|--------------|-------|--------|--------|----------|----------|-------|-------|-------|
| **E1** | Baseline × prompts | gpt-4o-mini | 3 personas | json | imperative | 0 | original | 1 | 1,650 |
| **E2** | Strong model | gpt-4o | evidence_first | json | imperative | 0 | original | 1 | 550 |
| **E3** | Calibration | gpt-4o-mini | evidence_first | abc | imperative | 0 | original | 1 | 550 |
| **E4** | Few-shot | gpt-4o-mini | evidence_first | json | imperative | 1-shot | original | 1 | 550 |
| **E5** | Phrasing | gpt-4o-mini | evidence_first | json | 3 styles | 0 | original | 1 | 1,650 |
| **E6** | Resolution | gpt-4o-mini | evidence_first | json | imperative | 0 | 3 sizes | 1 | 1,650 |
| **E7** | Multi-sample | gpt-4o-mini | evidence_first | json | imperative | 0 | original | mv3 | 1,650 |
| **E8** | Cross-model | gemini-2.0-flash | evidence_first | json | imperative | 0 | original | 1 | 550 |
| **E9** | Cross-model | claude-sonnet-4 | evidence_first | json | imperative | 0 | original | 1 | 550 |
| | | | | | | | | **Total** | **~9,350** |

**Cost estimate**: ~$25-40 depending on models (gpt-4o is the most expensive at ~$7.50/1K images).

**Execution strategy**: Run E1 first (baseline), analyze, then prioritize E2-E9 based on findings.

---

## 6. Metrics Framework

### 6.1. Primary DVs (same convention: POSITIVE = BUG)

| Metric | What it measures | Target |
|--------|-----------------|--------|
| FNR (Miss Rate) | Bugs that escape | < 5% |
| FPR (False Alarm) | Test flakiness | < 15% |
| Balanced Accuracy | Robust overall | > 85% |
| MCC | Best single metric for imbalanced binary | > 0.7 |

### 6.2. Stratified Analysis (the key analyses)

#### A. By Cognitive Tier (primary research question)

```
For each tier T in {P, U, R}:
  - FNR_T, FPR_T, Acc_T, MCC_T
  - Bootstrap 95% CI (1000 iter)
  - McNemar's test: P vs U, U vs R, P vs R (within same prompt)
  - Effect size: Cohen's d for tier differences
```

**Expected**: FNR_R >> FNR_P. If FNR_R ≈ FNR_P, the model uses linguistic shortcuts.

#### B. By Screen Complexity (confound check)

```
For each visual_density bin {sparse, moderate, dense, cluttered}:
  - Same metrics
  - Cross with tier: is complexity orthogonal to tier, or confounded?
```

**Expected**: Dense screens hurt P-tier (more text to scan) but not necessarily R-tier (reasoning doesn't depend on element count).

#### C. By App Category

```
For each app_category:
  - FNR, FPR, Acc
  - Identify categories where VLMs struggle (games? maps? custom UI?)
```

**Expected**: Standardized UIs (settings, forms) are easier; creative/custom UIs (games, media players) are harder.

#### D. By Assertion Characteristics

```
For each specificity level {vague, moderate, precise}:
  - Is precision good? (vague assertions → more FP? more FN?)

For each n_visual_cues bin {1, 2, 3+}:
  - Does multi-cue requirement degrade performance?

For each domain_knowledge {true, false}:
  - Does domain knowledge requirement hurt more than visual complexity?
```

#### E. Prompt × Tier Interaction (2-way)

```
For each (prompt, tier) combination:
  - FNR, FPR
  - Interaction test: does evidence_first help more on R-tier than P-tier?
```

#### F. Phrasing Effect (E5)

```
For each (phrasing, tier):
  - Does interrogative framing reduce overconfidence?
  - Does declarative framing increase false PASSes?
```

### 6.3. Calibration (per tier, requires E3)

```
For each tier:
  - Brier score
  - ECE (10 bins) + reliability diagram data
  - Overconfidence ratio
  - Per-bin: (avg_confidence, avg_accuracy, count)
```

**Expected**: Overconfidence_R >> Overconfidence_P (models are confident but wrong on reasoning).

### 6.4. Cost-Effectiveness Analysis

```
For each experiment:
  - Cost per correct decision (USD)
  - Cost per caught bug (USD)
  - Cost per avoided false alarm (USD)
  - ROI vs human QA baseline (estimate human at $30/h, ~2 min/assertion)
```

### 6.5. Statistical Rigor

| Method | Used for | When |
|--------|----------|------|
| Bootstrap 95% CI (1000 iter) | All metrics | Always |
| McNemar's test | Paired prompt/model comparison | E1 (3 prompts), E5 (3 phrasings), E8-E9 (cross-model) |
| Cohen's κ | Inter-prompt/model agreement | Same as McNemar |
| Bonferroni correction | Multiple comparison control | When testing >3 pairs |
| Cohen's d | Effect size for significant differences | When McNemar p < 0.05 |
| Spearman ρ | Correlation of continuous covariates | screen_complexity × accuracy, assertion_length × accuracy |

---

## 7. Annotation Protocol

### Phase 1: Data Collection (automated, ~2h)

1. Download AMEX from HuggingFace (`Yuxiang007/AMEX`)
2. Sample 100 diverse screenshots:
   - Stratified by app_category (at least 5 per category for top-8 categories)
   - Stratified by n_elements (25 sparse + 25 moderate + 25 dense + 25 cluttered)
   - Deduplicate similar screens from same app
3. Extract screenshot-level metadata from AMEX annotations

### Phase 2: Assertion Generation (semi-automated, ~4h)

1. **P-tier** (auto): Extract element text → generate exact-match assertions + perturbations
   - Target: 200 assertions (120 PASS + 80 FAIL)
   - Auto-tag: `text_match_exact`, `presence`, `absence`
   - Auto-tag perturbation: `char_swap`, `case`, `number`, etc.

2. **U-tier** (LLM + human): Feed screenshot + element annotations to GPT-4o
   - Generate 3 understanding assertions per screen (pick best 2)
   - Target: 200 assertions (120 PASS + 80 FAIL)
   - Generate FAIL via semantic inversion, role confusion, state mismatch

3. **R-tier** (LLM + human): Feed task instruction + final screenshot to GPT-4o
   - Generate 2 reasoning assertions per task
   - Target: 150 assertions (85 PASS + 65 FAIL)
   - Use TASK_COMPLETE/IMPOSSIBLE labels as ground truth anchor

### Phase 3: Human Validation (manual, ~5h)

**Annotation guide** for each U/R assertion:

```
For each assertion, answer:

1. AMBIGUITY CHECK: Can a human determine PASS/FAIL from the screenshot alone?
   [ ] Yes, clearly    [ ] Probably    [ ] Ambiguous → REJECT

2. LABEL CHECK: Is the expected label correct?
   [ ] Correct    [ ] Wrong → FIX    [ ] Can't tell → REJECT

3. TIER CHECK: Does this assertion genuinely require the tagged tier?
   - P: Could be answered by text search / OCR alone
   - U: Requires understanding element function, not just reading
   - R: Requires cross-referencing multiple cues or inferring state
   [ ] Correct tier    [ ] Should be {P/U/R} → RE-TAG

4. SPECIFICITY: How specific is the assertion?
   [ ] Vague (many interpretations)
   [ ] Moderate (some room for interpretation)
   [ ] Precise (only one way to check)

5. VISUAL CUES: How many distinct screen regions must be checked?
   [ ] 1    [ ] 2    [ ] 3+

6. DOMAIN KNOWLEDGE: Does answering require knowledge beyond what's on screen?
   [ ] No    [ ] Yes (describe: ________________)

7. CONFIDENCE: How confident are you in your annotation?
   [ ] Certain    [ ] Probable    [ ] Unsure
```

**Quality targets**:
- Inter-annotator agreement: κ > 0.7 (if using 2 annotators on a subset)
- Rejection rate: < 20% of generated assertions
- After validation: minimum 150 P + 150 U + 100 R = 400 usable assertions

### Phase 4: Balance & Diversity Check

| Dimension | Target | Check |
|-----------|--------|-------|
| PASS/FAIL ratio | 55-60% / 40-45% | Per tier |
| Per tier | ≥ 100 assertions after validation | Count |
| Per app category (top-8) | ≥ 5 assertions each | Count |
| Specificity distribution | ≥ 20% vague, ≥ 20% precise | Per tier |
| n_visual_cues | ≥ 30% multi-cue for U/R | Count |
| Screen complexity | ≥ 15 per density bin | Count |

If imbalanced, generate additional targeted assertions for underrepresented cells.

---

## 8. Expected Conclusions

Based on the literature, we expect to show:

### H1: Steep cognitive tier gradient
Accuracy drops 15-25% from P→R, confirming VLMs are visual grounding tools, not reasoning engines (consistent with LENS, MMBench-GUI).

### H2: Prompt × tier interaction
CoT/evidence_first helps on R-tier (forces step-by-step reasoning) but adds noise on P-tier (overthinks simple checks). Strict_oracle may show lowest FNR_R but highest FPR_R.

### H3: Calibration gap widens with tier
Models are well-calibrated on P-tier but overconfident on R-tier (Brier_R >> Brier_P). This means confidence thresholds are reliable for P/U but not R.

### H4: FNR deployment threshold varies by tier
- FNR_P < 5% → deployable without human oversight
- FNR_U ~ 8-12% → deployable with confidence-based routing to human review
- FNR_R > 15% → not deployable as autonomous oracle; useful as assistant

### H5: Cross-model divergence increases with tier
gpt-4o-mini ≈ gpt-4o on P-tier, but gpt-4o >> gpt-4o-mini on R-tier. Reasoning capability matters more at higher tiers.

### H6: Phrasing effect
Interrogative phrasing (*"Is the cart empty?"*) reduces overconfidence on R-tier assertions. Declarative phrasing (*"The cart is empty"*) increases false PASS rate.

### H7: Resolution sensitivity
Downscaled images hurt P-tier disproportionately (OCR degrades) but barely affect R-tier (reasoning is resolution-independent once elements are recognizable).

### H8: Screen complexity is a confound, not an explanation
Visual density correlates with difficulty, but does not explain the tier gradient — R-tier assertions on sparse screens are still harder than P-tier on cluttered screens.

### Practical Implications for Test Automation

| Tier | Recommendation | Confidence threshold |
|------|---------------|---------------------|
| **P** | Deploy autonomously. VLM replaces manual visual checks. | conf > 0.7 → auto-decide |
| **U** | Deploy with routing. Low-confidence → human review. | conf > 0.85 → auto, else human |
| **R** | Assistive only. VLM flags potential issues, human decides. | Always review |

---

## 9. Implementation Roadmap

| Step | Task | Effort | Dependency |
|------|------|--------|------------|
| 1 | Write `convert_amex.py` — download + convert AMEX to benchmark format | 1 day | HuggingFace access |
| 2 | Generate P-tier assertions from element annotations | 2h | Step 1 |
| 3 | Generate U/R-tier assertions via GPT-4o | 3h + ~$5 API | Step 1 |
| 4 | Screenshot-level annotation (auto + manual) | 1.5h | Step 1 |
| 5 | Human validation of U/R assertions (Phase 3) | 5h manual | Step 3 |
| 6 | Balance check + gap filling | 1h | Step 5 |
| 7 | Run E1 (baseline, 1,650 calls) | 1h + ~$3 API | Step 6 |
| 8 | Analyze E1, decide priority for E2-E9 | 1h | Step 7 |
| 9 | Run prioritized experiments (E2-E9) | 3h + ~$25 API | Step 8 |
| 10 | Compute metrics + generate full report | 1h | Step 9 |
| 11 | Write findings + practical recommendations | 1 day | Step 10 |

**Total estimated effort**: 4-5 days + ~$35 API costs

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
- VisualWebBench (2024): Resolution sensitivity, 7 tasks across 3 granularity levels
- PaliGemma 2 (2024): Resolution × model size interaction ablations
- Renze & Guven (2024): Temperature has near-null effect on accuracy 0.0-1.0
- Lee & Zeng (2025): Bias-corrected CIs for LLM-judge evaluations
- VHELM (NeurIPS 2024): 22 VLMs, 9 aspects — efficiency vs full model gaps on bias
