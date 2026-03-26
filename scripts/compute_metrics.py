#!/usr/bin/env python3
"""
Compute standard ML metrics for UI Assertion VLM Benchmark.

Convention: POSITIVE = BUG (expected FAIL)

Core Metrics:
- Confusion matrix (TP/FP/TN/FN with POSITIVE=BUG)
- FNR (Miss Rate) - CRITICAL: bugs that escape
- FPR (False Alarm Rate) - causes test flakiness
- FOR (False Omission Rate) - P(bug | PASS predicted)
- TPR/PPV/NPV/F1 - standard classification metrics
- balanced_accuracy, mcc - robust to class imbalance

Calibration (if confidence available):
- Brier Score, ECE (10 bins), reliability diagram data
- Overconfidence ratio

Statistical rigor:
- Bootstrap 95% CI (1000 iterations)
- McNemar's test for prompt-pair significance
- Cohen's kappa for inter-prompt agreement

Stratification by tags (v4 taxonomy):
- Operation tags (presence, text_match_*, count_*, state, layout, etc.)
- Difficulty tags (near_miss, small_text, cluttered, etc.)
- Perturbation type (char_swap, number, antonym, absence, etc.)
- Assertion source (ground_truth, perturbation, absence_inversion, unanswerable)
"""

from __future__ import annotations

import json
import math
import re
import argparse
from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass

RESULTS_DIR = Path(__file__).parent.parent / "results"
DATASET_DIR = Path(__file__).parent.parent / "dataset"


# =============================================================================
# V4 TAG DEFINITIONS (aligned with tags_v2.yaml)
# =============================================================================

OPERATION_TAGS = {
    "presence", "absence",
    "text_match_exact", "text_match_normalized", "text_match_semantic",
    "count_raw", "count_filtered",
    "state", "layout", "order",
    "consistency", "arithmetic", "comparison", "negation",
}

DIFFICULTY_TAGS = {
    "near_miss", "small_text", "low_contrast",
    "cluttered", "occluded", "confusable", "truncated",
}

PERTURBATION_TAGS = {
    "perturb_char_swap", "perturb_case", "perturb_word_drop",
    "perturb_number", "perturb_antonym", "perturb_absence",
}


def normalize_label(label: str | None) -> str | None:
    """Normalize result/expected labels to standard values."""
    if label is None:
        return None
    upper = str(label).upper().strip()
    if upper in ("PASS", "OK", "TRUE", "SUCCESS", "1"):
        return "PASS"
    if upper in ("FAIL", "FALSE", "FAILURE", "0"):
        return "FAIL"
    if upper in ("UNCLEAR", "ABSTAIN", "UNKNOWN", "UNSURE", "SKIP", "N/A"):
        return "UNCLEAR"
    return None


# =============================================================================
# CONFUSION MATRIX WITH STANDARD ML METRICS
# =============================================================================

@dataclass
class ConfusionMatrix:
    """Confusion matrix with POSITIVE = BUG convention.

    TP: predicted FAIL, expected FAIL (bug caught)
    FP: predicted FAIL, expected PASS (false alarm / flakiness)
    TN: predicted PASS, expected PASS (correct pass)
    FN: predicted PASS, expected FAIL (BUG ESCAPED!)
    """
    tp: int = 0
    fp: int = 0
    tn: int = 0
    fn: int = 0
    abstained: int = 0
    skipped_invalid: int = 0

    @property
    def total(self) -> int:
        return self.tp + self.fp + self.tn + self.fn

    @property
    def total_positive(self) -> int:
        return self.tp + self.fn

    @property
    def total_negative(self) -> int:
        return self.tn + self.fp

    @property
    def accuracy(self) -> float:
        return (self.tp + self.tn) / self.total if self.total > 0 else 0.0

    @property
    def tpr(self) -> float:
        return self.tp / self.total_positive if self.total_positive > 0 else 0.0

    @property
    def fnr(self) -> float:
        return self.fn / self.total_positive if self.total_positive > 0 else 0.0

    @property
    def tnr(self) -> float:
        return self.tn / self.total_negative if self.total_negative > 0 else 0.0

    @property
    def fpr(self) -> float:
        return self.fp / self.total_negative if self.total_negative > 0 else 0.0

    @property
    def ppv(self) -> float:
        pp = self.tp + self.fp
        return self.tp / pp if pp > 0 else 0.0

    @property
    def npv(self) -> float:
        pn = self.tn + self.fn
        return self.tn / pn if pn > 0 else 0.0

    @property
    def fdr(self) -> float:
        return 1.0 - self.ppv

    @property
    def for_(self) -> float:
        return 1.0 - self.npv

    @property
    def f1(self) -> float:
        p, r = self.ppv, self.tpr
        return 2 * p * r / (p + r) if (p + r) > 0 else 0.0

    @property
    def coverage(self) -> float:
        t = self.total + self.abstained
        return self.total / t if t > 0 else 0.0

    @property
    def abstain_rate(self) -> float:
        t = self.total + self.abstained
        return self.abstained / t if t > 0 else 0.0

    @property
    def fail_rate_decided(self) -> float:
        pp = self.tp + self.fp
        return pp / self.total if self.total > 0 else 0.0

    @property
    def balanced_accuracy(self) -> float:
        return (self.tpr + self.tnr) / 2

    @property
    def mcc(self) -> float:
        num = (self.tp * self.tn) - (self.fp * self.fn)
        denom = math.sqrt(
            (self.tp + self.fp) * (self.tp + self.fn) *
            (self.tn + self.fp) * (self.tn + self.fn)
        )
        return num / denom if denom > 0 else 0.0

    # Aliases
    precision = ppv
    recall = tpr
    sensitivity = tpr
    specificity = tnr
    miss_rate = fnr
    false_alarm_rate = fpr

    def to_dict(self) -> dict:
        return {
            "tp": self.tp, "fp": self.fp, "tn": self.tn, "fn": self.fn,
            "abstained": self.abstained, "skipped_invalid": self.skipped_invalid,
            "decided": self.total,
            "total_with_abstained": self.total + self.abstained,
            "accuracy": round(self.accuracy, 4),
            "tpr": round(self.tpr, 4),
            "fnr": round(self.fnr, 4),
            "tnr": round(self.tnr, 4),
            "fpr": round(self.fpr, 4),
            "ppv": round(self.ppv, 4),
            "npv": round(self.npv, 4),
            "for": round(self.for_, 4),
            "f1": round(self.f1, 4),
            "balanced_accuracy": round(self.balanced_accuracy, 4),
            "mcc": round(self.mcc, 4),
            "coverage": round(self.coverage, 4),
            "abstain_rate": round(self.abstain_rate, 4),
            "fail_rate_decided": round(self.fail_rate_decided, 4),
        }


# =============================================================================
# CALIBRATION METRICS
# =============================================================================

def compute_brier_score(predictions: list[tuple[float, int]]) -> float | None:
    """Brier score: mean squared error of probabilistic predictions.
    Lower is better. 0 = perfect, 0.25 = random coin flip."""
    if not predictions:
        return None
    total = sum((p - a) ** 2 for p, a in predictions)
    return round(total / len(predictions), 4)


def compute_ece(predictions: list[tuple[float, int]], n_bins: int = 10) -> dict | None:
    """Expected Calibration Error with full reliability diagram data.

    Returns dict with:
      - ece: scalar ECE value
      - bins: list of {bin_lower, bin_upper, avg_confidence, avg_accuracy, count}
      - overconfidence_ratio: fraction of bins where confidence > accuracy
      - underconfidence_ratio: fraction of bins where confidence < accuracy
    """
    if not predictions:
        return None

    bins = defaultdict(list)
    for prob, actual in predictions:
        bin_idx = min(int(prob * n_bins), n_bins - 1)
        bins[bin_idx].append((prob, actual))

    ece = 0.0
    total = len(predictions)
    bin_data = []
    overconf = 0
    underconf = 0
    n_nonempty = 0

    for i in range(n_bins):
        bin_preds = bins.get(i, [])
        entry = {
            "bin_lower": round(i / n_bins, 2),
            "bin_upper": round((i + 1) / n_bins, 2),
            "count": len(bin_preds),
        }
        if bin_preds:
            avg_conf = sum(p for p, _ in bin_preds) / len(bin_preds)
            avg_acc = sum(a for _, a in bin_preds) / len(bin_preds)
            gap = abs(avg_conf - avg_acc)
            ece += len(bin_preds) / total * gap
            entry["avg_confidence"] = round(avg_conf, 4)
            entry["avg_accuracy"] = round(avg_acc, 4)
            entry["gap"] = round(gap, 4)
            n_nonempty += 1
            if avg_conf > avg_acc + 0.01:
                overconf += 1
            elif avg_conf < avg_acc - 0.01:
                underconf += 1
        bin_data.append(entry)

    return {
        "ece": round(ece, 4),
        "n_bins": n_bins,
        "bins": bin_data,
        "overconfidence_ratio": round(overconf / n_nonempty, 2) if n_nonempty > 0 else None,
        "underconfidence_ratio": round(underconf / n_nonempty, 2) if n_nonempty > 0 else None,
        "n_predictions": total,
    }


# =============================================================================
# BOOTSTRAP CONFIDENCE INTERVALS
# =============================================================================

def bootstrap_ci(results: list[dict], metric_fn, n_iter: int = 1000,
                 alpha: float = 0.05, seed: int = 42) -> dict:
    """Compute bootstrap confidence interval for any metric.

    Args:
        results: list of result dicts
        metric_fn: function(list[dict]) -> float (e.g., lambda rs: build_confusion_matrix(rs).accuracy)
        n_iter: bootstrap iterations
        alpha: significance level (0.05 = 95% CI)

    Returns:
        {point: float, ci_lower: float, ci_upper: float, std: float}
    """
    import random
    rng = random.Random(seed)
    n = len(results)
    if n == 0:
        return {"point": 0.0, "ci_lower": 0.0, "ci_upper": 0.0, "std": 0.0}

    point = metric_fn(results)
    boot_values = []

    for _ in range(n_iter):
        sample = rng.choices(results, k=n)
        boot_values.append(metric_fn(sample))

    boot_values.sort()
    lo_idx = int(n_iter * alpha / 2)
    hi_idx = int(n_iter * (1 - alpha / 2))

    mean_boot = sum(boot_values) / len(boot_values)
    std_boot = math.sqrt(sum((v - mean_boot) ** 2 for v in boot_values) / len(boot_values))

    return {
        "point": round(point, 4),
        "ci_lower": round(boot_values[lo_idx], 4),
        "ci_upper": round(boot_values[hi_idx], 4),
        "std": round(std_boot, 4),
    }


def mcnemar_test(results_a: list[dict], results_b: list[dict]) -> dict | None:
    """McNemar's test for paired prompt comparison.

    Compares two prompts on the same test items. Returns chi2 statistic and p-value.
    Only meaningful when both prompts ran on identical test_ids.
    """
    # Build lookup by test_id
    a_by_id = {r["test_id"]: r for r in results_a}
    b_by_id = {r["test_id"]: r for r in results_b}

    common_ids = set(a_by_id) & set(b_by_id)
    if len(common_ids) < 10:
        return None

    # Count discordant pairs
    b_right_a_wrong = 0  # B correct, A wrong
    a_right_b_wrong = 0  # A correct, B wrong

    for tid in common_ids:
        ra = a_by_id[tid]
        rb = b_by_id[tid]
        a_correct = normalize_label(ra.get("result")) == normalize_label(ra.get("expected"))
        b_correct = normalize_label(rb.get("result")) == normalize_label(rb.get("expected"))

        if b_correct and not a_correct:
            b_right_a_wrong += 1
        elif a_correct and not b_correct:
            a_right_b_wrong += 1

    n_discordant = b_right_a_wrong + a_right_b_wrong
    if n_discordant == 0:
        return {"chi2": 0.0, "p_value": 1.0, "n_common": len(common_ids),
                "n_discordant": 0, "significant": False}

    # McNemar's chi-squared (with continuity correction)
    chi2 = (abs(b_right_a_wrong - a_right_b_wrong) - 1) ** 2 / n_discordant

    # Approximate p-value from chi-squared distribution (1 df)
    # Using the complementary error function approximation
    p_value = math.exp(-chi2 / 2)  # rough approximation for 1 df

    return {
        "chi2": round(chi2, 4),
        "p_value": round(p_value, 4),
        "n_common": len(common_ids),
        "n_discordant": n_discordant,
        "b_right_a_wrong": b_right_a_wrong,
        "a_right_b_wrong": a_right_b_wrong,
        "significant": p_value < 0.05,
    }


def cohen_kappa(results_a: list[dict], results_b: list[dict]) -> float | None:
    """Cohen's kappa for inter-prompt agreement.

    Measures agreement beyond chance between two prompts' predictions.
    Range: -1 (complete disagreement) to 1 (perfect agreement). 0 = chance.
    """
    a_by_id = {r["test_id"]: r for r in results_a}
    b_by_id = {r["test_id"]: r for r in results_b}
    common_ids = set(a_by_id) & set(b_by_id)
    if len(common_ids) < 10:
        return None

    agree = 0
    total = 0
    a_fail_count = 0
    b_fail_count = 0

    for tid in common_ids:
        pred_a = normalize_label(a_by_id[tid].get("result"))
        pred_b = normalize_label(b_by_id[tid].get("result"))
        if pred_a is None or pred_b is None:
            continue
        total += 1
        if pred_a == pred_b:
            agree += 1
        if pred_a == "FAIL":
            a_fail_count += 1
        if pred_b == "FAIL":
            b_fail_count += 1

    if total == 0:
        return None

    p_o = agree / total  # observed agreement
    # Expected agreement by chance
    p_a_fail = a_fail_count / total
    p_b_fail = b_fail_count / total
    p_e = p_a_fail * p_b_fail + (1 - p_a_fail) * (1 - p_b_fail)

    if p_e == 1.0:
        return 1.0
    kappa = (p_o - p_e) / (1 - p_e)
    return round(kappa, 4)


# =============================================================================
# DATA LOADING
# =============================================================================

def load_results(results_path: Path = None) -> list[dict]:
    """Load evaluation results from raw.jsonl or specified file."""
    results = []
    if results_path:
        results_file = results_path
    else:
        results_file = RESULTS_DIR / "raw.jsonl"
        if not results_file.exists():
            raw_files = sorted(RESULTS_DIR.glob("raw_*.jsonl"), reverse=True)
            if raw_files:
                results_file = raw_files[0]
                print(f"Using latest results: {results_file.name}")
    if not results_file.exists():
        print(f"Warning: {results_file} not found")
        return []
    with open(results_file, "r") as f:
        for line in f:
            if line.strip():
                results.append(json.loads(line))
    return results


def load_test_metadata() -> dict[str, dict]:
    """Load test metadata from tests.json files in dataset screenshots dir."""
    tests = {}
    screenshots_dir = DATASET_DIR / "screenshots"
    if not screenshots_dir.exists():
        return {}
    for shot_dir in sorted(screenshots_dir.iterdir()):
        if shot_dir.is_dir():
            tests_file = shot_dir / "tests.json"
            if tests_file.exists():
                with open(tests_file, "r") as f:
                    shot_tests = json.load(f)
                    for test in shot_tests:
                        test_id = test.get("test_id")
                        if test_id:
                            tests[test_id] = test
    print(f"Loaded {len(tests)} test cases from {screenshots_dir}")
    return tests


def classify_assertion_source(test: dict) -> str:
    """Classify how the assertion was generated."""
    tags = set(test.get("tags", []))
    expected = test.get("expected", "")
    perturbation = test.get("_perturbation", "")

    if "absence" in tags and expected == "PASS":
        return "unanswerable"
    if perturbation:
        if "absence" in perturbation:
            return "absence_inversion"
        return "perturbation"
    if expected == "PASS":
        return "ground_truth"
    return "other"


# =============================================================================
# CORE ANALYSIS
# =============================================================================

def build_confusion_matrix(results: list[dict]) -> ConfusionMatrix:
    """Build confusion matrix from results. POSITIVE = BUG (expected FAIL)."""
    cm = ConfusionMatrix()
    for r in results:
        result = normalize_label(r.get("result"))
        expected = normalize_label(r.get("expected"))
        abstained = r.get("abstained", False)
        if abstained or result == "UNCLEAR" or result is None:
            cm.abstained += 1
            continue
        if expected not in ("PASS", "FAIL"):
            cm.skipped_invalid += 1
            continue
        if expected == "FAIL":
            if result == "FAIL":
                cm.tp += 1
            else:
                cm.fn += 1
        elif expected == "PASS":
            if result == "FAIL":
                cm.fp += 1
            else:
                cm.tn += 1
    return cm


def analyze_by_provider(results: list[dict]) -> dict:
    """Group results by provider and compute metrics."""
    by_provider = defaultdict(list)
    for r in results:
        by_provider[r.get("provider", "unknown")].append(r)
    return {
        provider: build_confusion_matrix(items).to_dict()
        for provider, items in by_provider.items()
    }


def stratify_results(results: list[dict], test_metadata: dict[str, dict]) -> dict:
    """Stratify results by all dimensions (v4 taxonomy)."""

    groups = {
        "by_profile": defaultdict(list),
        "by_screenshot": defaultdict(list),
        "by_polarity": defaultdict(list),
        "by_operation": defaultdict(list),
        "by_difficulty": defaultdict(list),
        "by_perturbation": defaultdict(list),
        "by_assertion_source": defaultdict(list),
    }

    for r in results:
        test_id = r.get("test_id")
        meta = test_metadata.get(test_id, {})

        # Merge expected from metadata if missing
        if "expected" not in r and "expected" in meta:
            r["expected"] = meta["expected"]

        tags = meta.get("tags", [])
        expected = r.get("expected", meta.get("expected", "unknown"))
        profile = r.get("profile", "unknown")
        image_id = r.get("image_id", meta.get("image_id", "unknown"))

        # Profile (system prompt)
        groups["by_profile"][profile].append(r)

        # Screenshot
        groups["by_screenshot"][image_id].append(r)

        # Polarity
        expected_norm = normalize_label(expected) or "unknown"
        groups["by_polarity"][expected_norm].append(r)

        # Operations
        has_op = False
        for tag in tags:
            if tag in OPERATION_TAGS:
                groups["by_operation"][tag].append(r)
                has_op = True
        if not has_op and tags:
            groups["by_operation"]["untagged_op"].append(r)

        # Difficulty
        has_diff = False
        for tag in tags:
            if tag in DIFFICULTY_TAGS:
                groups["by_difficulty"][tag].append(r)
                has_diff = True
        if not has_diff:
            groups["by_difficulty"]["baseline"].append(r)

        # Perturbation type
        for tag in tags:
            if tag in PERTURBATION_TAGS:
                groups["by_perturbation"][tag].append(r)

        # Assertion source
        source = classify_assertion_source(meta) if meta else "unknown"
        groups["by_assertion_source"][source].append(r)

    def compute_group_metrics(group_dict: dict) -> dict:
        return {
            name: build_confusion_matrix(items).to_dict()
            for name, items in group_dict.items()
            if items
        }

    return {dim: compute_group_metrics(grp) for dim, grp in groups.items()}


def compute_calibration(results: list[dict]) -> dict:
    """Compute calibration metrics from results with confidence/logprobs."""
    predictions = []

    for r in results:
        result = normalize_label(r.get("result"))
        expected = normalize_label(r.get("expected"))
        if result is None or expected is None:
            continue

        correct = 1 if result == expected else 0

        # Try logprobs first (ABC format)
        p_pass = r.get("p_pass")
        p_fail = r.get("p_fail")
        if p_pass is not None and p_fail is not None:
            # Confidence = probability assigned to the predicted class
            if result == "PASS":
                conf = 1.0 - p_pass if p_pass < 0.5 else p_pass
            else:
                conf = 1.0 - p_fail if p_fail < 0.5 else p_fail
            # Clamp to [0, 1]
            conf = max(0.0, min(1.0, conf))
            predictions.append((conf, correct))
            continue

        # Try JSON confidence field
        conf_raw = r.get("confidence")
        if conf_raw is not None:
            try:
                conf = float(conf_raw)
                if conf > 1.0:
                    conf = conf / 100.0  # Normalize 0-100 to 0-1
                predictions.append((max(0.0, min(1.0, conf)), correct))
            except (ValueError, TypeError):
                pass

    if not predictions:
        return {"available": False, "reason": "no confidence/logprobs in results"}

    brier = compute_brier_score(predictions)
    ece_data = compute_ece(predictions)

    return {
        "available": True,
        "n_predictions": len(predictions),
        "brier_score": brier,
        "ece": ece_data,
    }


def compute_prompt_comparisons(results: list[dict]) -> dict:
    """Compute pairwise statistical tests between prompts."""
    by_profile = defaultdict(list)
    for r in results:
        by_profile[r.get("profile", "unknown")].append(r)

    profiles = sorted(by_profile.keys())
    if len(profiles) < 2:
        return {}

    comparisons = {}
    for i, pa in enumerate(profiles):
        for pb in profiles[i + 1:]:
            key = f"{pa}_vs_{pb}"
            mcn = mcnemar_test(by_profile[pa], by_profile[pb])
            kappa = cohen_kappa(by_profile[pa], by_profile[pb])
            comparisons[key] = {
                "mcnemar": mcn,
                "cohen_kappa": kappa,
            }

    return comparisons


def compute_bootstrap_summary(results: list[dict], n_iter: int = 1000) -> dict:
    """Bootstrap CIs for key metrics."""
    metrics_fns = {
        "accuracy": lambda rs: build_confusion_matrix(rs).accuracy,
        "fnr": lambda rs: build_confusion_matrix(rs).fnr,
        "fpr": lambda rs: build_confusion_matrix(rs).fpr,
        "f1": lambda rs: build_confusion_matrix(rs).f1,
        "mcc": lambda rs: build_confusion_matrix(rs).mcc,
        "balanced_accuracy": lambda rs: build_confusion_matrix(rs).balanced_accuracy,
    }

    return {
        name: bootstrap_ci(results, fn, n_iter=n_iter)
        for name, fn in metrics_fns.items()
    }


# =============================================================================
# OUTPUT
# =============================================================================

def print_summary(analysis: dict, stratification: dict, calibration: dict = None,
                  comparisons: dict = None, bootstrap: dict = None):
    """Print formatted summary to console."""
    print("\n" + "=" * 70)
    print("UI ASSERTION VLM BENCHMARK — METRICS REPORT (v4)")
    print("Convention: POSITIVE = BUG (expected FAIL)")
    print("=" * 70)

    for provider, metrics in analysis.items():
        print(f"\n### Provider: {provider}")
        print("-" * 50)

        print(f"\nConfusion Matrix (POSITIVE = BUG):")
        print(f"  TP={metrics['tp']:3d}  FP={metrics['fp']:3d}  (TP=bug caught, FP=false alarm)")
        print(f"  FN={metrics['fn']:3d}  TN={metrics['tn']:3d}  (FN=BUG ESCAPED!, TN=correct pass)")
        print(f"  Abstained: {metrics['abstained']}  Skipped: {metrics['skipped_invalid']}")

        print(f"\n🔴 CRITICAL (deployment):")
        print(f"  FNR (Miss Rate):    {metrics['fnr']:.1%} (bugs escaped! want < 5%)")
        print(f"  FOR:                {metrics['for']:.1%} (P(bug|PASS) — hidden bugs)")

        print(f"\n🟡 Stability:")
        print(f"  FPR (False Alarm):  {metrics['fpr']:.1%} (causes flakiness)")
        print(f"  TPR (Recall):       {metrics['tpr']:.1%} (bug detection rate)")

        print(f"\n📊 Classification:")
        print(f"  Accuracy:           {metrics['accuracy']:.1%}")
        print(f"  Balanced Accuracy:  {metrics['balanced_accuracy']:.1%}")
        print(f"  MCC:                {metrics['mcc']:.3f}")
        print(f"  F1:                 {metrics['f1']:.1%}")
        print(f"  PPV (Precision):    {metrics['ppv']:.1%}")
        print(f"  NPV:                {metrics['npv']:.1%}")
        print(f"  Coverage:           {metrics['coverage']:.1%}")

    # Bootstrap CIs
    if bootstrap:
        print(f"\n{'=' * 70}")
        print("BOOTSTRAP 95% CONFIDENCE INTERVALS (1000 iterations)")
        print("=" * 70)
        for metric, ci in bootstrap.items():
            print(f"  {metric:<22} {ci['point']:.1%}  [{ci['ci_lower']:.1%}, {ci['ci_upper']:.1%}]  (σ={ci['std']:.3f})")

    # Calibration
    if calibration and calibration.get("available"):
        print(f"\n{'=' * 70}")
        print("CALIBRATION")
        print("=" * 70)
        print(f"  Brier Score:          {calibration['brier_score']}")
        ece = calibration["ece"]
        print(f"  ECE:                  {ece['ece']}")
        print(f"  Overconfidence ratio: {ece['overconfidence_ratio']}")
        print(f"  Predictions:          {ece['n_predictions']}")
        print(f"\n  Reliability Diagram (10 bins):")
        print(f"  {'Bin':<12} {'Conf':>8} {'Acc':>8} {'Gap':>8} {'N':>6}")
        print(f"  {'-'*44}")
        for b in ece["bins"]:
            if b["count"] > 0:
                print(f"  [{b['bin_lower']:.1f}-{b['bin_upper']:.1f})  "
                      f"{b['avg_confidence']:>7.3f}  {b['avg_accuracy']:>7.3f}  "
                      f"{b['gap']:>7.3f}  {b['count']:>5}")
    elif calibration:
        print(f"\n⚠️  Calibration: {calibration.get('reason', 'not available')}")

    # Prompt comparisons
    if comparisons:
        print(f"\n{'=' * 70}")
        print("PROMPT PAIRWISE COMPARISONS")
        print("=" * 70)
        for key, comp in comparisons.items():
            kappa = comp.get("cohen_kappa")
            mcn = comp.get("mcnemar")
            print(f"\n  {key}:")
            if kappa is not None:
                print(f"    Cohen's κ: {kappa:.3f} ({'substantial' if kappa > 0.6 else 'moderate' if kappa > 0.4 else 'fair' if kappa > 0.2 else 'slight'})")
            if mcn:
                sig = "***" if mcn["significant"] else "n.s."
                print(f"    McNemar χ²: {mcn['chi2']:.2f}, p={mcn['p_value']:.4f} {sig}")
                print(f"    Discordant pairs: {mcn['n_discordant']} / {mcn['n_common']}")

    # Stratification
    print(f"\n{'=' * 70}")
    print("STRATIFICATION ANALYSIS")
    print("=" * 70)

    for dim_name, dim_data in stratification.items():
        if dim_name == "by_screenshot":
            continue  # Too verbose for console
        print(f"\n### {dim_name.replace('by_', '').replace('_', ' ').title()}")
        print(f"  {'Group':<25} {'FNR':<8} {'FPR':<8} {'Acc':<8} {'MCC':<8} {'N':<6}")
        print(f"  {'-'*63}")
        for group, metrics in sorted(dim_data.items()):
            if metrics["decided"] > 0:
                print(f"  {group:<25} {metrics['fnr']:.1%}    "
                      f"{metrics['fpr']:.1%}    "
                      f"{metrics['accuracy']:.1%}    "
                      f"{metrics['mcc']:.3f}   {metrics['decided']}")


def extract_timestamp(filepath: Path) -> str | None:
    match = re.search(r'(\d{8}_\d{6})', filepath.name)
    return match.group(1) if match else None


def save_report(analysis: dict, stratification: dict, input_file: Path = None,
                calibration: dict = None, comparisons: dict = None,
                bootstrap: dict = None):
    """Save comprehensive JSON report."""
    timestamp = extract_timestamp(input_file) if input_file else None
    output = RESULTS_DIR / (f"metrics_{timestamp}.json" if timestamp else "metrics.json")

    report = {
        "meta": {
            "taxonomy_version": "v4",
            "convention": "POSITIVE = BUG (expected FAIL)",
            "input_file": str(input_file) if input_file else None,
        },
        "by_provider": analysis,
        "stratification": stratification,
    }

    if calibration:
        report["calibration"] = calibration
    if comparisons:
        report["prompt_comparisons"] = comparisons
    if bootstrap:
        report["bootstrap_ci"] = bootstrap

    with open(output, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\nReport saved to {output}")
    return output


# =============================================================================
# CLI
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Compute metrics from VLM benchmark results")
    parser.add_argument("pos_input_file", type=str, nargs="?", default=None,
                        help="Path to results JSONL file (positional)")
    parser.add_argument("-i", "--input_file", type=str, default=None,
                        help="Path to results JSONL file (flagged)")
    parser.add_argument("--dataset-dir", type=str, default=None,
                        help="Path to dataset dir (default: dataset). "
                             "Use 'dataset_screenqa' for ScreenQA data.")
    parser.add_argument("--no-bootstrap", action="store_true",
                        help="Skip bootstrap CI computation (faster)")
    parser.add_argument("--bootstrap-iter", type=int, default=1000,
                        help="Number of bootstrap iterations (default: 1000)")
    args = parser.parse_args()
    args.input_file = args.input_file or args.pos_input_file
    return args


def main():
    args = parse_args()

    global DATASET_DIR
    if args.dataset_dir:
        DATASET_DIR = Path(__file__).parent.parent / args.dataset_dir

    input_path = Path(args.input_file) if args.input_file else None
    results = load_results(input_path)

    if not results:
        print("No results found. Run evaluation first with: python scripts/run_eval.py")
        return

    print(f"Loaded {len(results)} evaluation results")

    test_metadata = load_test_metadata()

    # Core analysis
    analysis = analyze_by_provider(results)
    stratification = stratify_results(results, test_metadata)

    # Calibration
    calibration = compute_calibration(results)

    # Prompt comparisons
    comparisons = compute_prompt_comparisons(results)

    # Bootstrap CIs
    bootstrap = None
    if not args.no_bootstrap:
        print("Computing bootstrap confidence intervals...")
        bootstrap = compute_bootstrap_summary(results, n_iter=args.bootstrap_iter)

    # Output
    print_summary(analysis, stratification, calibration, comparisons, bootstrap)
    save_report(analysis, stratification, input_path, calibration, comparisons, bootstrap)


if __name__ == "__main__":
    main()
