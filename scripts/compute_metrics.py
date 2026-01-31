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

Optional (if confidence available):
- Brier Score, ECE for calibration

Stratification by tags:
- Cognitive level (L1/L2/L3)
- Operation tags (presence, text_match_*, count_*, etc.)
- Difficulty tags (near_miss, small_text, etc.)
"""

import json
import re
import argparse
from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass

RESULTS_DIR = Path(__file__).parent.parent / "results"
DATASET_DIR = Path(__file__).parent.parent / "dataset"


# =============================================================================
# V3 TAG DEFINITIONS
# =============================================================================

OPERATION_TAGS = {
    "presence", "absence", 
    "text_match_exact", "text_match_normalized", "text_match_semantic",
    "count_raw", "count_filtered",
    "state", "layout", "order", "consistency"
}

DIFFICULTY_TAGS = {
    "near_miss", "small_text", "low_contrast", 
    "cluttered", "occluded", "confusable"
}


def normalize_label(label: str | None) -> str | None:
    """Normalize result/expected labels to standard values.
    
    Returns:
        'PASS', 'FAIL', 'UNCLEAR', or None
    """
    if label is None:
        return None
    
    upper = str(label).upper().strip()
    
    # PASS aliases (safe: OK/TRUE/SUCCESS are unambiguous)
    if upper in ("PASS", "OK", "TRUE", "SUCCESS", "1"):
        return "PASS"
    
    # FAIL aliases (safe: FALSE/FAILURE are unambiguous)
    if upper in ("FAIL", "FALSE", "FAILURE", "0"):
        return "FAIL"
    
    # Abstention aliases
    if upper in ("UNCLEAR", "ABSTAIN", "UNKNOWN", "UNSURE", "SKIP", "N/A"):
        return "UNCLEAR"
    
    return None  # Unrecognized


# =============================================================================
# CONFUSION MATRIX WITH STANDARD ML METRICS
# =============================================================================
# Convention: POSITIVE = BUG (expected FAIL), NEGATIVE = OK (expected PASS)
# This matches standard ML where we care about detecting the "problem" class.

@dataclass
class ConfusionMatrix:
    """Confusion matrix with standard ML naming.
    
    Convention (POSITIVE = BUG):
    - TP: predicted FAIL, expected FAIL (correctly detected bug)
    - FP: predicted FAIL, expected PASS (false alarm, causes flakiness)
    - TN: predicted PASS, expected PASS (correctly passed)
    - FN: predicted PASS, expected FAIL (missed bug! dangerous)
    """
    tp: int = 0  # True Positive: predicted FAIL, expected FAIL (bug caught)
    fp: int = 0  # False Positive: predicted FAIL, expected PASS (false alarm)
    tn: int = 0  # True Negative: predicted PASS, expected PASS (correct pass)
    fn: int = 0  # False Negative: predicted PASS, expected FAIL (BUG ESCAPED!)
    abstained: int = 0  # Model said UNCLEAR/abstained
    skipped_invalid: int = 0  # Dataset issues (expected not PASS/FAIL)

    @property
    def total(self) -> int:
        return self.tp + self.fp + self.tn + self.fn

    @property
    def total_positive(self) -> int:
        """Total actual bugs (expected FAIL)."""
        return self.tp + self.fn

    @property
    def total_negative(self) -> int:
        """Total actual OK (expected PASS)."""
        return self.tn + self.fp

    # ========== STANDARD ML METRICS ==========

    @property
    def accuracy(self) -> float:
        """Overall accuracy = (TP + TN) / Total."""
        return (self.tp + self.tn) / self.total if self.total > 0 else 0.0

    @property
    def tpr(self) -> float:
        """True Positive Rate (Recall/Sensitivity).
        P(predict FAIL | actual bug). Higher = better bug detection."""
        return self.tp / self.total_positive if self.total_positive > 0 else 0.0

    @property
    def fnr(self) -> float:
        """False Negative Rate (Miss Rate).
        P(predict PASS | actual bug). CRITICAL: bugs that escaped! Lower = safer."""
        return self.fn / self.total_positive if self.total_positive > 0 else 0.0

    @property
    def tnr(self) -> float:
        """True Negative Rate (Specificity).
        P(predict PASS | actual OK). Higher = fewer false alarms."""
        return self.tn / self.total_negative if self.total_negative > 0 else 0.0

    @property
    def fpr(self) -> float:
        """False Positive Rate (False Alarm Rate).
        P(predict FAIL | actual OK). Causes test flakiness. Lower = more stable."""
        return self.fp / self.total_negative if self.total_negative > 0 else 0.0

    @property
    def ppv(self) -> float:
        """Positive Predictive Value (Precision).
        P(actual bug | predict FAIL). Higher = confident FAILs are real bugs."""
        predicted_positive = self.tp + self.fp
        return self.tp / predicted_positive if predicted_positive > 0 else 0.0

    @property
    def npv(self) -> float:
        """Negative Predictive Value.
        P(actual OK | predict PASS). Higher = confident PASSes are truly OK."""
        predicted_negative = self.tn + self.fn
        return self.tn / predicted_negative if predicted_negative > 0 else 0.0

    @property
    def fdr(self) -> float:
        """False Discovery Rate = 1 - PPV.
        P(actual OK | predict FAIL). False alarms among FAILs."""
        return 1.0 - self.ppv

    @property
    def for_(self) -> float:
        """False Omission Rate = 1 - NPV.
        P(actual bug | predict PASS). Bugs hidden in PASSes! Critical."""
        return 1.0 - self.npv

    @property
    def f1(self) -> float:
        """F1 score = harmonic mean of precision and recall."""
        p, r = self.ppv, self.tpr
        return 2 * p * r / (p + r) if (p + r) > 0 else 0.0

    @property
    def coverage(self) -> float:
        """% of tests decided (not abstained/UNCLEAR)."""
        total_with_abstained = self.total + self.abstained
        return self.total / total_with_abstained if total_with_abstained > 0 else 0.0

    @property
    def abstain_rate(self) -> float:
        """% of tests where model abstained (said UNCLEAR)."""
        total_with_abstained = self.total + self.abstained
        return self.abstained / total_with_abstained if total_with_abstained > 0 else 0.0

    @property
    def fail_rate_decided(self) -> float:
        """% of decided predictions that are FAIL."""
        predicted_positive = self.tp + self.fp
        return predicted_positive / self.total if self.total > 0 else 0.0

    @property
    def balanced_accuracy(self) -> float:
        """Balanced accuracy = (TPR + TNR) / 2. Robust to class imbalance."""
        return (self.tpr + self.tnr) / 2

    @property
    def mcc(self) -> float:
        """Matthews Correlation Coefficient. Best single metric for imbalanced binary.
        Range: -1 (worst) to +1 (perfect). 0 = random."""
        import math
        numerator = (self.tp * self.tn) - (self.fp * self.fn)
        denominator = math.sqrt(
            (self.tp + self.fp) * (self.tp + self.fn) *
            (self.tn + self.fp) * (self.tn + self.fn)
        )
        return numerator / denominator if denominator > 0 else 0.0

    # Aliases for readability
    precision = ppv
    recall = tpr
    sensitivity = tpr
    specificity = tnr
    miss_rate = fnr
    false_alarm_rate = fpr

    def to_dict(self) -> dict:
        return {
            # Raw counts
            "tp": self.tp,
            "fp": self.fp,
            "tn": self.tn,
            "fn": self.fn,
            "abstained": self.abstained,
            "skipped_invalid": self.skipped_invalid,
            "decided": self.total,
            "total_with_abstained": self.total + self.abstained,
            # Standard ML metrics
            "accuracy": round(self.accuracy, 4),
            "tpr": round(self.tpr, 4),  # recall / sensitivity
            "fnr": round(self.fnr, 4),  # miss rate (bugs escaped)
            "tnr": round(self.tnr, 4),  # specificity
            "fpr": round(self.fpr, 4),  # false alarm rate
            "ppv": round(self.ppv, 4),  # precision
            "npv": round(self.npv, 4),
            "for": round(self.for_, 4),  # P(bug|PASS) - critical!
            "f1": round(self.f1, 4),
            "balanced_accuracy": round(self.balanced_accuracy, 4),
            "mcc": round(self.mcc, 4),
            "coverage": round(self.coverage, 4),
            "abstain_rate": round(self.abstain_rate, 4),
            "fail_rate_decided": round(self.fail_rate_decided, 4),
        }


# =============================================================================
# CALIBRATION METRICS (optional, requires confidence scores)
# =============================================================================

def compute_brier_score(predictions: list[tuple[float, int]]) -> float:
    """Brier score: measures how well confidence matches reality.
    Lower is better. 0 = perfect, 0.25 = random."""
    if not predictions:
        return None
    total = sum((p - a) ** 2 for p, a in predictions)
    return round(total / len(predictions), 4)


def compute_ece(predictions: list[tuple[float, int]], n_bins: int = 10) -> float:
    """Expected Calibration Error.
    Lower is better. 0 = perfectly calibrated."""
    if not predictions:
        return None
    
    bins = defaultdict(list)
    for prob, actual in predictions:
        bin_idx = min(int(prob * n_bins), n_bins - 1)
        bins[bin_idx].append((prob, actual))
    
    ece = 0.0
    total = len(predictions)
    
    for bin_preds in bins.values():
        if bin_preds:
            avg_confidence = sum(p for p, _ in bin_preds) / len(bin_preds)
            avg_accuracy = sum(a for _, a in bin_preds) / len(bin_preds)
            ece += len(bin_preds) / total * abs(avg_confidence - avg_accuracy)
    
    return round(ece, 4)


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
        
        # If symlink doesn't exist, find the latest raw_*.jsonl
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
    """Load test metadata from v3 tests.json files."""
    tests = {}
    screenshots_dir = DATASET_DIR / "screenshots"
    
    if not screenshots_dir.exists():
        return {}
    
    for shot_dir in sorted(screenshots_dir.iterdir()):
        if shot_dir.is_dir() and shot_dir.name.startswith("shot_"):
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


# =============================================================================
# CORE ANALYSIS
# =============================================================================

def compute_cognitive_level(tags: list[str]) -> str:
    """Derive cognitive level from v3 flat tags."""
    tag_set = set(tags)
    
    # L3: semantic understanding or business rules
    if "consistency" in tag_set or "text_match_semantic" in tag_set:
        return "L3"
    
    # L2: comparison, spatial reasoning, or filtering
    ops = tag_set & OPERATION_TAGS
    if len(ops) >= 2 or "layout" in tag_set or "order" in tag_set or "count_filtered" in tag_set:
        return "L2"
    
    # L1: single operation, no complexity
    return "L1"


def build_confusion_matrix(results: list[dict]) -> ConfusionMatrix:
    """Build confusion matrix from results.
    
    Convention: POSITIVE = BUG (expected FAIL)
    - TP: predicted FAIL, expected FAIL (bug caught)
    - FP: predicted FAIL, expected PASS (false alarm)
    - TN: predicted PASS, expected PASS (correct pass)
    - FN: predicted PASS, expected FAIL (bug escaped!)
    
    Handles various label formats via normalize_label().
    """
    cm = ConfusionMatrix()
    
    for r in results:
        result = normalize_label(r.get("result"))
        expected = normalize_label(r.get("expected"))
        abstained = r.get("abstained", False)
        
        # Handle abstention: UNCLEAR, None, or explicit flag
        if abstained or result == "UNCLEAR" or result is None:
            cm.abstained += 1
            continue
        
        # Skip if expected is not recognized (dataset issue, not model abstention)
        if expected not in ("PASS", "FAIL"):
            cm.skipped_invalid += 1
            continue
        
        # POSITIVE = BUG = expected FAIL
        if expected == "FAIL":  # Actual positive (bug exists)
            if result == "FAIL":
                cm.tp += 1  # Correctly detected bug
            else:
                cm.fn += 1  # Bug escaped! (predicted PASS)
        elif expected == "PASS":  # Actual negative (no bug)
            if result == "FAIL":
                cm.fp += 1  # False alarm (flaky)
            else:
                cm.tn += 1  # Correctly passed
    
    return cm


def analyze_by_provider(results: list[dict]) -> dict:
    """Group results by provider and compute metrics."""
    by_provider = defaultdict(list)
    for r in results:
        provider = r.get("provider", "unknown")
        by_provider[provider].append(r)
    
    analysis = {}
    for provider, provider_results in by_provider.items():
        cm = build_confusion_matrix(provider_results)
        analysis[provider] = cm.to_dict()
    
    return analysis


def stratify_results(results: list[dict], test_metadata: dict[str, dict]) -> dict:
    """Stratify results by v3 flat tags."""
    
    by_cognitive_level = defaultdict(list)
    by_operation = defaultdict(list)
    by_difficulty = defaultdict(list)
    by_polarity = defaultdict(list)
    by_profile = defaultdict(list)
    by_screenshot = defaultdict(list)  # NEW
    
    for r in results:
        test_id = r.get("test_id")
        meta = test_metadata.get(test_id, {})
        
        # Ensure expected is set
        if "expected" not in r and "expected" in meta:
            r["expected"] = meta["expected"]
        
        tags = meta.get("tags", [])
        expected = r.get("expected", meta.get("expected", "unknown"))
        profile = r.get("profile", "unknown")
        image_id = r.get("image_id", meta.get("image_id", "unknown"))  # NEW
        
        # System prompt profile
        by_profile[profile].append(r)
        
        # Screenshot
        by_screenshot[image_id].append(r)  # NEW
        
        # Cognitive level
        level = compute_cognitive_level(tags) if tags else "unknown"
        by_cognitive_level[level].append(r)
        
        # Polarity (normalized)
        expected_norm = normalize_label(expected) or "unknown"
        by_polarity[expected_norm].append(r)
        
        # Operations
        for tag in tags:
            if tag in OPERATION_TAGS:
                by_operation[tag].append(r)
        
        # Difficulty
        has_difficulty = False
        for tag in tags:
            if tag in DIFFICULTY_TAGS:
                by_difficulty[tag].append(r)
                has_difficulty = True
        if not has_difficulty:
            by_difficulty["baseline"].append(r)
    
    def compute_metrics(group_dict: dict) -> dict:
        return {
            name: build_confusion_matrix(items).to_dict()
            for name, items in group_dict.items()
            if items
        }
    
    return {
        "by_profile": compute_metrics(by_profile),
        "by_screenshot": compute_metrics(by_screenshot),  # NEW
        "by_cognitive_level": compute_metrics(by_cognitive_level),
        "by_polarity": compute_metrics(by_polarity),
        "by_operation": compute_metrics(by_operation),
        "by_difficulty": compute_metrics(by_difficulty),
    }


# =============================================================================
# OUTPUT
# =============================================================================

def print_summary(analysis: dict, stratification: dict):
    """Print formatted summary to console."""
    print("\n" + "=" * 70)
    print("UI ASSERTION VLM BENCHMARK - STANDARD ML METRICS")
    print("Convention: POSITIVE = BUG (expected FAIL)")
    print("=" * 70)
    
    for provider, metrics in analysis.items():
        print(f"\n### Provider: {provider}")
        print("-" * 50)
        
        print(f"\nConfusion Matrix (POSITIVE = BUG):")
        print(f"  TP={metrics['tp']:3d}  FP={metrics['fp']:3d}  (TP=bug caught, FP=false alarm)")
        print(f"  FN={metrics['fn']:3d}  TN={metrics['tn']:3d}  (FN=BUG ESCAPED!, TN=correct pass)")
        print(f"  Abstained: {metrics['abstained']}  Skipped: {metrics['skipped_invalid']}")
        
        print(f"\n🔴 CRITICAL METRICS:")
        print(f"  FNR (Miss Rate):    {metrics['fnr']:.1%} (bugs escaped! want < 5%)")
        print(f"  FOR:                {metrics['for']:.1%} (P(bug|PASS) - hidden bugs)")
        
        print(f"\n🟡 Other Rates:")
        print(f"  FPR (False Alarm):  {metrics['fpr']:.1%} (causes flakiness)")
        print(f"  TPR (Recall):       {metrics['tpr']:.1%} (bug detection rate, want high)")
        print(f"  FAIL Rate:          {metrics['fail_rate_decided']:.1%} (of decided)")
        
        print(f"\n📊 Standard Metrics:")
        print(f"  Accuracy:           {metrics['accuracy']:.1%}")
        print(f"  PPV (Precision):    {metrics['ppv']:.1%}")
        print(f"  NPV:                {metrics['npv']:.1%}")
        print(f"  F1:                 {metrics['f1']:.1%}")
        print(f"  Coverage:           {metrics['coverage']:.1%}")
        print(f"  Abstain Rate:       {metrics['abstain_rate']:.1%}")
    
    # Stratification
    print("\n" + "=" * 70)
    print("STRATIFICATION ANALYSIS")
    print("=" * 70)
    
    for dim_name, dim_data in stratification.items():
        print(f"\n### {dim_name.replace('by_', '').replace('_', ' ').title()}")
        print(f"{'Group':<25} {'FNR':<8} {'FPR':<8} {'Acc':<8} {'N':<6}")
        print("-" * 55)
        for group, metrics in sorted(dim_data.items()):
            if metrics["decided"] > 0:
                print(f"{group:<25} {metrics['fnr']:.1%}    "
                      f"{metrics['fpr']:.1%}    "
                      f"{metrics['accuracy']:.1%}    {metrics['decided']}")


def extract_timestamp(filepath: Path) -> str | None:
    """Extract timestamp from filename like raw_20260131_014523.jsonl."""
    match = re.search(r'(\d{8}_\d{6})', filepath.name)
    return match.group(1) if match else None


def save_report(analysis: dict, stratification: dict, input_file: Path = None):
    """Save JSON report. Uses timestamp from input file if available."""
    timestamp = extract_timestamp(input_file) if input_file else None
    
    if timestamp:
        output = RESULTS_DIR / f"metrics_{timestamp}.json"
    else:
        output = RESULTS_DIR / "metrics.json"
    
    report = {
        "by_provider": analysis,
        "stratification": stratification,
    }
    
    with open(output, "w") as f:
        json.dump(report, f, indent=2)
    
    print(f"\nReport saved to {output}")
    return output


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Compute metrics from VLM benchmark results"
    )
    # Support both positional and flagged input_file
    parser.add_argument(
        "pos_input_file",
        type=str,
        nargs="?",
        default=None,
        help="Path to results JSONL file (positional)"
    )
    parser.add_argument(
        "-i", "--input_file",
        type=str,
        default=None,
        help="Path to results JSONL file (flagged)"
    )
    
    args = parser.parse_args()
    # Resolve which input file to use (flag takes precedence if both provided)
    args.input_file = args.input_file or args.pos_input_file
    return args


def main():
    args = parse_args()
    
    input_path = Path(args.input_file) if args.input_file else None
    results = load_results(input_path)
    
    if not results:
        print("No results found. Run evaluation first with: python scripts/run_eval.py")
        return
    
    print(f"Loaded {len(results)} evaluation results")
    
    test_metadata = load_test_metadata()
    
    analysis = analyze_by_provider(results)
    stratification = stratify_results(results, test_metadata)
    
    print_summary(analysis, stratification)
    save_report(analysis, stratification, input_path)


if __name__ == "__main__":
    main()
