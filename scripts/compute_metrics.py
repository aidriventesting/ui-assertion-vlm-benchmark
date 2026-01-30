#!/usr/bin/env python3
"""
Compute QA-focused metrics for UI Assertion VLM Benchmark.

Core QA Metrics:
- Confusion matrix (TP/FP/TN/FN)
- Bug Escape Rate (FPR) - CRITICAL: bugs that slip through
- Flaky Rate (FNR) - false alarms  
- Yes-Bias - does model favor PASS?
- False PASS Rate - P(bug | model says PASS)

Optional (if confidence available):
- Brier Score, ECE for calibration

Stratification by v3 tags:
- Cognitive level (L1/L2/L3)
- Operation tags (presence, text_match_*, count_*, etc.)
- Difficulty tags (near_miss, small_text, etc.)
"""

import json
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


# =============================================================================
# CONFUSION MATRIX WITH QA METRICS
# =============================================================================

@dataclass
class ConfusionMatrix:
    """Confusion matrix with QA-focused metrics."""
    tp: int = 0  # True PASS (predicted PASS, expected PASS)
    fp: int = 0  # False PASS (predicted PASS, expected FAIL) - BUGS ESCAPED!
    tn: int = 0  # True FAIL (predicted FAIL, expected FAIL)
    fn: int = 0  # False FAIL (predicted FAIL, expected PASS) - flaky tests
    abstained: int = 0

    @property
    def total(self) -> int:
        return self.tp + self.fp + self.tn + self.fn

    @property
    def total_answered(self) -> int:
        return self.total

    @property
    def total_expected_pass(self) -> int:
        return self.tp + self.fn

    @property
    def total_expected_fail(self) -> int:
        return self.tn + self.fp

    # ========== CORE QA METRICS ==========

    @property
    def accuracy(self) -> float:
        """Overall accuracy. Careful: can be misleading if unbalanced."""
        return (self.tp + self.tn) / self.total if self.total > 0 else 0.0

    @property
    def bug_escape_rate(self) -> float:
        """CRITICAL: % of real bugs (expected FAIL) that model said PASS.
        This is the same as FPR. Lower is better. Target: < 5%."""
        actual_bugs = self.tn + self.fp
        return self.fp / actual_bugs if actual_bugs > 0 else 0.0

    @property
    def flaky_rate(self) -> float:
        """% of good screens (expected PASS) that model said FAIL.
        This is the same as FNR. Causes test flakiness. Lower is better."""
        actual_good = self.tp + self.fn
        return self.fn / actual_good if actual_good > 0 else 0.0

    @property
    def yes_bias(self) -> float:
        """Ratio of PASS predictions. 0.5 = balanced, >0.5 favors PASS."""
        total_predictions = self.tp + self.fp + self.tn + self.fn
        pass_predictions = self.tp + self.fp
        return pass_predictions / total_predictions if total_predictions > 0 else 0.5

    @property
    def false_pass_rate(self) -> float:
        """When model says PASS, what % are actually bugs?
        P(bug | predicted PASS). Lower is better."""
        pass_predictions = self.tp + self.fp
        return self.fp / pass_predictions if pass_predictions > 0 else 0.0

    @property
    def fail_detection_rate(self) -> float:
        """When there's a bug, how often do we catch it?
        Same as TNR / Recall on FAIL. Higher is better."""
        actual_bugs = self.tn + self.fp
        return self.tn / actual_bugs if actual_bugs > 0 else 0.0

    @property
    def coverage(self) -> float:
        """% of tests that got an answer (not abstained)."""
        total = self.total + self.abstained
        return self.total / total if total > 0 else 0.0

    # ========== TRADITIONAL ML METRICS (for reference) ==========

    @property
    def precision(self) -> float:
        """Precision for PASS class."""
        return self.tp / (self.tp + self.fp) if (self.tp + self.fp) > 0 else 0.0

    @property
    def recall(self) -> float:
        """Recall for PASS class."""
        return self.tp / (self.tp + self.fn) if (self.tp + self.fn) > 0 else 0.0

    @property
    def f1(self) -> float:
        """F1 score for PASS class."""
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) > 0 else 0.0

    def to_dict(self) -> dict:
        return {
            # Raw counts
            "tp": self.tp,
            "fp": self.fp,
            "tn": self.tn,
            "fn": self.fn,
            "abstained": self.abstained,
            "total": self.total,
            # QA metrics (primary)
            "accuracy": round(self.accuracy, 4),
            "bug_escape_rate": round(self.bug_escape_rate, 4),
            "flaky_rate": round(self.flaky_rate, 4),
            "yes_bias": round(self.yes_bias, 4),
            "false_pass_rate": round(self.false_pass_rate, 4),
            "fail_detection_rate": round(self.fail_detection_rate, 4),
            "coverage": round(self.coverage, 4),
            # Traditional ML (secondary)
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1": round(self.f1, 4),
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

def load_results() -> list[dict]:
    """Load evaluation results from raw.jsonl."""
    results = []
    results_file = RESULTS_DIR / "raw.jsonl"
    
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
    """Build confusion matrix from results."""
    cm = ConfusionMatrix()
    
    for r in results:
        result = r.get("result")
        expected = r.get("expected")
        abstained = r.get("abstained", False)
        
        if abstained or result == "ABSTAIN" or result is None:
            cm.abstained += 1
            continue
        
        if expected == "PASS":
            if result == "PASS":
                cm.tp += 1
            else:
                cm.fn += 1
        elif expected == "FAIL":
            if result == "PASS":
                cm.fp += 1
            else:
                cm.tn += 1
    
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
    
    for r in results:
        test_id = r.get("test_id")
        meta = test_metadata.get(test_id, {})
        
        # Ensure expected is set
        if "expected" not in r and "expected" in meta:
            r["expected"] = meta["expected"]
        
        tags = meta.get("tags", [])
        expected = r.get("expected", meta.get("expected", "unknown"))
        
        # Cognitive level
        level = compute_cognitive_level(tags) if tags else "unknown"
        by_cognitive_level[level].append(r)
        
        # Polarity
        by_polarity[expected].append(r)
        
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
    print("UI ASSERTION VLM BENCHMARK - QA METRICS")
    print("=" * 70)
    
    for provider, metrics in analysis.items():
        print(f"\n### Provider: {provider}")
        print("-" * 50)
        
        print(f"\nConfusion Matrix:")
        print(f"  TP={metrics['tp']:3d}  FP={metrics['fp']:3d}  (FP = bugs escaped!)")
        print(f"  FN={metrics['fn']:3d}  TN={metrics['tn']:3d}  (Abstained: {metrics['abstained']})")
        
        print(f"\n🔴 CRITICAL QA METRICS:")
        print(f"  Bug Escape Rate:    {metrics['bug_escape_rate']:.1%} (want < 5%)")
        print(f"  False PASS Rate:    {metrics['false_pass_rate']:.1%} (P(bug|PASS))")
        
        print(f"\n🟡 Other QA Metrics:")
        print(f"  Flaky Rate:         {metrics['flaky_rate']:.1%}")
        print(f"  Yes-Bias:           {metrics['yes_bias']:.1%} (50% = balanced)")
        print(f"  Fail Detection:     {metrics['fail_detection_rate']:.1%} (want high)")
        
        print(f"\n📊 Standard Metrics:")
        print(f"  Accuracy:           {metrics['accuracy']:.1%}")
        print(f"  Precision:          {metrics['precision']:.1%}")
        print(f"  F1:                 {metrics['f1']:.1%}")
        print(f"  Coverage:           {metrics['coverage']:.1%}")
    
    # Stratification
    print("\n" + "=" * 70)
    print("STRATIFICATION ANALYSIS")
    print("=" * 70)
    
    for dim_name, dim_data in stratification.items():
        print(f"\n### {dim_name.replace('by_', '').replace('_', ' ').title()}")
        print(f"{'Group':<25} {'BugEsc':<8} {'Flaky':<8} {'Acc':<8} {'N':<6}")
        print("-" * 55)
        for group, metrics in sorted(dim_data.items()):
            if metrics["total"] > 0:
                print(f"{group:<25} {metrics['bug_escape_rate']:.1%}    "
                      f"{metrics['flaky_rate']:.1%}    "
                      f"{metrics['accuracy']:.1%}    {metrics['total']}")


def save_report(analysis: dict, stratification: dict):
    """Save JSON report."""
    output = RESULTS_DIR / "metrics.json"
    
    report = {
        "by_provider": analysis,
        "stratification": stratification,
    }
    
    with open(output, "w") as f:
        json.dump(report, f, indent=2)
    
    print(f"\nReport saved to {output}")


def main():
    results = load_results()
    
    if not results:
        print("No results found. Run evaluation first with: python scripts/run_eval.py")
        return
    
    print(f"Loaded {len(results)} evaluation results")
    
    test_metadata = load_test_metadata()
    
    analysis = analyze_by_provider(results)
    stratification = stratify_results(results, test_metadata)
    
    print_summary(analysis, stratification)
    save_report(analysis, stratification)


if __name__ == "__main__":
    main()
