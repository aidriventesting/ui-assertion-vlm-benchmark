#!/usr/bin/env python3
"""
Compute comprehensive metrics for UI Assertion VLM Benchmark.

Metrics include:
- Confusion matrix (TP/FP/TN/FN)
- TPR/FNR/FPR/TNR
- Accuracy, precision, recall, F1
- Abstention rate and conditional accuracy
- Calibration: Brier score, ECE
- Stratification by tags, cognitive level, assertion type
- Cost analysis (tokens, latency, $/eval)
"""

import json
import math
from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional

RESULTS_DIR = Path(__file__).parent.parent / "results"
DATASET_DIR = Path(__file__).parent.parent / "dataset"
THRESHOLDS = [50, 70, 80, 90]


@dataclass
class ConfusionMatrix:
    """Confusion matrix for binary classification (PASS/FAIL)."""
    tp: int = 0  # True PASS (predicted PASS, expected PASS)
    fp: int = 0  # False PASS (predicted PASS, expected FAIL) - most dangerous!
    tn: int = 0  # True FAIL (predicted FAIL, expected FAIL)
    fn: int = 0  # False FAIL (predicted FAIL, expected PASS)
    abstained: int = 0  # Count of abstentions

    @property
    def total(self) -> int:
        return self.tp + self.fp + self.tn + self.fn

    @property
    def total_with_abstain(self) -> int:
        return self.total + self.abstained

    @property
    def tpr(self) -> float:
        """True Positive Rate (Recall for PASS class)."""
        return self.tp / (self.tp + self.fn) if (self.tp + self.fn) > 0 else 0.0

    @property
    def tnr(self) -> float:
        """True Negative Rate (Recall for FAIL class)."""
        return self.tn / (self.tn + self.fp) if (self.tn + self.fp) > 0 else 0.0

    @property
    def fpr(self) -> float:
        """False Positive Rate - predicted PASS when expected FAIL."""
        return self.fp / (self.fp + self.tn) if (self.fp + self.tn) > 0 else 0.0

    @property
    def fnr(self) -> float:
        """False Negative Rate - predicted FAIL when expected PASS."""
        return self.fn / (self.fn + self.tp) if (self.fn + self.tp) > 0 else 0.0

    @property
    def accuracy(self) -> float:
        """Overall accuracy."""
        return (self.tp + self.tn) / self.total if self.total > 0 else 0.0

    @property
    def precision(self) -> float:
        """Precision for PASS class."""
        return self.tp / (self.tp + self.fp) if (self.tp + self.fp) > 0 else 0.0

    @property
    def recall(self) -> float:
        """Recall for PASS class (same as TPR)."""
        return self.tpr

    @property
    def f1(self) -> float:
        """F1 score for PASS class."""
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) > 0 else 0.0

    @property
    def coverage(self) -> float:
        """Coverage = proportion of non-abstained responses."""
        return self.total / self.total_with_abstain if self.total_with_abstain > 0 else 0.0

    @property
    def accuracy_given_answered(self) -> float:
        """Accuracy conditional on having answered (not abstained)."""
        return self.accuracy  # Same as accuracy since we only count answered

    def to_dict(self) -> dict:
        return {
            "tp": self.tp,
            "fp": self.fp,
            "tn": self.tn,
            "fn": self.fn,
            "abstained": self.abstained,
            "total": self.total,
            "tpr": self.tpr,
            "tnr": self.tnr,
            "fpr": self.fpr,
            "fnr": self.fnr,
            "accuracy": self.accuracy,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "coverage": self.coverage,
        }


@dataclass
class CostMetrics:
    """Track cost and efficiency metrics."""
    input_tokens: int = 0
    output_tokens: int = 0
    total_latency_ms: int = 0
    api_calls: int = 0

    def add(self, cost_data: dict):
        self.input_tokens += cost_data.get("input_tokens", 0)
        self.output_tokens += cost_data.get("output_tokens", 0)
        self.total_latency_ms += cost_data.get("latency_ms", 0)
        self.api_calls += cost_data.get("api_calls", 1)

    def to_dict(self, pricing: dict = None) -> dict:
        pricing = pricing or {"input_per_million": 0.15, "output_per_million": 0.60}
        total_cost = (
            self.input_tokens * pricing["input_per_million"] / 1_000_000 +
            self.output_tokens * pricing["output_per_million"] / 1_000_000
        )
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.input_tokens + self.output_tokens,
            "total_latency_ms": self.total_latency_ms,
            "avg_latency_ms": self.total_latency_ms / self.api_calls if self.api_calls > 0 else 0,
            "api_calls": self.api_calls,
            "total_cost_usd": total_cost,
            "cost_per_eval_usd": total_cost / self.api_calls if self.api_calls > 0 else 0,
        }


def compute_brier_score(predictions: list[tuple[float, int]]) -> float:
    """
    Compute Brier score for calibration.
    
    Args:
        predictions: List of (predicted_prob_fail, actual_is_fail) tuples
    
    Returns:
        Brier score (lower is better, 0 = perfect calibration)
    """
    if not predictions:
        return 0.0
    
    total = sum((p - a) ** 2 for p, a in predictions)
    return total / len(predictions)


def compute_ece(predictions: list[tuple[float, int]], n_bins: int = 10) -> float:
    """
    Compute Expected Calibration Error.
    
    Args:
        predictions: List of (predicted_prob, actual_outcome) tuples
        n_bins: Number of bins for calibration
    
    Returns:
        ECE (lower is better, 0 = perfect calibration)
    """
    if not predictions:
        return 0.0
    
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
    
    return ece


def load_results() -> list[dict]:
    """Load all results from raw.jsonl."""
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
    """Load test case metadata for stratification."""
    tests = {}
    tests_file = DATASET_DIR / "tests.jsonl"
    
    if not tests_file.exists():
        return {}
    
    with open(tests_file, "r") as f:
        for line in f:
            if line.strip():
                test = json.loads(line)
                test_id = test.get("assertion_id") or test.get("id")
                tests[test_id] = test
    return tests


def build_confusion_matrix(results: list[dict], threshold: int = None) -> ConfusionMatrix:
    """Build confusion matrix from results, optionally filtering by confidence threshold."""
    cm = ConfusionMatrix()
    
    for r in results:
        result = r.get("result")
        expected = r.get("expected") or r.get("gt_label")
        confidence = r.get("confidence")
        abstained = r.get("abstained", False)
        
        # Handle abstention
        if abstained or result == "ABSTAIN":
            cm.abstained += 1
            continue
        
        # Apply confidence threshold filter
        if threshold is not None and confidence is not None:
            if confidence < threshold:
                cm.abstained += 1
                continue
        
        # Classify result
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
        # Skip AMBIG cases
    
    return cm


def compute_calibration(results: list[dict]) -> dict:
    """Compute calibration metrics (Brier score, ECE)."""
    predictions = []
    
    for r in results:
        confidence = r.get("confidence")
        expected = r.get("expected") or r.get("gt_label")
        result = r.get("result")
        
        if confidence is None or expected == "AMBIG":
            continue
        
        # Convert confidence to probability of FAIL
        # If model says PASS with 90% confidence, P(FAIL) = 10%
        # If model says FAIL with 90% confidence, P(FAIL) = 90%
        if result == "PASS":
            prob_fail = (100 - confidence) / 100
        else:
            prob_fail = confidence / 100
        
        actual_is_fail = 1 if expected == "FAIL" else 0
        predictions.append((prob_fail, actual_is_fail))
    
    if not predictions:
        return {"brier_score": None, "ece": None, "n_predictions": 0}
    
    return {
        "brier_score": compute_brier_score(predictions),
        "ece": compute_ece(predictions),
        "n_predictions": len(predictions),
    }


def analyze_by_profile(results: list[dict]) -> dict:
    """Group results by profile and compute all metrics."""
    by_profile = defaultdict(list)
    for r in results:
        by_profile[r.get("profile", "unknown")].append(r)
    
    analysis = {}
    for profile, profile_results in by_profile.items():
        cm = build_confusion_matrix(profile_results)
        cost = CostMetrics()
        for r in profile_results:
            if "cost" in r:
                cost.add(r["cost"])
        
        analysis[profile] = {
            "confusion_matrix": cm.to_dict(),
            "calibration": compute_calibration(profile_results),
            "cost": cost.to_dict(),
            "by_threshold": {},
        }
        
        for t in THRESHOLDS:
            cm_t = build_confusion_matrix(profile_results, threshold=t)
            analysis[profile]["by_threshold"][t] = cm_t.to_dict()
    
    return analysis


def stratify_results(results: list[dict], test_metadata: dict[str, dict]) -> dict:
    """Stratify results by various dimensions from test metadata."""
    
    # Group results by various dimensions
    by_cognitive_level = defaultdict(list)
    by_assertion_type = defaultdict(list)
    by_difficulty = defaultdict(list)
    by_severity = defaultdict(list)
    
    for r in results:
        test_id = r.get("test_id") or r.get("assertion_id")
        meta = test_metadata.get(test_id, {})
        
        # Add expected from metadata if not in result
        if "expected" not in r and "gt_label" in meta:
            r["expected"] = meta["gt_label"]
        
        # Stratify by cognitive level
        level = meta.get("cognitive_level", "unknown")
        by_cognitive_level[level].append(r)
        
        # Stratify by assertion type
        atype = meta.get("assertion_type", "unknown")
        by_assertion_type[atype].append(r)
        
        # Stratify by difficulty drivers
        tags = meta.get("tags", {})
        drivers = tags.get("difficulty_drivers", [])
        if not drivers:
            by_difficulty["none"].append(r)
        for driver in drivers:
            by_difficulty[driver].append(r)
        
        # Stratify by severity
        severity = tags.get("severity", "unknown")
        by_severity[severity].append(r)
    
    # Also stratify by mode fields
    by_text_match_mode = defaultdict(list)
    by_count_scope = defaultdict(list)
    
    for r in results:
        test_id = r.get("test_id") or r.get("assertion_id")
        meta = test_metadata.get(test_id, {})
        
        if meta.get("assertion_type") == "text_match":
            mode = meta.get("text_match_mode", "unknown")
            by_text_match_mode[mode].append(r)
        
        if meta.get("assertion_type") == "count":
            scope = meta.get("count_scope", "unknown")
            by_count_scope[scope].append(r)
    
    def compute_group_metrics(group_dict: dict) -> dict:
        return {
            name: build_confusion_matrix(items).to_dict()
            for name, items in group_dict.items()
        }
    
    strat = {
        "by_cognitive_level": compute_group_metrics(by_cognitive_level),
        "by_assertion_type": compute_group_metrics(by_assertion_type),
        "by_difficulty_driver": compute_group_metrics(by_difficulty),
        "by_severity": compute_group_metrics(by_severity),
    }
    
    # Add mode stratifications only if they have data
    if by_text_match_mode:
        strat["by_text_match_mode"] = compute_group_metrics(by_text_match_mode)
    if by_count_scope:
        strat["by_count_scope"] = compute_group_metrics(by_count_scope)
    
    return strat


def print_summary(analysis: dict, stratification: dict):
    """Print formatted summary to console."""
    print("\n" + "=" * 80)
    print("UI ASSERTION VLM BENCHMARK - COMPREHENSIVE METRICS")
    print("=" * 80)
    
    for profile, data in analysis.items():
        print(f"\n### Profile: {profile}")
        print("-" * 60)
        
        cm = data["confusion_matrix"]
        print(f"\nConfusion Matrix:")
        print(f"  TP={cm['tp']:3d}  FP={cm['fp']:3d}  (FP = False PASS = missed bugs!)")
        print(f"  FN={cm['fn']:3d}  TN={cm['tn']:3d}  (Abstained: {cm['abstained']})")
        
        print(f"\nKey Metrics:")
        print(f"  Accuracy:  {cm['accuracy']:.2%}")
        print(f"  FPR:       {cm['fpr']:.2%} (want low - avoid missing bugs)")
        print(f"  FNR:       {cm['fnr']:.2%} (want low - avoid false alarms)")
        print(f"  Precision: {cm['precision']:.2%}")
        print(f"  Recall:    {cm['recall']:.2%}")
        print(f"  F1:        {cm['f1']:.2%}")
        print(f"  Coverage:  {cm['coverage']:.2%}")
        
        cal = data["calibration"]
        if cal["brier_score"] is not None:
            print(f"\nCalibration:")
            print(f"  Brier Score: {cal['brier_score']:.4f} (lower = better)")
            print(f"  ECE:         {cal['ece']:.4f} (lower = better)")
        
        cost = data["cost"]
        if cost["api_calls"] > 0:
            print(f"\nCost:")
            print(f"  Total tokens:  {cost['total_tokens']:,}")
            print(f"  Avg latency:   {cost['avg_latency_ms']:.0f}ms")
            print(f"  Cost/eval:     ${cost['cost_per_eval_usd']:.5f}")
        
        print(f"\nBy Confidence Threshold:")
        print(f"{'Thresh':<8} {'FPR':<8} {'FNR':<8} {'Acc':<8} {'Cov':<8}")
        print("-" * 40)
        for t, metrics in data["by_threshold"].items():
            if metrics["total"] > 0:
                print(f"{t:<8} {metrics['fpr']:.2%}   {metrics['fnr']:.2%}   "
                      f"{metrics['accuracy']:.2%}   {metrics['coverage']:.2%}")
    
    # Stratification summary
    print("\n" + "=" * 80)
    print("STRATIFICATION ANALYSIS")
    print("=" * 80)
    
    for dim_name, dim_data in stratification.items():
        print(f"\n### {dim_name.replace('_', ' ').title()}")
        print(f"{'Group':<20} {'FPR':<8} {'Acc':<8} {'N':<6}")
        print("-" * 42)
        for group, metrics in sorted(dim_data.items()):
            if metrics["total"] > 0:
                print(f"{group:<20} {metrics['fpr']:.2%}   {metrics['accuracy']:.2%}   {metrics['total']}")


def save_summary(analysis: dict, stratification: dict):
    """Save comprehensive summary to markdown file."""
    output = RESULTS_DIR / "summary.md"
    
    with open(output, "w") as f:
        f.write("# UI Assertion VLM Benchmark - Results Summary\n\n")
        
        f.write("## Overview\n\n")
        f.write("| Metric | Description |\n")
        f.write("|--------|-------------|\n")
        f.write("| FPR | False Positive Rate - predicted PASS when bug exists (critical!) |\n")
        f.write("| FNR | False Negative Rate - predicted FAIL when test should pass |\n")
        f.write("| Coverage | % of tests answered (not abstained) |\n")
        f.write("| Brier | Calibration score (lower = better, 0 = perfect) |\n\n")
        
        for profile, data in analysis.items():
            f.write(f"## Profile: `{profile}`\n\n")
            
            cm = data["confusion_matrix"]
            f.write("### Confusion Matrix\n\n")
            f.write("```\n")
            f.write(f"              Predicted\n")
            f.write(f"              PASS    FAIL\n")
            f.write(f"Actual PASS   {cm['tp']:4d}    {cm['fn']:4d}   (TPR={cm['tpr']:.2%})\n")
            f.write(f"Actual FAIL   {cm['fp']:4d}    {cm['tn']:4d}   (TNR={cm['tnr']:.2%})\n")
            f.write(f"\nAbstained: {cm['abstained']}\n")
            f.write("```\n\n")
            
            f.write("### Key Metrics\n\n")
            f.write(f"| Metric | Value |\n")
            f.write(f"|--------|-------|\n")
            f.write(f"| Accuracy | {cm['accuracy']:.2%} |\n")
            f.write(f"| **FPR** | **{cm['fpr']:.2%}** |\n")
            f.write(f"| FNR | {cm['fnr']:.2%} |\n")
            f.write(f"| Precision | {cm['precision']:.2%} |\n")
            f.write(f"| Recall | {cm['recall']:.2%} |\n")
            f.write(f"| F1 | {cm['f1']:.2%} |\n")
            f.write(f"| Coverage | {cm['coverage']:.2%} |\n\n")
            
            cal = data["calibration"]
            if cal["brier_score"] is not None:
                f.write("### Calibration\n\n")
                f.write(f"| Metric | Value |\n")
                f.write(f"|--------|-------|\n")
                f.write(f"| Brier Score | {cal['brier_score']:.4f} |\n")
                f.write(f"| ECE | {cal['ece']:.4f} |\n\n")
            
            f.write("### Performance by Confidence Threshold\n\n")
            f.write("| Threshold | FPR | FNR | Accuracy | Coverage |\n")
            f.write("|-----------|-----|-----|----------|----------|\n")
            for t, metrics in data["by_threshold"].items():
                if metrics["total"] > 0:
                    f.write(f"| {t} | {metrics['fpr']:.2%} | {metrics['fnr']:.2%} | "
                            f"{metrics['accuracy']:.2%} | {metrics['coverage']:.2%} |\n")
            f.write("\n")
            
            cost = data["cost"]
            if cost["api_calls"] > 0:
                f.write("### Cost Analysis\n\n")
                f.write(f"| Metric | Value |\n")
                f.write(f"|--------|-------|\n")
                f.write(f"| Total tokens | {cost['total_tokens']:,} |\n")
                f.write(f"| Avg latency | {cost['avg_latency_ms']:.0f}ms |\n")
                f.write(f"| Cost/eval | ${cost['cost_per_eval_usd']:.5f} |\n")
                f.write(f"| Total cost | ${cost['total_cost_usd']:.4f} |\n\n")
        
        f.write("---\n\n")
        f.write("## Stratification Analysis\n\n")
        f.write("> Performance breakdown by test characteristics\n\n")
        
        for dim_name, dim_data in stratification.items():
            f.write(f"### {dim_name.replace('by_', '').replace('_', ' ').title()}\n\n")
            f.write("| Group | FPR | Accuracy | N |\n")
            f.write("|-------|-----|----------|---|\n")
            for group, metrics in sorted(dim_data.items()):
                if metrics["total"] > 0:
                    f.write(f"| {group} | {metrics['fpr']:.2%} | {metrics['accuracy']:.2%} | {metrics['total']} |\n")
            f.write("\n")
        
        f.write("## Key Insights\n\n")
        f.write("- **Near-miss detection**: Compare `near_miss` FPR vs other difficulty drivers\n")
        f.write("- **Cognitive complexity**: Does L3 (semantic) have worse FPR than L1 (perceptive)?\n")
        f.write("- **Calibration**: Is confidence score useful? (low Brier = yes)\n")
        f.write("- **Cost efficiency**: Quality per $ across profiles\n")
    
    print(f"\nSummary saved to {output}")


def save_json_report(analysis: dict, stratification: dict):
    """Save machine-readable JSON report."""
    output = RESULTS_DIR / "metrics.json"
    
    report = {
        "by_profile": analysis,
        "stratification": stratification,
    }
    
    with open(output, "w") as f:
        json.dump(report, f, indent=2)
    
    print(f"JSON report saved to {output}")


def main():
    results = load_results()
    
    if not results:
        print("No results found. Run evaluation first.")
        return
    
    print(f"Loaded {len(results)} results")
    
    test_metadata = load_test_metadata()
    print(f"Loaded metadata for {len(test_metadata)} test cases")
    
    analysis = analyze_by_profile(results)
    stratification = stratify_results(results, test_metadata)
    
    print_summary(analysis, stratification)
    save_summary(analysis, stratification)
    save_json_report(analysis, stratification)


if __name__ == "__main__":
    main()
