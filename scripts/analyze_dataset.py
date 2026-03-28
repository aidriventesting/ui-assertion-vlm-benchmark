#!/usr/bin/env python3
import json
import os
from pathlib import Path
from collections import defaultdict

DATASET_DIR = Path(__file__).parent.parent / "dataset" / "screenshots"

OPERATION_TAGS = {
    "presence", "absence", "text_match_exact", "text_match_normalized", 
    "text_match_semantic", "count_raw", "count_filtered", "state", 
    "layout", "order", "consistency"
}

DIFFICULTY_TAGS = {
    "near_miss", "small_text", "low_contrast", "cluttered", "occluded", "confusable"
}

def get_cognitive_level(tags):
    # L3: High level reasoning
    if "consistency" in tags or "text_match_semantic" in tags:
        return "L3 (Reasoning)"
    
    # L2: Spatial or multi-step
    ops = [t for t in tags if t in OPERATION_TAGS]
    if "layout" in tags or "order" in tags or "count_filtered" in tags or len(ops) > 1:
        return "L2 (Spatial/Multi)"
    
    # L1: Pure perception
    return "L1 (Perception)"

def compute_dataset_stats(dataset_dir=None):
    """Aggregate tag, outcome, and cognitive-level stats from dataset/screenshots.
    Returns a dict with total_shots, total_tests, outcomes, tags, levels.
    """
    base = dataset_dir if dataset_dir is not None else DATASET_DIR
    base = Path(base)
    stats = {
        "total_shots": 0,
        "total_tests": 0,
        "outcomes": defaultdict(int),
        "tags": defaultdict(int),
        "levels": defaultdict(int)
    }
    for shot_dir in sorted(base.iterdir()):
        if not shot_dir.is_dir():
            continue
        tests_file = shot_dir / "tests.json"
        if not tests_file.exists():
            continue
        stats["total_shots"] += 1
        with open(tests_file, "r") as f:
            tests = json.load(f)
        for test in tests:
            stats["total_tests"] += 1
            expected = test.get("expected", "UNKNOWN").upper()
            stats["outcomes"][expected] += 1
            tags = test.get("tags", [])
            for tag in tags:
                stats["tags"][tag] += 1
            level = get_cognitive_level(tags)
            stats["levels"][level] += 1
    return stats


def analyze():
    return compute_dataset_stats()

def print_markdown(stats):
    print("# Dataset Insights\n")
    print(f"- **Total Screenshots**: {stats['total_shots']}")
    print(f"- **Total Test Cases**: {stats['total_tests']}")
    print(f"- **Balance**: {stats['outcomes']['PASS']} PASS / {stats['outcomes']['FAIL']} FAIL")
    print("\n## Tag Distribution\n")
    print("| Tag | Count | % |")
    print("|-----|-------|---|")
    sorted_tags = sorted(stats["tags"].items(), key=lambda x: x[1], reverse=True)
    for tag, count in sorted_tags:
        perc = (count / stats["total_tests"]) * 100
        print(f"| {tag} | {count} | {perc:.1f}% |")
    
    print("\n## Cognitive Levels\n")
    print("```mermaid")
    print("pie title Cognitive Levels")
    for level, count in stats["levels"].items():
        print(f'    "{level}" : {count}')
    print("```")

    print("\n## Tag Frequency (Mermaid Bar Chart)\n")
    print("```mermaid")
    print("xychart-beta")
    print("    title \"Tag Frequency\"")
    print(f"    x-axis [{', '.join([f'\"{t}\"' for t, _ in sorted_tags[:10]])}]")
    print(f"    y-axis \"Count\" 0 --> {max(stats['tags'].values()) + 5}")
    print(f"    bar [{', '.join([str(c) for _, c in sorted_tags[:10]])}]")
    print("```")

if __name__ == "__main__":
    results = analyze()
    print_markdown(results)
