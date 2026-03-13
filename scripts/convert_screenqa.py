#!/usr/bin/env python3
"""
Convert ScreenQA dataset to UI assertion benchmark format.

Takes QA pairs from Rico-ScreenQA and converts them into:
  - PASS assertions (correct answer derived from ground truth)
  - FAIL assertions (perturbed/wrong answer)
  - Proper tagging following tags_v1.yaml taxonomy

Output: dataset_screenqa/screenshots/<screen_id>/tests.json + screenshot.jpg
        dataset_screenqa/images.json

Usage:
    python scripts/convert_screenqa.py --n-screens 100 --max-tests-per-screen 10
    python scripts/convert_screenqa.py --n-screens 50 --split test --seed 42
"""

import argparse
import json
import os
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

from datasets import load_dataset


# ---------------------------------------------------------------------------
# Question classification → tag mapping
# ---------------------------------------------------------------------------

# Patterns that map ScreenQA question types to our tag taxonomy
QUESTION_PATTERNS = [
    # Counting questions
    (r"\bhow many\b", ["presence", "count_raw"]),
    (r"\bnumber of\b", ["presence", "count_raw"]),
    (r"\bcount\b", ["presence", "count_raw"]),
    # State / selection questions
    (r"\bselected\b", ["presence", "state"]),
    (r"\bactive\b", ["presence", "state"]),
    (r"\benabled\b", ["presence", "state"]),
    (r"\bdisabled\b", ["presence", "state"]),
    (r"\bchecked\b", ["presence", "state"]),
    (r"\btoggle\b", ["presence", "state"]),
    (r"\bunlocked\b", ["presence", "state"]),
    (r"\blocked\b", ["presence", "state"]),
    (r"\bcurrent mode\b", ["presence", "state"]),
    (r"\bcurrent tab\b", ["presence", "state"]),
    (r"\bwhich tab\b", ["presence", "state"]),
    # Layout / position questions
    (r"\bwhere\b", ["presence", "layout"]),
    (r"\bposition\b", ["presence", "layout"]),
    (r"\btop.?right\b", ["presence", "layout"]),
    (r"\btop.?left\b", ["presence", "layout"]),
    (r"\bbottom\b", ["presence", "layout"]),
    (r"\babove\b", ["presence", "layout"]),
    (r"\bbelow\b", ["presence", "layout"]),
    (r"\bnext to\b", ["presence", "layout"]),
    # List / enumeration questions (multiple elements)
    (r"\bwhat are the\b.*\boptions\b", ["presence", "count_raw"]),
    (r"\bwhat are the\b.*\bcategories\b", ["presence", "count_raw"]),
    (r"\bwhat are the\b.*\bitems\b", ["presence", "count_raw"]),
    (r"\blist\b", ["presence", "count_raw"]),
    # Text presence (most common — default)
    (r"\bwhat is the\b.*\bname\b", ["presence", "text_match_exact"]),
    (r"\bwhat is the\b.*\btitle\b", ["presence", "text_match_exact"]),
    (r"\bwhat is the\b.*\btext\b", ["presence", "text_match_exact"]),
    (r"\bwhat does\b.*\bsay\b", ["presence", "text_match_exact"]),
    (r"\bwhat is\b.*\bshown\b", ["presence", "text_match_exact"]),
    (r"\bwhat is\b.*\bdisplayed\b", ["presence", "text_match_exact"]),
    (r"\bsearch\b", ["presence", "text_match_exact"]),
]

# Default tags if no pattern matches
DEFAULT_TAGS = ["presence", "text_match_exact"]


def classify_question(question: str) -> list[str]:
    """Classify a ScreenQA question into assertion tags."""
    q_lower = question.lower()
    for pattern, tags in QUESTION_PATTERNS:
        if re.search(pattern, q_lower):
            return tags
    return DEFAULT_TAGS


# ---------------------------------------------------------------------------
# FAIL assertion generation (perturbation strategies)
# ---------------------------------------------------------------------------

def perturb_text(text: str) -> str | None:
    """Create a plausible but incorrect version of text."""
    if not text or text == "<no answer>":
        return None

    strategies = []

    # Strategy 1: swap case
    if text != text.lower() and text != text.upper():
        strategies.append(text.swapcase())

    # Strategy 2: remove a word (if multi-word)
    words = text.split()
    if len(words) > 1:
        idx = random.randint(0, len(words) - 1)
        strategies.append(" ".join(words[:idx] + words[idx + 1:]))

    # Strategy 3: numeric perturbation
    numbers = re.findall(r"\d+", text)
    if numbers:
        num = random.choice(numbers)
        n = int(num)
        delta = random.choice([-1, 1, 2, -2])
        new_n = max(0, n + delta)
        strategies.append(text.replace(num, str(new_n), 1))

    # Strategy 4: character swap (for short texts)
    if len(text) >= 4:
        chars = list(text)
        i = random.randint(0, len(chars) - 2)
        chars[i], chars[i + 1] = chars[i + 1], chars[i]
        strategies.append("".join(chars))

    # Strategy 5: replace with a plausible alternative
    alternatives = {
        "yes": "no", "no": "yes",
        "on": "off", "off": "on",
        "true": "false", "false": "true",
        "enabled": "disabled", "disabled": "enabled",
    }
    text_lower = text.lower().strip()
    if text_lower in alternatives:
        strategies.append(alternatives[text_lower])

    if not strategies:
        return None

    return random.choice(strategies)


def perturb_number(text: str) -> str | None:
    """Specifically perturb numeric answers."""
    numbers = re.findall(r"\d+", text)
    if not numbers:
        return None
    num = numbers[0]
    n = int(num)
    delta = random.choice([-1, 1, 2, -2, 3])
    new_n = max(0, n + delta)
    if new_n == n:
        new_n = n + 1
    return text.replace(num, str(new_n), 1)


# ---------------------------------------------------------------------------
# Assertion generation from QA pairs
# ---------------------------------------------------------------------------

def extract_key_text(ground_truth: list[dict]) -> str | None:
    """Extract the key answer text from ground truth annotations."""
    for gt in ground_truth:
        if gt.get("full_answer", "") == "<no answer>":
            continue
        # Prefer UI element text (more precise)
        elements = gt.get("ui_elements", [])
        if elements:
            texts = [e["text"] for e in elements if e.get("text")]
            if texts:
                return texts[0]  # Primary element text
        # Fall back to extracting quoted text from full_answer
        full = gt.get("full_answer", "")
        quoted = re.findall(r'"([^"]+)"', full)
        if quoted:
            return quoted[0]
    return None


def extract_all_element_texts(ground_truth: list[dict]) -> list[str]:
    """Extract all unique UI element texts from ground truth."""
    texts = set()
    for gt in ground_truth:
        for el in gt.get("ui_elements", []):
            t = el.get("text", "").strip()
            if t and t != "<no answer>":
                texts.add(t)
    return list(texts)


def make_pass_assertion(question: str, ground_truth: list[dict],
                        tags: list[str]) -> dict | None:
    """Create a PASS assertion from a QA pair."""
    key_text = extract_key_text(ground_truth)
    if not key_text:
        return None

    # Generate assertion text based on question type
    q_lower = question.lower()

    if re.search(r"\bhow many\b|\bnumber of\b|\bcount\b", q_lower):
        # Counting assertion
        numbers = re.findall(r"\d+", key_text)
        if numbers:
            assertion = f"Verify that the text '{key_text}' is visible on the screen."
        else:
            # Extract number from full answer
            for gt in ground_truth:
                full = gt.get("full_answer", "")
                nums = re.findall(r"\d+", full)
                if nums:
                    assertion = f"Verify that '{nums[0]}' is displayed, representing the count related to: {question}"
                    break
            else:
                assertion = f"Verify that the text '{key_text}' is visible on the screen."

    elif re.search(r"\bselected\b|\bactive\b|\bcurrent\b|\btab\b|\bunlocked\b|\blocked\b|\bmode\b", q_lower):
        # State assertion
        assertion = f"Verify that '{key_text}' is the currently selected/active element."

    elif re.search(r"\bwhere\b|\bposition\b", q_lower):
        # Layout assertion
        assertion = f"Verify that '{key_text}' is visible on the screen."

    elif re.search(r"\bwhat are\b.*\b(options|categories|items|types)\b", q_lower):
        # List/enumeration
        all_texts = extract_all_element_texts(ground_truth)
        if len(all_texts) > 1:
            items = "', '".join(all_texts)
            assertion = f"Verify that the items '{items}' are all visible on the screen."
        else:
            assertion = f"Verify that '{key_text}' is visible on the screen."

    else:
        # Default: text presence
        assertion = f"Verify that the exact text '{key_text}' is visible on the screen."

    return {
        "assertion": assertion,
        "expected": "PASS",
        "tags": tags,
        "_source_question": question,
        "_source_answer": key_text,
    }


def make_fail_assertion(question: str, ground_truth: list[dict],
                        tags: list[str]) -> dict | None:
    """Create a FAIL assertion (perturbed answer) from a QA pair."""
    key_text = extract_key_text(ground_truth)
    if not key_text:
        return None

    q_lower = question.lower()

    # Choose perturbation strategy based on question type
    if re.search(r"\bhow many\b|\bnumber of\b|\bcount\b", q_lower):
        # For counting, perturb the number
        for gt in ground_truth:
            full = gt.get("full_answer", "")
            perturbed = perturb_number(full)
            if perturbed:
                nums_orig = re.findall(r"\d+", full)
                nums_new = re.findall(r"\d+", perturbed)
                if nums_orig and nums_new and nums_orig[0] != nums_new[0]:
                    assertion = f"Verify that '{nums_new[0]}' is displayed as the count related to: {question}"
                    return {
                        "assertion": assertion,
                        "expected": "FAIL",
                        "tags": tags + ["near_miss"],
                        "_source_question": question,
                        "_source_answer": key_text,
                        "_perturbation": f"{nums_orig[0]} → {nums_new[0]}",
                    }

    # General text perturbation
    perturbed = perturb_text(key_text)
    if not perturbed or perturbed == key_text:
        return None

    assertion = f"Verify that the exact text '{perturbed}' is visible on the screen."
    return {
        "assertion": assertion,
        "expected": "FAIL",
        "tags": tags + ["near_miss"],
        "_source_question": question,
        "_source_answer": key_text,
        "_perturbation": f"'{key_text}' → '{perturbed}'",
    }


def make_absence_assertion(ground_truth: list[dict]) -> dict | None:
    """Create an absence assertion — assert something that IS present is absent."""
    key_text = extract_key_text(ground_truth)
    if not key_text:
        return None

    assertion = f"Verify that the text '{key_text}' is NOT visible on the screen."
    return {
        "assertion": assertion,
        "expected": "FAIL",
        "tags": ["absence", "text_match_exact"],
        "_source_answer": key_text,
        "_perturbation": "absence of present element",
    }


def make_unanswerable_pass(question: str) -> dict:
    """For questions with <no answer>, the assertion that it's absent should PASS."""
    return {
        "assertion": f"Verify that the screen does NOT contain a direct answer to: '{question}'",
        "expected": "PASS",
        "tags": ["absence"],
        "_source_question": question,
        "_perturbation": "unanswerable question",
    }


# ---------------------------------------------------------------------------
# Main conversion pipeline
# ---------------------------------------------------------------------------

def is_answerable(ground_truth: list[dict]) -> bool:
    """Check if at least one annotator provided a real answer."""
    for gt in ground_truth:
        if gt.get("full_answer", "") != "<no answer>":
            elements = gt.get("ui_elements", [])
            if elements:
                return True
    return False


def convert_screen(screen_id: str, qa_pairs: list[dict],
                   max_tests: int, rng: random.Random) -> list[dict]:
    """Convert all QA pairs for one screen into assertions."""
    tests = []
    test_idx = 1

    # Shuffle QA pairs for variety
    pairs = list(qa_pairs)
    rng.shuffle(pairs)

    for pair in pairs:
        if len(tests) >= max_tests:
            break

        question = pair["question"]
        ground_truth = pair["ground_truth"]

        if not is_answerable(ground_truth):
            # Unanswerable question → absence assertion
            t = make_unanswerable_pass(question)
            t["test_id"] = f"rico{screen_id}_{test_idx:02d}"
            tests.append(t)
            test_idx += 1
            continue

        tags = classify_question(question)

        # Generate PASS assertion
        pass_test = make_pass_assertion(question, ground_truth, tags)
        if pass_test and len(tests) < max_tests:
            pass_test["test_id"] = f"rico{screen_id}_{test_idx:02d}"
            tests.append(pass_test)
            test_idx += 1

        # Generate FAIL assertion (perturbed) — ~50% of the time
        if rng.random() < 0.5 and len(tests) < max_tests:
            fail_test = make_fail_assertion(question, ground_truth, tags)
            if fail_test:
                fail_test["test_id"] = f"rico{screen_id}_{test_idx:02d}"
                tests.append(fail_test)
                test_idx += 1

        # Occasionally generate absence assertion (~20% of the time)
        if rng.random() < 0.2 and len(tests) < max_tests:
            abs_test = make_absence_assertion(ground_truth)
            if abs_test:
                abs_test["test_id"] = f"rico{screen_id}_{test_idx:02d}"
                tests.append(abs_test)
                test_idx += 1

    return tests


def main():
    parser = argparse.ArgumentParser(
        description="Convert ScreenQA to UI assertion benchmark format"
    )
    parser.add_argument("--n-screens", type=int, default=100,
                        help="Number of screens to include (default: 100)")
    parser.add_argument("--max-tests-per-screen", type=int, default=10,
                        help="Max assertions per screenshot (default: 10)")
    parser.add_argument("--split", default="test",
                        choices=["train", "validation", "test"],
                        help="ScreenQA split to use (default: test)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory (default: dataset_screenqa)")
    parser.add_argument("--min-qa-per-screen", type=int, default=3,
                        help="Min QA pairs to include a screen (default: 3)")
    args = parser.parse_args()

    rng = random.Random(args.seed)

    # Output paths
    base_dir = Path(__file__).parent.parent
    output_dir = Path(args.output_dir) if args.output_dir else base_dir / "dataset_screenqa"
    screenshots_dir = output_dir / "screenshots"
    screenshots_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading ScreenQA ({args.split} split)...")
    ds = load_dataset("rootsautomation/RICO-ScreenQA", split=args.split)

    # Group QA pairs by screen_id
    screen_qa = defaultdict(list)
    for example in ds:
        sid = example["screen_id"]
        screen_qa[sid].append({
            "question": example["question"],
            "ground_truth": example["ground_truth"],
            "image": example["image"],
            "file_name": example["file_name"],
        })

    # Filter screens with enough QA pairs
    eligible = {
        sid: pairs for sid, pairs in screen_qa.items()
        if len(pairs) >= args.min_qa_per_screen
    }
    print(f"Eligible screens (>= {args.min_qa_per_screen} QA pairs): "
          f"{len(eligible)} / {len(screen_qa)}")

    # Sample screens
    screen_ids = sorted(eligible.keys(), key=int)
    if len(screen_ids) > args.n_screens:
        screen_ids = rng.sample(screen_ids, args.n_screens)
        screen_ids.sort(key=int)

    print(f"Selected {len(screen_ids)} screens")

    # Convert and save
    images_meta = []
    total_tests = 0
    total_pass = 0
    total_fail = 0
    tag_counts = defaultdict(int)

    for i, sid in enumerate(screen_ids):
        pairs = eligible[sid]

        # Generate assertions
        tests = convert_screen(sid, pairs, args.max_tests_per_screen, rng)
        if not tests:
            continue

        # Clean internal fields for output (keep for debug)
        tests_clean = []
        for t in tests:
            clean = {
                "test_id": t["test_id"],
                "assertion": t["assertion"],
                "expected": t["expected"],
                "tags": t["tags"],
            }
            # Keep source metadata as optional fields
            if "_source_question" in t:
                clean["_source_question"] = t["_source_question"]
            if "_perturbation" in t:
                clean["_perturbation"] = t["_perturbation"]
            tests_clean.append(clean)

            # Stats
            if t["expected"] == "PASS":
                total_pass += 1
            else:
                total_fail += 1
            for tag in t["tags"]:
                tag_counts[tag] += 1

        total_tests += len(tests_clean)

        # Save screenshot
        screen_dir = screenshots_dir / f"rico_{sid}"
        screen_dir.mkdir(exist_ok=True)

        # Save image from first QA pair
        img = pairs[0]["image"]
        img_path = screen_dir / "screenshot.jpg"
        if not img_path.exists():
            img.save(str(img_path))

        # Save tests.json
        tests_path = screen_dir / "tests.json"
        with open(tests_path, "w") as f:
            json.dump(tests_clean, f, indent=2, ensure_ascii=False)

        # Build image metadata
        images_meta.append({
            "image_id": f"rico_{sid}",
            "file": f"screenshots/rico_{sid}/screenshot.jpg",
            "env": {
                "ui_type": "mobile",
                "platform": "android",
                "device": "unknown",
                "locale": "en_US",
                "theme": "unknown",
            },
            "app": {
                "name": "unknown",
                "build": "unknown",
            },
            "source": {
                "dataset": "ScreenQA",
                "split": args.split,
                "screen_id": sid,
                "rico_file": pairs[0].get("file_name", ""),
                "n_qa_pairs": len(pairs),
            },
            "notes": [],
        })

        if (i + 1) % 20 == 0:
            print(f"  Processed {i + 1}/{len(screen_ids)} screens "
                  f"({total_tests} tests so far)")

    # Save images.json
    images_path = output_dir / "images.json"
    with open(images_path, "w") as f:
        json.dump(images_meta, f, indent=2, ensure_ascii=False)

    # Copy tags_v1.yaml
    tags_src = base_dir / "dataset" / "tags_v1.yaml"
    tags_dst = output_dir / "tags_v1.yaml"
    if tags_src.exists():
        import shutil
        shutil.copy2(tags_src, tags_dst)

    # Print summary
    print(f"\n{'=' * 60}")
    print(f"Conversion complete!")
    print(f"{'=' * 60}")
    print(f"Output dir:       {output_dir}")
    print(f"Screenshots:      {len(images_meta)}")
    print(f"Total assertions: {total_tests}")
    print(f"  PASS:           {total_pass} ({100*total_pass/max(total_tests,1):.1f}%)")
    print(f"  FAIL:           {total_fail} ({100*total_fail/max(total_tests,1):.1f}%)")
    print(f"\nTag distribution:")
    for tag, count in sorted(tag_counts.items(), key=lambda x: -x[1]):
        print(f"  {tag:25s} {count:5d} ({100*count/max(total_tests,1):.1f}%)")


if __name__ == "__main__":
    main()
