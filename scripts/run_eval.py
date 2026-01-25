#!/usr/bin/env python3
"""
Benchmark runner for UI assertion evaluation.
Loads tests from per-screenshot folders and runs against all system prompts.
Uses ImgBB for image hosting (cheaper) with base64 fallback.
"""

import json
import os
import re
import time
from pathlib import Path
from datetime import datetime

# Load .env file
from dotenv import load_dotenv
load_dotenv()

import openai

from imgbb_uploader import get_image_for_api

# Configuration
MODEL = os.getenv("VLM_MODEL", "gpt-4.1-mini")
DATASET_DIR = Path(__file__).parent.parent / "dataset" / "screenshots"
PROMPTS_DIR = Path(__file__).parent.parent / "prompts" / "system_prompts"
RESULTS_DIR = Path(__file__).parent.parent / "results"

RESULTS_DIR.mkdir(exist_ok=True)


def discover_test_folders() -> list[Path]:
    """Find all screenshot folders that have a tests.json file."""
    folders = []
    for item in sorted(DATASET_DIR.iterdir()):
        if item.is_dir():
            tests_file = item / "tests.json"
            if tests_file.exists():
                folders.append(item)
    return folders


def load_tests_from_folder(folder: Path) -> list[dict]:
    """Load tests from a screenshot folder's tests.json."""
    tests_file = folder / "tests.json"
    with open(tests_file, "r") as f:
        tests = json.load(f)
    
    # Add image_id from folder name
    image_id = folder.name
    for test in tests:
        test["image_id"] = image_id
    
    return tests


def load_system_prompts() -> dict[str, str]:
    """Load all system prompt templates."""
    prompts = {}
    for prompt_file in sorted(PROMPTS_DIR.glob("*.txt")):
        prompts[prompt_file.stem] = prompt_file.read_text()
    return prompts


def find_screenshot(folder: Path) -> Path:
    """Find the screenshot image in a folder."""
    for ext in ["png", "jpg", "jpeg", "webp"]:
        for name in ["screenshot", folder.name]:
            candidate = folder / f"{name}.{ext}"
            if candidate.exists():
                return candidate
    # Fallback: find any image
    for ext in ["png", "jpg", "jpeg", "webp"]:
        images = list(folder.glob(f"*.{ext}"))
        if images:
            return images[0]
    raise FileNotFoundError(f"No screenshot found in {folder}")


def call_vlm(image_path: Path, system_prompt: str, assertion: str, image_cache: dict) -> dict:
    """Call VLM with image and prompt."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable is not set.")
    
    client = openai.OpenAI(api_key=api_key)
    
    # Get image (from cache or upload/encode)
    cache_key = str(image_path)
    if cache_key not in image_cache:
        image_cache[cache_key] = get_image_for_api(image_path)
    
    image_content = image_cache[cache_key]
    image_method = image_content.pop("_method", "unknown")
    
    # Build messages
    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": [
                image_content,
                {"type": "text", "text": f"Assertion: {assertion}"}
            ]
        }
    ]
    
    max_retries = 2
    last_error = None
    
    for attempt in range(max_retries):
        try:
            start_time = time.time()
            response = client.chat.completions.create(
                model=MODEL,
                messages=messages,
                max_tokens=500,
                temperature=0.0
            )
            latency_ms = int((time.time() - start_time) * 1000)
            break  # Success
        except Exception as e:
            last_error = e
            error_msg = str(e).lower()
            
            # If ImgBB URL failed, retry with base64
            if attempt == 0 and image_method == "imgbb" and ("timeout" in error_msg or "invalid_image_url" in error_msg):
                print(f"\n⚠️  ImgBB URL failed, retrying with base64...", end=" ")
                
                # Force base64 fallback
                import base64 as b64
                with open(image_path, "rb") as f:
                    base64_data = b64.b64encode(f.read()).decode("utf-8")
                suffix = image_path.suffix.lower()
                mime_types = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".webp": "image/webp"}
                mime_type = mime_types.get(suffix, "image/png")
                
                fallback_content = {
                    "type": "image_url",
                    "image_url": {"url": f"data:{mime_type};base64,{base64_data}"}
                }
                messages[1]["content"][0] = fallback_content
                image_method = "base64_fallback"
                continue
            else:
                raise
    else:
        raise last_error
    
    raw_text = response.choices[0].message.content
    usage = response.usage
    
    # Parse response
    result = None
    confidence = None
    evidence = None
    reasoning = None
    
    # Try to parse JSON
    json_match = re.search(r'\{[^}]+\}', raw_text, re.DOTALL)
    if json_match:
        try:
            parsed = json.loads(json_match.group())
            result = parsed.get("result")
            confidence = parsed.get("confidence")
            evidence = parsed.get("evidence")
            reasoning = parsed.get("reasoning")
        except json.JSONDecodeError:
            pass
    
    # Fallback: extract PASS/FAIL or true/false from text
    if result is None:
        text_upper = raw_text.upper()
        if "PASS" in text_upper:
            result = "PASS"
        elif "FAIL" in text_upper:
            result = "FAIL"
        elif "TRUE" in text_upper:
            result = "PASS"
        elif "FALSE" in text_upper:
            result = "FAIL"
    
    # Restore _method for next use
    image_cache[cache_key]["_method"] = image_method
    
    return {
        "result": result,
        "confidence": confidence,
        "evidence": evidence,
        "reasoning": reasoning,
        "raw": raw_text,
        "image_method": image_method,
        "cost": {
            "input_tokens": usage.prompt_tokens if usage else 0,
            "output_tokens": usage.completion_tokens if usage else 0,
            "latency_ms": latency_ms,
            "api_calls": 1
        }
    }


def run_evaluation():
    """Run all tests × all prompts."""
    folders = discover_test_folders()
    prompts = load_system_prompts()
    
    # Count total tests
    total_tests = 0
    for folder in folders:
        tests = load_tests_from_folder(folder)
        total_tests += len(tests)
    
    print(f"Discovered {len(folders)} screenshot folders with tests")
    print(f"Total tests: {total_tests}")
    print(f"System prompts: {list(prompts.keys())}")
    print(f"Total evaluations: {total_tests * len(prompts)}")
    print(f"Model: {MODEL}")
    
    # Check ImgBB availability
    imgbb_key = os.getenv("IMGBB_API_KEY")
    if imgbb_key:
        print("🖼️  ImgBB: enabled (using URL upload)")
    else:
        print("⚠️  ImgBB: disabled (using base64 - set IMGBB_API_KEY to reduce costs)")
    
    print("-" * 60)
    
    results_file = RESULTS_DIR / "raw.jsonl"
    image_cache = {}  # Cache uploaded image URLs per screenshot
    
    with open(results_file, "w") as f:
        for folder in folders:
            tests = load_tests_from_folder(folder)
            image_path = find_screenshot(folder)
            
            print(f"\n📁 {folder.name} ({len(tests)} tests)")
            
            for test in tests:
                for prompt_name, system_prompt in prompts.items():
                    test_id = test.get("test_id", test.get("id", "unknown"))
                    assertion = test.get("assertion", test.get("assertion_text", ""))
                    expected = test.get("expected", test.get("gt_label", ""))
                    
                    print(f"  {test_id} × {prompt_name}...", end=" ", flush=True)
                    
                    try:
                        response = call_vlm(image_path, system_prompt, assertion, image_cache)
                        
                        record = {
                            "test_id": test_id,
                            "image_id": test["image_id"],
                            "profile": prompt_name,
                            "model": MODEL,
                            "assertion": assertion,
                            "expected": expected,
                            "result": response["result"],
                            "confidence": response["confidence"],
                            "evidence": response["evidence"],
                            "reasoning": response["reasoning"],
                            "tags": test.get("tags", []),
                            "image_method": response["image_method"],
                            "cost": response["cost"],
                            "timestamp": datetime.now().isoformat()
                        }
                        
                        f.write(json.dumps(record) + "\n")
                        f.flush()
                        
                        status = "✓" if response["result"] == expected else "✗"
                        print(f"{status} {response['result']} (expected {expected})")
                        
                    except Exception as e:
                        print(f"ERROR: {e}")
                        record = {
                            "test_id": test_id,
                            "image_id": test["image_id"],
                            "profile": prompt_name,
                            "model": MODEL,
                            "assertion": assertion,
                            "expected": expected,
                            "result": None,
                            "error": str(e),
                            "tags": test.get("tags", []),
                            "timestamp": datetime.now().isoformat()
                        }
                        f.write(json.dumps(record) + "\n")
                        f.flush()
    
    print("\n" + "-" * 60)
    print(f"✅ Results saved to {results_file}")


if __name__ == "__main__":
    run_evaluation()
