#!/usr/bin/env python3
"""
Benchmark runner for UI assertion evaluation.
Loads tests from per-screenshot folders and runs against all system prompts.
Supports multiple VLM providers: OpenAI, Gemini, Anthropic.
"""

import argparse
import json
import os
from pathlib import Path
from datetime import datetime

# Load .env file
from dotenv import load_dotenv
load_dotenv()

from providers import get_provider, get_all_providers, PROVIDERS

# Configuration
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


def run_evaluation(provider_names: list[str], model_override: str = None):
    """Run all tests × all prompts × selected providers."""
    folders = discover_test_folders()
    prompts = load_system_prompts()
    
    # Initialize providers
    providers = {}
    for name in provider_names:
        try:
            provider = get_provider(name)
            # Override model if specified
            if model_override:
                provider.model = model_override
            providers[name] = provider
            print(f"✓ {name}: {providers[name].model}")
        except ValueError as e:
            print(f"✗ {name}: {e}")
        except ImportError as e:
            print(f"✗ {name}: {e}")
    
    if not providers:
        print("\n❌ No providers available. Check your API keys in .env")
        return
    
    # Count total tests
    total_tests = 0
    for folder in folders:
        tests = load_tests_from_folder(folder)
        total_tests += len(tests)
    
    total_evals = total_tests * len(prompts) * len(providers)
    
    print(f"\nDiscovered {len(folders)} screenshot folders with tests")
    print(f"Total tests: {total_tests}")
    print(f"System prompts: {list(prompts.keys())}")
    print(f"Providers: {list(providers.keys())}")
    print(f"Total evaluations: {total_evals}")
    print("-" * 60)
    
    # Generate timestamped filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_suffix = model_override.replace("/", "-") if model_override else "default"
    provider_suffix = "_".join(providers.keys())
    results_file = RESULTS_DIR / f"raw_{timestamp}_{provider_suffix}_{model_suffix}.jsonl"
    
    # Also create a symlink to latest for convenience
    latest_link = RESULTS_DIR / "raw.jsonl"
    if latest_link.exists() or latest_link.is_symlink():
        latest_link.unlink()
    latest_link.symlink_to(results_file.name)
    
    with open(results_file, "w") as f:
        for folder in folders:
            tests = load_tests_from_folder(folder)
            image_path = find_screenshot(folder)
            
            print(f"\n📁 {folder.name} ({len(tests)} tests)")
            
            for test in tests:
                for prompt_name, system_prompt in prompts.items():
                    for provider_name, provider in providers.items():
                        test_id = test.get("test_id", test.get("id", "unknown"))
                        assertion = test.get("assertion", test.get("assertion_text", ""))
                        expected = test.get("expected", test.get("gt_label", ""))
                        
                        model_name = provider.get_model_name()
                        print(f"  {test_id} × {prompt_name} × {provider_name}...", end=" ", flush=True)
                        
                        try:
                            response = provider.call(image_path, system_prompt, assertion)
                            
                            record = {
                                "test_id": test_id,
                                "image_id": test["image_id"],
                                "profile": prompt_name,
                                "provider": provider_name,
                                "model": model_name,
                                "assertion": assertion,
                                "expected": expected,
                                "result": response.result,
                                "confidence": response.confidence,
                                "evidence": response.evidence,
                                "reasoning": response.reasoning,
                                "tags": test.get("tags", []),
                                "cost": response.cost,
                                "timestamp": datetime.now().isoformat()
                            }
                            
                            f.write(json.dumps(record) + "\n")
                            f.flush()
                            
                            status = "✓" if response.result == expected else "✗"
                            print(f"{status} {response.result} (expected {expected})")
                            
                        except Exception as e:
                            print(f"ERROR: {e}")
                            record = {
                                "test_id": test_id,
                                "image_id": test["image_id"],
                                "profile": prompt_name,
                                "provider": provider_name,
                                "model": model_name,
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


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run UI assertion benchmark with VLM providers"
    )
    parser.add_argument(
        "--provider", "-p",
        type=str,
        default="openai",
        help=f"Provider to use: {', '.join(PROVIDERS.keys())}, or 'all' for all providers"
    )
    parser.add_argument(
        "--model", "-m",
        type=str,
        default=None,
        help="Override model name (e.g., gpt-4o-mini, gpt-4o, claude-3-5-sonnet-20241022)"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    if args.provider == "all":
        provider_names = list(PROVIDERS.keys())
    else:
        provider_names = [p.strip() for p in args.provider.split(",")]
    
    run_evaluation(provider_names, model_override=args.model)
