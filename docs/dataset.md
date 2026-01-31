# Dataset Structure

Each test case is located in its own directory under `dataset/screenshots/`.

## Directory Layout
```text
dataset/screenshots/<shot_id>/
├── screenshot.png  # UI Image
└── tests.json      # Assertions to verify
```

## `tests.json` Schema
```json
[
  {
    "test_id": "shot001_01",
    "assertion": "Is the 'Next' button visible?",
    "expected": "PASS",
    "tags": ["presence", "text_match_exact"]
  }
]
```
