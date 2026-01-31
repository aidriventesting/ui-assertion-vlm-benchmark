# Prompt Engineering

The benchmark uses specialized prompts organized by directory.

## Directory Structure
- `prompts/personas/`: Detailed prompts for high-token **JSON** output (includes reasoning/evidence).
- `prompts/scoring/`: Concise prompts for 1-token **ABC** output (A=PASS, B=FAIL, C=UNCLEAR).
- `prompts/legacy/`: Prompts that include a profile for QA, ask for PAASS/FAIL and verbal confidence.

## How it's run
The `run_eval.py` script:
1. Loads all `.txt` files in the directory specified by `prompt_dir`.
2. Runs the evaluation for **each** prompt found in that folder.

