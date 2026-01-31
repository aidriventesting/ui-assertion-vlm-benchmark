# Experiments

Evaluations are driven by YAML configuration files in the `experiments/` directory.

## YAML Schema
```yaml
provider: openai        # openai | gemini | anthropic
model: gpt-4o-mini      # model id
prompt_dir: personas    # folder under prompts/
temperature: 0          # sampling temp
max_tokens: 500         # max output tokens
logprobs: true          # request logprobs (calibration)
output_format: json     # json (reasoning) | abc (classification)
```

## Running Evaluations
```bash
env/bin/python3 scripts/run_eval.py --config experiments/gpt4o-mini_personas.yaml
```
- Parameters like `logprobs` and `output_format` are passed directly to the provider.
- `output_format: abc` will force the model to output a single token (A/B/C).
