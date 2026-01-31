# Tagging & Taxonomy
This is the tags used as V1
Flat tags, multiple tags per test. Cognitive level is derived.

## Operation Tags
- `presence` / `absence`: Element existence.
- `text_match_exact`: Character-by-character.
- `text_match_normalized`: Normalizing case/whitespace.
- `text_match_semantic`: Open interpretation.
- `count_raw`: Count all visible.
- `count_filtered`: Count with condition.
- `state`: Widget status (on/off, enabled).
- `layout`: Spatial relationships (left-of, above).
- `order`: Sorting (alphabetical, numeric).
- `consistency`: Cross-field validation / business rules.

## Difficulty Tags
- `near_miss`: Subtle differences / typos.
- `small_text`: Requires high OCR precision.
- `low_contrast`: Poor visibility.
- `cluttered`: Busy screen.
- `occluded`: Partially hidden.
- `confusable`: Similar looking elements.

## Tagging Rules
1. Add ALL applicable operation tags.
2. Add difficulty tags ONLY if they apply.
3. No manual cognitive level tags (derived).
4. No polarity tags (derived from expected).

## Derived Metrics
- **L1**: Single operations (`presence`, `state`, etc.).
- **L2**: `layout`, `order`, `count_filtered`, or multiple operations.
- **L3**: `consistency` or `text_match_semantic`.
