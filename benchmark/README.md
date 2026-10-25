# Annotator Benchmark

Measures annotator accuracy, latency, and cost on frozen targets and human-reviewed page labels,
so different annotators (for example `LLMAnnotator` and `JevAnnotator`) can be compared without
extraction variance. It is run by hand, not by `pytest`, because it calls paid APIs.

## Prerequisites

Put provider keys in `.env` at the repo root: `ANTHROPIC_API_KEY` for silver labeling and
`OPENAI_API_KEY` for the `openai:gpt-4o` baseline.

## Workflow

1. **Silver-label** every page with a strong model (default `anthropic:claude-opus-5-5`):

   ```bash
   poetry run python -m benchmark.label [dataset ...] [--model MODEL]
   ```

   This fills `pages` in `benchmark/data/*.json` and also records an alias string match (each
   target's `name` and `known_as`, case-insensitive, word-bounded). Pages that are already
   labeled are left alone, so re-running only labels new pages. If some pages fail (for
   example on a rate limit), the others are still saved; re-run to retry the failures.

2. **Review** `benchmark/review/<dataset>.md`. It lists every page where the silver labels and
   the string match disagree. For each one, fix `labels` in the dataset JSON and set
   `"reviewed": true`. This step is mandatory: unreviewed silver labels would bias the
   benchmark toward the LLM annotator.

3. **Run** an annotator:

   ```bash
   poetry run python -m benchmark.run --annotator llm --model openai:gpt-4o
   ```

   `run.py` refuses to run while any disagreement page is unreviewed, or while any page's
   text no longer matches its `text_hash`. It writes
   `benchmark/results/YYYY-MM-DD-<annotator>-<model>-<gold digest>.json` and regenerates
   `RESULTS.md`. The gold digest fingerprints the selected datasets, page text, and labels;
   ΔF1 is only reported against a baseline run with the same digest.
   Use `--price-in` and `--price-out` (USD per million tokens) for models missing from the
   pricing table in `usage.py`.

4. **Commit** the results JSON and `RESULTS.md`.

## Data format

```json
{
  "name": "christmas_carol",
  "source": "sample_data/christmas_carol.txt",
  "segmentor": {"type": "PageSegmentor", "paragraphs_per_page": 10},
  "sample": {"size": 60, "seed": 1843},
  "target_class": "Character",
  "targets": [{"name": "Ebenezer Scrooge", "known_as": ["Scrooge", "Ebenezer"], "...": "..."}],
  "pages": [
    {"index": 0, "text_hash": "9f2c…", "labels": [0, 2], "silver": [0, 2], "string_match": [0], "reviewed": true}
  ]
}
```

Page text is regenerated from `source` and `segmentor` rather than stored; `text_hash` records
a fingerprint of it at labeling time, so a changed source can't move labels onto other text. `sample` picks a
fixed, seeded subset of pages (`null` means every page). `targets` are hand-curated and frozen;
`labels` are indices into them.

## Metrics

- Micro-averaged precision, recall, and F1 over every (page, target) pair, overall and per
  dataset.
- Wall-clock latency, total and mean per page.
- Token usage and estimated cost.
- Annotators that expose `annotate_scores_async` are swept over thresholds 0.10–0.90 (step
  0.05) and every aggregator. Their precision, recall, and F1 are 2-fold cross-validated:
  pages alternate between two folds (so each fold spans every dataset), and each fold is
  scored with the best configuration on the other fold. Tuning and scoring on the same pages
  would inflate F1. The best configuration on all pages is reported as the one to use, and the
  results JSON also saves its (optimistic) in-sample F1, the per-fold configurations, the full
  PR curve, and every page's per-chunk scores, so the metrics can be recomputed without
  re-running the annotator.

## Acceptance bar

A candidate annotator's F1 (cross-validated, if it tunes an aggregator and threshold) must be
within **0.05** of the `LLMAnnotator` (`openai:gpt-4o`) baseline F1 on the same data.
