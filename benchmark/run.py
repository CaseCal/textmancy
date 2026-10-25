"""
Runs an annotator over the reviewed gold datasets and records accuracy, latency, and cost.

Usage: python -m benchmark.run --annotator llm --model openai:gpt-4o
"""

import argparse
import asyncio
import json
import re
import statistics
import sys
import time
from datetime import date
from pathlib import Path
from typing import Callable, Mapping, Sequence

import dotenv
from pydantic import BaseModel

from textmancy.components import Annotator, LLMAnnotator
from textmancy.utils import chunk_input, gather_bounded

from .dataset import REPO_ROOT, GoldDataset, gold_digest, load_dataset, select_dataset_paths
from .metrics import Aggregator, best_config, cross_validate, predict, score, sweep
from .usage import UsageTracker, UsageTrackingModel, estimate_cost

RESULTS_DIR = REPO_ROOT / "benchmark" / "results"
RESULTS_MD = REPO_ROOT / "benchmark" / "RESULTS.md"
BASELINE = ("llm", "openai:gpt-4o")
ACCEPTANCE_DELTA = 0.05
FOLDS = 2

# Score-returning annotators are swept over these. Issue #8 swaps in the library's built-ins.
DEFAULT_AGGREGATORS: dict[str, Aggregator] = {"max": max, "mean": statistics.fmean}

AnnotatorFactory = Callable[[list[BaseModel], str, UsageTracker], Annotator]
ANNOTATORS: dict[str, AnnotatorFactory] = {
    "llm": lambda targets, model, tracker: LLMAnnotator(
        targets, model=UsageTrackingModel(model, tracker)
    ),
}


class UnreviewedPagesError(RuntimeError):
    """Raised when gold labels are incomplete or still need human review."""


def check_reviewed(datasets: Sequence[GoldDataset]) -> None:
    """Raises unless every page is labeled and every disagreement page is reviewed."""
    problems = []
    for dataset in datasets:
        texts = dataset.page_texts()
        labeled = sorted(page.index for page in dataset.pages)
        if labeled != sorted(texts):
            problems.append(f"{dataset.name}: pages are unlabeled; run benchmark/label.py")
            continue
        stale = dataset.stale_pages(texts)
        if stale:
            problems.append(
                f"{dataset.name}: pages {stale} have a missing or outdated text_hash; if the "
                "source or segmentor changed, delete those pages and re-label them"
            )
            continue
        unreviewed = [page.index for page in dataset.unreviewed_disagreements()]
        if unreviewed:
            problems.append(f"{dataset.name}: unreviewed disagreement pages {unreviewed}")
    if problems:
        raise UnreviewedPagesError("Gold labels are not ready:\n" + "\n".join(problems))


async def run_benchmark_async(
    datasets: Sequence[GoldDataset],
    annotators: Mapping[str, Annotator],
    max_concurrency: int = 10,
    chunk_size: int = 4000,
    aggregators: Mapping[str, Aggregator] = DEFAULT_AGGREGATORS,
) -> dict:
    """
    Annotates every gold page and scores the predictions.

    Args:
        datasets: Reviewed gold datasets.
        annotators: Dataset name to annotator; each dataset has its own target list.
        max_concurrency: The maximum number of pages annotated concurrently.
        chunk_size: The chunk size in characters.
        aggregators: Aggregators swept for annotators that expose annotate_scores_async.

    Returns:
        Metrics (overall and per dataset), latency, and per-page predictions. For
        score-returning annotators, the metrics and predictions are cross-validated, and the
        result also has the best config on all pages, the per-fold configs, the sweep curves,
        and the per-chunk scores.
    """
    scoring = all(hasattr(a, "annotate_scores_async") for a in annotators.values())

    async def _page(key, text):
        annotator = annotators[key[0]]
        start = time.perf_counter()
        if scoring:
            # Score each chunk separately so every aggregator can be applied afterwards.
            chunk_scores = [
                await annotator.annotate_scores_async(chunk, chunk_size)
                for chunk in chunk_input(text, chunk_size)
            ]
            output = {
                i: [scores.get(i, 0.0) for scores in chunk_scores]
                for i in range(len(annotator.targets))
            }
        else:
            output = await annotator.annotate_async(text, chunk_size)
        return key, output, time.perf_counter() - start

    gold, pages = {}, []
    for dataset in datasets:
        labels = {page.index: set(page.labels) for page in dataset.pages}
        for index, text in dataset.page_texts().items():
            gold[(dataset.name, index)] = labels[index]
            pages.append(((dataset.name, index), text))

    start = time.perf_counter()
    outputs = await gather_bounded((_page(key, text) for key, text in pages), max_concurrency)
    total = time.perf_counter() - start
    page_seconds = [seconds for _, _, seconds in outputs]

    names = [dataset.name for dataset in datasets]
    curves, config, cross_validation, saved_scores = None, None, None, None
    if scoring:
        scores = {key: output for key, output, _ in outputs}
        curves = sweep(gold, scores, aggregators)
        aggregator, threshold = best_config(curves)
        config = {"aggregator": aggregator, "threshold": threshold}
        # Report metrics on pages the config wasn't tuned on. Alternating pages keeps every
        # dataset in every fold.
        folds = [sorted(scores)[i::FOLDS] for i in range(FOLDS)]
        predicted, fold_configs = cross_validate(gold, scores, aggregators, folds)
        cross_validation = {
            "folds": FOLDS,
            "configs": fold_configs,
            "in_sample_f1": score(gold, predict(scores, aggregators[aggregator], threshold)).f1,
        }
        saved_scores = {
            name: {
                str(k[1]): {str(i): ps for i, ps in s.items()}
                for k, s in scores.items()
                if k[0] == name
            }
            for name in names
        }
    else:
        predicted = {key: output for key, output, _ in outputs}

    return {
        "metrics": {
            "overall": score(gold, predicted).model_dump(),
            "per_dataset": {
                name: score({k: v for k, v in gold.items() if k[0] == name}, predicted).model_dump()
                for name in names
            },
        },
        "best_config": config,
        "cross_validation": cross_validation,
        "sweep": curves,
        "latency": {
            "total_seconds": total,
            "mean_page_seconds": statistics.fmean(page_seconds) if page_seconds else 0.0,
            "pages": len(page_seconds),
        },
        "predictions": {
            name: {str(k[1]): sorted(v) for k, v in predicted.items() if k[0] == name}
            for name in names
        },
        "scores": saved_scores,
    }


def result_path(annotator: str, model: str, day: date, digest: str) -> Path:
    slug = re.sub(r"[^A-Za-z0-9._-]", "-", model)
    return RESULTS_DIR / f"{day.isoformat()}-{annotator}-{slug}-{digest}.json"


def load_results() -> list[dict]:
    return [json.loads(p.read_text(encoding="utf-8")) for p in sorted(RESULTS_DIR.glob("*.json"))]


def _config(result: dict) -> str:
    config = result.get("best_config")
    return f"{config['aggregator']} @ {config['threshold']:.2f}" if config else "—"


def render_results_md(results: list[dict]) -> str:
    """
    Renders the summary tables, with each run's F1 compared to the latest baseline scored on the
    same gold data.
    """
    results = sorted(results, key=lambda r: (r["date"], r["annotator"], r["model"]))
    baseline_f1 = {
        r["gold_digest"]: r["metrics"]["overall"]["f1"]
        for r in results
        if (r["annotator"], r["model"]) == BASELINE and "gold_digest" in r
    }
    lines = [
        "# Annotator Benchmark Results",
        "",
        "Generated by `python -m benchmark.run`; do not edit by hand.",
        "",
        f"**Acceptance bar:** a candidate annotator's F1 must be within **{ACCEPTANCE_DELTA}** "
        f"of the `{BASELINE[0]}` (`{BASELINE[1]}`) baseline F1 on the same data. ΔF1 compares "
        "against the latest baseline with the same gold digest (dataset selection, page text, "
        "and labels); — means there is none, so re-run the baseline.",
        "",
        "For annotators that tune an aggregator and threshold, P, R, and F1 are "
        f"{FOLDS}-fold cross-validated: each fold of pages is scored with the config chosen on "
        "the others. Config is the one chosen on all pages; its in-sample F1 is optimistic and "
        "is recorded in the results JSON only.",
        "",
        "| Date | Annotator | Model | Config | P | R | F1 | ΔF1 vs baseline | Total s | s/page "
        "| Tokens in/out | Cost |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in results:
        m, lat, usage = r["metrics"]["overall"], r["latency"], r["usage"]
        base = baseline_f1.get(r.get("gold_digest"))
        delta = "—" if base is None else f"{m['f1'] - base:+.3f}"
        cost = "—" if r.get("cost_usd") is None else f"${r['cost_usd']:.4f}"
        lines.append(
            f"| {r['date']} | {r['annotator']} | {r['model']} | {_config(r)} "
            f"| {m['precision']:.3f} | {m['recall']:.3f} | {m['f1']:.3f} | {delta} "
            f"| {lat['total_seconds']:.1f} | {lat['mean_page_seconds']:.2f} "
            f"| {usage['input_tokens']} / {usage['output_tokens']} | {cost} |"
        )
    names = sorted({name for r in results for name in r["metrics"]["per_dataset"]})
    lines += [
        "",
        "## F1 by dataset",
        "",
        "| Date | Annotator | Model | " + " | ".join(names) + " |",
        "|---|---|---|" + "---|" * len(names),
    ]
    for r in results:
        per = r["metrics"]["per_dataset"]
        cells = " | ".join(f"{per[n]['f1']:.3f}" if n in per else "—" for n in names)
        lines.append(f"| {r['date']} | {r['annotator']} | {r['model']} | {cells} |")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("--annotator", choices=sorted(ANNOTATORS), required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--datasets", nargs="*", help="Dataset names (default: all)")
    parser.add_argument("--max-concurrency", type=int, default=10)
    parser.add_argument("--chunk-size", type=int, default=4000)
    parser.add_argument("--price-in", type=float, help="USD per million input tokens")
    parser.add_argument("--price-out", type=float, help="USD per million output tokens")
    args = parser.parse_args()
    dotenv.load_dotenv()

    try:
        datasets = [load_dataset(path) for path in select_dataset_paths(args.datasets or [])]
        check_reviewed(datasets)
    except (ValueError, UnreviewedPagesError) as error:
        sys.exit(str(error))

    tracker = UsageTracker()
    annotators = {
        d.name: ANNOTATORS[args.annotator](d.target_objects(), args.model, tracker)
        for d in datasets
    }
    output = asyncio.run(
        run_benchmark_async(datasets, annotators, args.max_concurrency, args.chunk_size)
    )
    today = date.today()
    digest = gold_digest(datasets)
    result = {
        "date": today.isoformat(),
        "annotator": args.annotator,
        "model": args.model,
        "datasets": [d.name for d in datasets],
        "gold_digest": digest,
        "usage": tracker.as_dict(),
        "cost_usd": estimate_cost(args.model, tracker, args.price_in, args.price_out),
        **output,
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = result_path(args.annotator, args.model, today, digest)
    path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    RESULTS_MD.write_text(render_results_md(load_results()), encoding="utf-8")
    overall = output["metrics"]["overall"]
    print(
        f"F1 {overall['f1']:.3f} (P {overall['precision']:.3f}, R {overall['recall']:.3f}) "
        f"-> {path}"
    )


if __name__ == "__main__":
    main()
