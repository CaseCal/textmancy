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

from .dataset import REPO_ROOT, GoldDataset, dataset_paths, load_dataset
from .metrics import Aggregator, best_config, score, sweep
from .usage import UsageTracker, UsageTrackingModel, estimate_cost

RESULTS_DIR = REPO_ROOT / "benchmark" / "results"
RESULTS_MD = REPO_ROOT / "benchmark" / "RESULTS.md"
BASELINE = ("llm", "openai:gpt-4o")
ACCEPTANCE_DELTA = 0.05

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
        labeled = sorted(page.index for page in dataset.pages)
        if labeled != dataset.page_indices():
            problems.append(f"{dataset.name}: pages are unlabeled; run benchmark/label.py")
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
        Metrics (overall and per dataset), best config and sweep curves (score-returning
        annotators only), latency, and per-page predictions.
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

    curves, config = None, None
    if scoring:
        scores = {key: output for key, output, _ in outputs}
        curves = sweep(gold, scores, aggregators)
        name, threshold = best_config(curves)
        config = {"aggregator": name, "threshold": threshold}
        aggregate = aggregators[name]
        predicted = {
            key: {i for i, ps in s.items() if aggregate(ps) >= threshold}
            for key, s in scores.items()
        }
    else:
        predicted = {key: output for key, output, _ in outputs}

    names = [dataset.name for dataset in datasets]
    return {
        "metrics": {
            "overall": score(gold, predicted).model_dump(),
            "per_dataset": {
                name: score({k: v for k, v in gold.items() if k[0] == name}, predicted).model_dump()
                for name in names
            },
        },
        "best_config": config,
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
    }


def result_path(annotator: str, model: str, day: date) -> Path:
    slug = re.sub(r"[^A-Za-z0-9._-]", "-", model)
    return RESULTS_DIR / f"{day.isoformat()}-{annotator}-{slug}.json"


def load_results() -> list[dict]:
    return [json.loads(p.read_text(encoding="utf-8")) for p in sorted(RESULTS_DIR.glob("*.json"))]


def _config(result: dict) -> str:
    config = result.get("best_config")
    return f"{config['aggregator']} @ {config['threshold']:.2f}" if config else "—"


def render_results_md(results: list[dict]) -> str:
    """Renders the summary tables, with each run's F1 compared to the latest baseline."""
    results = sorted(results, key=lambda r: (r["date"], r["annotator"], r["model"]))
    baselines = [r for r in results if (r["annotator"], r["model"]) == BASELINE]
    baseline_f1 = baselines[-1]["metrics"]["overall"]["f1"] if baselines else None
    lines = [
        "# Annotator Benchmark Results",
        "",
        "Generated by `python -m benchmark.run`; do not edit by hand.",
        "",
        "**Acceptance bar:** a candidate annotator's best-configuration F1 must be within "
        f"**{ACCEPTANCE_DELTA}** of the `{BASELINE[0]}` (`{BASELINE[1]}`) baseline F1 on the "
        "same data.",
        "",
        "| Date | Annotator | Model | Config | P | R | F1 | ΔF1 vs baseline | Total s | s/page "
        "| Tokens in/out | Cost |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in results:
        m, lat, usage = r["metrics"]["overall"], r["latency"], r["usage"]
        delta = "—" if baseline_f1 is None else f"{m['f1'] - baseline_f1:+.3f}"
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

    datasets = [
        load_dataset(p) for p in dataset_paths() if not args.datasets or p.stem in args.datasets
    ]
    try:
        check_reviewed(datasets)
    except UnreviewedPagesError as error:
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
    result = {
        "date": today.isoformat(),
        "annotator": args.annotator,
        "model": args.model,
        "datasets": [d.name for d in datasets],
        "usage": tracker.as_dict(),
        "cost_usd": estimate_cost(args.model, tracker, args.price_in, args.price_out),
        **output,
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = result_path(args.annotator, args.model, today)
    path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    RESULTS_MD.write_text(render_results_md(load_results()), encoding="utf-8")
    overall = output["metrics"]["overall"]
    print(
        f"F1 {overall['f1']:.3f} (P {overall['precision']:.3f}, R {overall['recall']:.3f}) "
        f"-> {path}"
    )


if __name__ == "__main__":
    main()
