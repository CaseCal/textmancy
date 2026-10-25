"""
Silver-labels gold datasets with a strong LLM and writes a review report of every page where
the LLM disagrees with an alias string match.

Usage: python -m benchmark.label [dataset names...] [--model MODEL] [--max-concurrency N]
"""

import argparse
import asyncio
import sys
from typing import Union

import dotenv

from textmancy.components import Annotator, LLMAnnotator
from textmancy.utils import gather_bounded

from .dataset import (
    REPO_ROOT,
    GoldDataset,
    GoldPage,
    load_dataset,
    page_hash,
    save_dataset,
    select_dataset_paths,
    string_match_labels,
)
from .usage import UsageTracker, UsageTrackingModel

REVIEW_DIR = REPO_ROOT / "benchmark" / "review"
DEFAULT_MODEL = "anthropic:claude-opus-5-5"


async def label_dataset_async(
    dataset: GoldDataset, annotator: Annotator, max_concurrency: int = 10
) -> tuple[GoldDataset, dict[int, Exception]]:
    """
    Labels every page not already in the dataset; existing pages are left untouched.

    A page that fails is left unlabeled rather than discarding the other pages' paid labels, so
    re-running retries only the failures.

    Returns:
        The dataset with the newly labeled pages, and the exception for each page that failed.
    """
    targets = dataset.target_objects()
    done = {page.index for page in dataset.pages}
    todo = {i: text for i, text in dataset.page_texts().items() if i not in done}

    async def _label(index: int, text: str) -> Union[GoldPage, Exception]:
        try:
            silver = sorted(await annotator.annotate_async(text))
        except Exception as error:
            return error
        return GoldPage(
            index=index,
            text_hash=page_hash(text),
            labels=silver,
            silver=silver,
            string_match=sorted(string_match_labels(targets, text)),
        )

    results = await gather_bounded((_label(i, t) for i, t in todo.items()), max_concurrency)
    failures = {i: r for i, r in zip(todo, results) if isinstance(r, Exception)}
    new_pages = [r for r in results if isinstance(r, GoldPage)]
    pages = sorted([*dataset.pages, *new_pages], key=lambda page: page.index)
    return dataset.model_copy(update={"pages": pages}), failures


def _names(indices, targets) -> str:
    return ", ".join(f"[{i}] {targets[i].name}" for i in sorted(indices)) or "(none)"


def render_review_report(dataset: GoldDataset) -> str:
    """Renders every unreviewed disagreement page as Markdown for human review."""
    targets = dataset.target_objects()
    pages = dataset.unreviewed_disagreements()
    texts = dataset.page_texts() if pages else {}
    lines = [
        f"# Review: {dataset.name}",
        "",
        f"{len(pages)} page(s) need review. Fix `labels` in the dataset JSON, then set "
        "`reviewed: true` on each page.",
        "",
        "Targets:",
        "",
        *(f"- [{i}] {t.name} (known as: {', '.join(t.known_as)})" for i, t in enumerate(targets)),
        "",
    ]
    for page in pages:
        silver, matched = set(page.silver), set(page.string_match)
        lines += [
            f"## Page {page.index}",
            "",
            "```text",
            texts[page.index],
            "```",
            "",
            f"- Silver: {_names(silver, targets)}",
            f"- String match: {_names(matched, targets)}",
            f"- Only silver: {_names(silver - matched, targets)}",
            f"- Only string match: {_names(matched - silver, targets)}",
            "",
        ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("datasets", nargs="*", help="Dataset names (default: all)")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--max-concurrency", type=int, default=10)
    args = parser.parse_args()
    dotenv.load_dotenv()

    try:
        paths = select_dataset_paths(args.datasets)
    except ValueError as error:
        sys.exit(str(error))

    REVIEW_DIR.mkdir(parents=True, exist_ok=True)
    failed = False
    for path in paths:
        dataset = load_dataset(path)
        tracker = UsageTracker()
        annotator = LLMAnnotator(
            dataset.target_objects(), model=UsageTrackingModel(args.model, tracker)
        )
        dataset, failures = asyncio.run(
            label_dataset_async(dataset, annotator, args.max_concurrency)
        )
        save_dataset(dataset, path)
        report = REVIEW_DIR / f"{dataset.name}.md"
        report.write_text(render_review_report(dataset), encoding="utf-8")
        print(
            f"{dataset.name}: {len(dataset.pages)} pages, "
            f"{len(dataset.unreviewed_disagreements())} need review -> {report}; "
            f"usage {tracker.as_dict()}"
        )
        for index, error in sorted(failures.items()):
            failed = True
            print(f"{dataset.name}: page {index} failed ({error!r}); re-run to retry it")
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
