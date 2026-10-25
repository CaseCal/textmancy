"""Gold dataset files: frozen targets plus per-page labels, with page text regenerated."""

import hashlib
import json
import random
import re
from pathlib import Path
from typing import Mapping, Optional, Sequence

from pydantic import BaseModel, model_validator

from textmancy import targets as target_module
from textmancy.components import segmentor as segmentor_module

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "benchmark" / "data"


def page_hash(text: str) -> str:
    """A short fingerprint of a page's text, so labels can't silently drift onto other text."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


class GoldPage(BaseModel):
    index: int
    text_hash: Optional[str] = None
    labels: list[int]
    silver: list[int]
    string_match: list[int]
    reviewed: bool = False

    @property
    def disagrees(self) -> bool:
        return set(self.silver) != set(self.string_match)


class SampleSpec(BaseModel):
    size: int
    seed: int


class GoldDataset(BaseModel):
    name: str
    source: str
    segmentor: dict
    sample: Optional[SampleSpec] = None
    target_class: str
    targets: list[dict]
    pages: list[GoldPage] = []

    @model_validator(mode="after")
    def _check_target_indices(self) -> "GoldDataset":
        valid = range(len(self.targets))
        for page in self.pages:
            indices = {*page.labels, *page.silver, *page.string_match}
            invalid = sorted(i for i in indices if i not in valid)
            if invalid:
                raise ValueError(
                    f"{self.name}: page {page.index} has target indices {invalid}, but only "
                    f"{len(self.targets)} targets exist"
                )
        return self

    def target_objects(self) -> list[BaseModel]:
        target_class = getattr(target_module, self.target_class)
        return [target_class(**t) for t in self.targets]

    def _all_pages(self) -> list[str]:
        config = dict(self.segmentor)
        segmentor_type = config.pop("type")
        segmentor_class = getattr(segmentor_module, segmentor_type, None)
        if segmentor_class is None:
            raise ValueError(f"Unknown segmentor type: {segmentor_type}")
        source = Path(self.source)
        if not source.is_absolute():
            source = REPO_ROOT / source
        return segmentor_class(**config).segment(source.read_text(encoding="utf-8"))

    def page_indices(self) -> list[int]:
        return self._indices(len(self._all_pages()))

    def _indices(self, page_count: int) -> list[int]:
        if self.sample is None:
            return list(range(page_count))
        rng = random.Random(self.sample.seed)
        return sorted(rng.sample(range(page_count), self.sample.size))

    def page_texts(self) -> dict[int, str]:
        pages = self._all_pages()
        return {i: pages[i] for i in self._indices(len(pages))}

    def unreviewed_disagreements(self) -> list[GoldPage]:
        return [p for p in self.pages if p.disagrees and not p.reviewed]

    def stale_pages(self, texts: Mapping[int, str]) -> list[int]:
        """Indices of labeled pages whose text_hash is missing or doesn't match their text."""
        return [
            p.index
            for p in self.pages
            if p.index in texts and p.text_hash != page_hash(texts[p.index])
        ]


def gold_digest(datasets: Sequence[GoldDataset]) -> str:
    """A fingerprint of the gold data a run is scored on: dataset names, page texts, labels."""
    gold = {d.name: [[p.index, p.text_hash, sorted(p.labels)] for p in d.pages] for d in datasets}
    return hashlib.sha256(json.dumps(gold, sort_keys=True).encode("utf-8")).hexdigest()[:12]


def load_dataset(path: Path) -> GoldDataset:
    return GoldDataset.model_validate_json(Path(path).read_text(encoding="utf-8"))


def save_dataset(dataset: GoldDataset, path: Path) -> None:
    text = json.dumps(dataset.model_dump(mode="json"), indent=2, ensure_ascii=False)
    Path(path).write_text(text + "\n", encoding="utf-8")


def dataset_paths() -> list[Path]:
    return sorted(DATA_DIR.glob("*.json"))


def select_dataset_paths(names: Sequence[str]) -> list[Path]:
    """Dataset files whose stem is in names, or every dataset file when names is empty."""
    paths = dataset_paths()
    available = [path.stem for path in paths]
    unknown = sorted(set(names) - set(available))
    if unknown:
        raise ValueError(f"Unknown datasets {unknown}; available: {available}")
    if not paths:
        raise ValueError(f"No datasets in {DATA_DIR}")
    return [path for path in paths if not names or path.stem in names]


def string_match_labels(targets: Sequence[BaseModel], text: str) -> set[int]:
    """Indices of targets whose name or any known_as alias appears as a whole word."""
    found = set()
    for i, target in enumerate(targets):
        aliases = [target.name, *getattr(target, "known_as", [])]
        for alias in aliases:
            if alias and re.search(rf"\b{re.escape(alias)}\b", text, re.IGNORECASE):
                found.add(i)
                break
    return found
