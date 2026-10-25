import json

import pytest

import benchmark.dataset as dataset_module
from benchmark.dataset import (
    GoldDataset,
    GoldPage,
    gold_digest,
    load_dataset,
    page_hash,
    save_dataset,
    select_dataset_paths,
    string_match_labels,
)
from textmancy.targets import Character


def make_character(name, known_as=()):
    return Character(
        name=name,
        description="d",
        physical_description="p",
        known_as=list(known_as),
        summary_of_actions="s",
    )


def make_dataset(tmp_path, sample=None, pages=()):
    source = tmp_path / "book.txt"
    source.write_text("\n".join(f"para {i}" for i in range(25)), encoding="utf-8")
    return GoldDataset(
        name="book",
        source=str(source),
        segmentor={"type": "PageSegmentor", "paragraphs_per_page": 10},
        sample=sample,
        target_class="Character",
        targets=[make_character("Ann").model_dump()],
        pages=list(pages),
    )


def test_page_texts_regenerates_all_pages(tmp_path):
    texts = make_dataset(tmp_path).page_texts()
    assert list(texts) == [0, 1, 2]
    assert texts[2] == "\n".join(f"para {i}" for i in range(20, 25))


def test_sample_is_seeded_sorted_and_stable(tmp_path):
    dataset = make_dataset(tmp_path, sample={"size": 2, "seed": 3})
    indices = dataset.page_indices()
    assert len(indices) == 2
    assert indices == sorted(indices)
    assert indices == make_dataset(tmp_path, sample={"size": 2, "seed": 3}).page_indices()
    assert list(dataset.page_texts()) == indices


def test_unknown_segmentor_raises(tmp_path):
    dataset = make_dataset(tmp_path)
    dataset.segmentor = {"type": "Nope"}
    with pytest.raises(ValueError, match="Nope"):
        dataset.page_texts()


def test_target_objects_builds_target_class(tmp_path):
    targets = make_dataset(tmp_path).target_objects()
    assert targets == [make_character("Ann")]


def test_string_match_is_case_insensitive_word_bounded_and_uses_aliases():
    targets = [
        make_character("Ann"),
        make_character("Robert Smith", ["Bob"]),
        make_character("Tim"),
    ]
    text = "ANN met bob. Timothy was absent."
    assert string_match_labels(targets, text) == {0, 1}


def test_unreviewed_disagreements(tmp_path):
    pages = [
        GoldPage(index=0, labels=[0], silver=[0], string_match=[0]),
        GoldPage(index=1, labels=[0], silver=[0], string_match=[], reviewed=True),
        GoldPage(index=2, labels=[], silver=[], string_match=[0]),
    ]
    dataset = make_dataset(tmp_path, pages=pages)
    assert [p.index for p in dataset.unreviewed_disagreements()] == [2]


def test_save_and_load_round_trip(tmp_path):
    dataset = make_dataset(
        tmp_path, pages=[GoldPage(index=0, labels=[0], silver=[0], string_match=[])]
    )
    path = tmp_path / "out.json"
    save_dataset(dataset, path)
    assert json.loads(path.read_text(encoding="utf-8"))["pages"][0]["labels"] == [0]
    assert load_dataset(path) == dataset


def test_label_indices_must_be_valid_targets(tmp_path):
    with pytest.raises(ValueError, match=r"page 0.*\[1\]"):
        make_dataset(tmp_path, pages=[GoldPage(index=0, labels=[1], silver=[0], string_match=[])])


def test_select_dataset_paths_defaults_to_all_and_filters_by_name(tmp_path, monkeypatch):
    monkeypatch.setattr(dataset_module, "DATA_DIR", tmp_path)
    for name in ("alpha", "beta"):
        (tmp_path / f"{name}.json").write_text("{}", encoding="utf-8")

    assert [p.stem for p in select_dataset_paths([])] == ["alpha", "beta"]
    assert [p.stem for p in select_dataset_paths(["beta"])] == ["beta"]


def test_select_dataset_paths_rejects_unknown_names(tmp_path, monkeypatch):
    monkeypatch.setattr(dataset_module, "DATA_DIR", tmp_path)
    (tmp_path / "alpha.json").write_text("{}", encoding="utf-8")

    with pytest.raises(ValueError, match=r"alhpa.*alpha"):
        select_dataset_paths(["alhpa"])


def test_select_dataset_paths_rejects_empty_data_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(dataset_module, "DATA_DIR", tmp_path)

    with pytest.raises(ValueError, match="No datasets"):
        select_dataset_paths([])


def test_stale_pages_flags_missing_and_mismatched_text_hashes(tmp_path):
    dataset = make_dataset(tmp_path)
    texts = dataset.page_texts()
    dataset.pages = [
        GoldPage(index=0, text_hash=page_hash(texts[0]), labels=[], silver=[], string_match=[]),
        GoldPage(index=1, text_hash=page_hash("old text"), labels=[], silver=[], string_match=[]),
        GoldPage(index=2, labels=[], silver=[], string_match=[]),
    ]

    assert dataset.stale_pages(texts) == [1, 2]


def test_gold_digest_tracks_labels_and_text(tmp_path):
    def digest(labels, text_hash="h"):
        page = GoldPage(index=0, text_hash=text_hash, labels=labels, silver=[], string_match=[])
        return gold_digest([make_dataset(tmp_path, pages=[page])])

    assert digest([0]) == digest([0])
    assert digest([0]) != digest([])
    assert digest([0]) != digest([0], text_hash="other")
