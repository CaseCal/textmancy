from benchmark.dataset import GoldDataset, GoldPage, page_hash
from benchmark.label import label_dataset_async, render_review_report
from textmancy.components import LLMAnnotator
from textmancy.targets import Character
from textmancy.utils import run_sync


def make_character(name, known_as=()):
    return Character(
        name=name,
        description="d",
        physical_description="p",
        known_as=list(known_as),
        summary_of_actions="s",
    )


def make_dataset(tmp_path, pages=()):
    source = tmp_path / "book.txt"
    source.write_text("Ann walks.\nBob runs.\nNobody here.", encoding="utf-8")
    return GoldDataset(
        name="book",
        source=str(source),
        segmentor={"type": "PageSegmentor", "paragraphs_per_page": 1},
        target_class="Character",
        targets=[make_character("Ann").model_dump(), make_character("Bob").model_dump()],
        pages=list(pages),
    )


def test_label_dataset_records_silver_and_string_match(tmp_path, recording_model):
    # Silver says page 0 → [0], page 1 → [] (missed Bob), page 2 → [0] (hallucinated Ann)
    answers = {"Ann walks.": [0], "Bob runs.": [], "Nobody here.": [0]}
    model = recording_model(lambda system, user, i: answers[user.removeprefix("Text:\n")])
    dataset = make_dataset(tmp_path)
    annotator = LLMAnnotator(dataset.target_objects(), model=model.model)

    labeled, failures = run_sync(label_dataset_async(dataset, annotator))

    assert [(p.index, p.silver, p.string_match, p.labels, p.reviewed) for p in labeled.pages] == [
        (0, [0], [0], [0], False),
        (1, [], [1], [], False),
        (2, [0], [], [0], False),
    ]
    assert [p.index for p in labeled.unreviewed_disagreements()] == [1, 2]
    assert labeled.pages[1].text_hash == page_hash("Bob runs.")
    assert failures == {}


def test_label_dataset_skips_existing_pages(tmp_path, recording_model):
    model = recording_model(lambda system, user, i: [])
    existing = GoldPage(index=0, labels=[1], silver=[0], string_match=[0], reviewed=True)
    dataset = make_dataset(tmp_path, pages=[existing])
    annotator = LLMAnnotator(dataset.target_objects(), model=model.model)

    labeled, failures = run_sync(label_dataset_async(dataset, annotator))

    assert len(model.calls) == 2
    assert labeled.pages[0] == existing


def test_label_dataset_keeps_pages_that_succeed_when_one_fails(tmp_path, recording_model):
    def respond(system, user, i):
        if "Bob" in user:
            raise RuntimeError("overloaded")
        return [0]

    model = recording_model(respond)
    dataset = make_dataset(tmp_path)
    annotator = LLMAnnotator(dataset.target_objects(), model=model.model)

    labeled, failures = run_sync(label_dataset_async(dataset, annotator))

    assert [p.index for p in labeled.pages] == [0, 2]
    assert list(failures) == [1]
    assert "overloaded" in str(failures[1])


def test_review_report_lists_only_unreviewed_disagreements(tmp_path):
    pages = [
        GoldPage(index=0, labels=[0], silver=[0], string_match=[0]),
        GoldPage(index=1, labels=[], silver=[], string_match=[1]),
        GoldPage(index=2, labels=[0], silver=[0], string_match=[], reviewed=True),
    ]
    report = render_review_report(make_dataset(tmp_path, pages=pages))
    assert "## Page 1" in report
    assert "Bob runs." in report
    assert "Only string match: [1] Bob" in report
    assert "## Page 0" not in report and "## Page 2" not in report
