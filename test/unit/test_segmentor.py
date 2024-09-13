from textmancy.components.segmentor import Segmentor, ParagraphSegmentor, PageSegmentor

import pytest


class SampleSegmentor(Segmentor):
    def _segment(self, text: str) -> list:
        return [p for p in text.split(" ") if p]


@pytest.mark.parametrize("max_length, handle_length", [
    (10, "split"),
    (20, "truncate"),
    (30, "raise")
])
def test_segmentor_init(max_length, handle_length):
    segmentor = SampleSegmentor(max_length=max_length, handle_length=handle_length)
    assert segmentor._max_length == max_length
    assert segmentor._handle_length == handle_length


def test_segmentor_init_validation():
    with pytest.raises(ValueError):
        SampleSegmentor(max_length=10, handle_length="invalid")


def test_segmentor_handle_max_length():
    text = "This is a test sentence"

    segmentor = SampleSegmentor(max_length=5, handle_length="split")
    expected_output = ["This", "is", "a", "test", "sente", "nce"]
    assert segmentor.segment(text) == expected_output

    segmentor = SampleSegmentor(max_length=5, handle_length="truncate")
    expected_output = ["This", "is", "a", "test", "sente"]
    assert segmentor.segment(text) == expected_output

    segmentor = SampleSegmentor(max_length=5, handle_length="raise")
    with pytest.raises(ValueError):
        segmentor.segment(text)


def test_paragraph_segmentor_empty_text():
    segmentor = ParagraphSegmentor()
    text = ""
    expected_output = []
    assert segmentor.segment(text) == expected_output


def test_paragraph_segmentor_single_paragraph():
    segmentor = ParagraphSegmentor()
    text = "This is a single paragraph."
    expected_output = ["This is a single paragraph."]
    assert segmentor.segment(text) == expected_output


def test_paragraph_segmentor_multiple_paragraphs():
    segmentor = ParagraphSegmentor()
    text = "This is the first paragraph.\n\nThis is the second paragraph."
    expected_output = ["This is the first paragraph.", "This is the second paragraph."]
    assert segmentor.segment(text) == expected_output


def test_page_segmentor_empty_text():
    segmentor = PageSegmentor()
    text = ""
    expected_output = []
    assert segmentor.segment(text) == expected_output


def test_page_segmentor_single_page():
    segmentor = PageSegmentor()
    text = "This is a single page."
    expected_output = ["This is a single page."]
    assert segmentor.segment(text) == expected_output


def test_page_segmentor_multiple_pages():
    segmentor = PageSegmentor(paragraphs_per_page=2)
    text = (
            "This is the first para.\nThis is the second para.\n"
            "This is the third para.\nThis is the fourth para.\n"
    )
    expected_output = [
        "This is the first para.\nThis is the second para.",
        "This is the third para.\nThis is the fourth para."
    ]
    assert segmentor.segment(text) == expected_output
