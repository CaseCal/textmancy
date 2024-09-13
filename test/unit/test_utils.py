from textmancy.utils import text_generator


def test_text_generator():
    texts = ["Lorem", "ipsum", "dolor", "sit", "amet"]
    max_chunk_size = 5
    expected_output = ["Lorem", "ipsum", "dolor", "sit", "amet"]

    generator = text_generator(texts, max_chunk_size)
    output = list(generator)

    assert output == expected_output


def test_text_generator_with_large_text():
    texts = ["Lorem ipsum dolor sit amet, consectetur adipiscing elit."]
    max_chunk_size = 10
    expected_output = [
        "Lorem ipsu",
        "m dolor si",
        "t amet, co",
        "nsectetur ",
        "adipiscing",
        " elit.",
    ]

    generator = text_generator(texts, max_chunk_size)
    output = list(generator)

    assert output == expected_output


def test_text_generator_with_empty_input():
    texts = []
    max_chunk_size = 5
    expected_output = []

    generator = text_generator(texts, max_chunk_size)
    output = list(generator)

    assert output == expected_output
