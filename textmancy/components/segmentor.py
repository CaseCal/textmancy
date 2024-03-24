from abc import ABC, abstractmethod
from typing import Optional


class Segmentor(ABC):
    """
    Class for segmenting text into various organizational units, such as chapters, paragraphs,
    sentences, scenes, etc.
    """

    def __init__(self, max_length: Optional[int] = None, handle_length: str = "split") -> None:
        self._max_length = max_length
        allowed_handle_lengths = ["split", "truncate", "raise"]
        if handle_length not in allowed_handle_lengths:
            raise ValueError("handle_length must be one of {}".format(allowed_handle_lengths))
        self._handle_length = handle_length

    @abstractmethod
    def _segment(self, text: str) -> list[str]:
        """
        Segments the given text into organizational units. Override this for
        specific implementations
        """
        pass  # pragma: no cover

    def segment(self, text: str) -> list[str]:
        """
        Segments the given text into organizational units.
        """
        text_segments = self._segment(text)
        print(text_segments)
        return self._handle_max_length(text_segments)

    def _handle_max_length(self, text_segments: list[str]) -> list[str]:
        """
        Splits the text into segments of the maximum length.
        """
        if self._max_length is None:
            return text_segments

        segments = []
        for i, segment in enumerate(text_segments):
            if len(segment) > self._max_length:
                if self._handle_length == "split":
                    segments.extend(
                        (segment[0: self._max_length], segment[self._max_length:])
                    )
                elif self._handle_length == "truncate":
                    segments.append(segment[: self._max_length])
                elif self._handle_length == "raise":
                    raise ValueError(
                        f"Segment {i} length {len(segment)} "
                        f"exceeds maximum length {self._max_length}"
                    )
            else:
                segments.append(segment)

        return segments


class ParagraphSegmentor(Segmentor):
    """
    A class that segments text into paragraphs.
    """

    def _segment(self, text: str) -> list:
        """
        Segments the given text into paragraphs.
        """
        return [p for p in text.split("\n") if p]


class PageSegmentor(Segmentor):
    """
    A class that segments text into pages. Each page contains 10 paragraphs.
    """

    def __init__(self, paragraphs_per_page: int = 10, **kwargs) -> None:
        super().__init__(**kwargs)
        self._paragraphs_per_page = paragraphs_per_page

    def _segment(self, text: str) -> list:
        """
        Segments the given text into pages.
        """
        paragraphs = [p for p in text.split("\n") if p]
        return [
            "\n".join(paragraphs[i : i + self._paragraphs_per_page])
            for i in range(0, len(paragraphs), self._paragraphs_per_page)
        ]
