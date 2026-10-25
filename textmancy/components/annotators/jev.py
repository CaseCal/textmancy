from typing import Any, Optional, Sequence, Union

from pydantic import BaseModel

from ... import utils
from .aggregation import Aggregator, get_aggregator
from .base import Annotator


class JevAnnotator(Annotator):
    """
    Annotates text with TypeSafe's Jev classifier, asking one yes/no (Noul) question per target.

    Each chunk is sent as the request state; Jev returns a probability per target. Scores from
    several chunks are combined with the chosen aggregator, and `annotate` keeps the targets
    scoring at least the threshold. Requires the `jev` extra and `TYPESAFE_API_KEY`.

    Attributes:
        model (str): The Jev model name.
        threshold (float): The default probability needed for a target to count as present.
        aggregator (Aggregator): Combines a target's per-chunk probabilities into one score.
        max_questions_per_request (int): Targets beyond this are split across requests.
        questions (dict[str, Noul]): One question per target, keyed "target_{index}".
    """

    def __init__(
        self,
        targets: Sequence[BaseModel],
        model: str = "jev-latest",
        threshold: float = 0.5,
        aggregate: Union[str, Aggregator] = "max",
        min_prob: float = 0.2,
        max_questions_per_request: int = 100,
        max_concurrency: int = 10,
        client: Optional[Any] = None,
    ):
        """
        Args:
            targets: The targets to identify; results are indices into this list.
            model: The Jev model name.
            threshold: The default probability needed for a target to count as present.
            aggregate: "max" (default), "noisy_or", "mean", or a callable taking a list of
                per-chunk probabilities.
            min_prob: Probabilities below this are ignored by "noisy_or".
            max_questions_per_request: The most target questions sent in one request.
            max_concurrency: The maximum number of requests in flight.
            client: A TypeSafe client exposing an async `system_one`; by default an
                `AsyncTypeSafeClient` is created on first use.
        """
        super().__init__(targets, max_concurrency)
        try:
            import typesafe_sdk
        except ImportError as error:
            raise ImportError(
                "JevAnnotator requires the 'jev' extra: pip install textmancy[jev]"
            ) from error
        if max_questions_per_request < 1:
            raise ValueError("max_questions_per_request must be at least 1")

        self._sdk = typesafe_sdk
        self._client = client
        self.model = model
        self.threshold = threshold
        self.aggregator = get_aggregator(aggregate, min_prob)
        self.max_questions_per_request = max_questions_per_request
        self.questions = self._create_questions()

    @property
    def client(self) -> Any:
        if self._client is None:
            self._client = self._sdk.AsyncTypeSafeClient()
        return self._client

    def _create_questions(self) -> dict[str, Any]:
        Noul, NoulCriteria = self._sdk.Noul, self._sdk.NoulCriteria
        name = self.target_name
        return {
            f"target_{i}": Noul(
                instructions={
                    "question": f"Does the text include or refer to this {name}?",
                    name: target.model_dump(mode="json"),
                },
                criteria=NoulCriteria(
                    true=f"This {name} appears or is referenced, including by alias",
                    false=f"This {name} does not appear",
                ),
            )
            for i, target in enumerate(self.targets)
        }

    def _batches(self) -> list[dict[str, Any]]:
        items = list(self.questions.items())
        size = self.max_questions_per_request
        return [dict(items[i : i + size]) for i in range(0, len(items), size)]

    async def _request_async(self, chunk: str, batch: dict[str, Any]) -> dict[int, float]:
        response = await self.client.system_one(state=chunk, questions=batch, model=self.model)
        return {int(name.removeprefix("target_")): response.nouls[name].noul for name in batch}

    async def _chunk_scores_async(self, chunks: list[str]) -> list[dict[int, float]]:
        """Returns each chunk's probability for every target, batching questions per request."""
        requests = [(c, batch) for c in range(len(chunks)) for batch in self._batches()]
        answers = await utils.gather_bounded(
            (self._request_async(chunks[c], batch) for c, batch in requests),
            self.max_concurrency,
        )
        scores: list[dict[int, float]] = [{} for _ in chunks]
        for (c, _), answer in zip(requests, answers):
            scores[c].update(answer)
        return scores

    async def annotate_scores_async(self, text: str, chunk_size: int = 4000) -> dict[int, float]:
        """
        Scores how likely each target is to appear in the text.

        Args:
            text: The text to annotate.
            chunk_size: The chunk size in characters.

        Returns:
            Every target index mapped to its aggregated probability.
        """
        chunk_scores = await self._chunk_scores_async(utils.chunk_input(text, chunk_size))
        self._logger.debug(f"Scored {len(chunk_scores)} chunks")
        return {
            i: self.aggregator([scores[i] for scores in chunk_scores])
            for i in range(len(self.targets))
        }

    def annotate_scores(self, text: str, chunk_size: int = 4000) -> dict[int, float]:
        """Synchronous wrapper for annotate_scores_async."""
        return utils.run_sync(self.annotate_scores_async(text, chunk_size))

    async def annotate_async(
        self, text: str, chunk_size: int = 4000, threshold: Optional[float] = None
    ) -> set[int]:
        """
        Identifies which targets appear in the text.

        Args:
            text: The text to annotate.
            chunk_size: The chunk size in characters.
            threshold: Overrides the constructor threshold for this call.

        Returns:
            The indices of targets whose aggregated probability is at least the threshold.
        """
        threshold = self.threshold if threshold is None else threshold
        scores = await self.annotate_scores_async(text, chunk_size)
        return {i for i, p in scores.items() if p >= threshold}

    def annotate(
        self, text: str, chunk_size: int = 4000, threshold: Optional[float] = None
    ) -> set[int]:
        """Synchronous wrapper for annotate_async."""
        return utils.run_sync(self.annotate_async(text, chunk_size, threshold))

    async def _annotate_chunk_async(self, chunk: str) -> set[int]:
        (scores,) = await self._chunk_scores_async([chunk])
        return {i for i, p in scores.items() if p >= self.threshold}
