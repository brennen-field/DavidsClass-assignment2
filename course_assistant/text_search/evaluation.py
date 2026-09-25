"""Repeatable comparison utilities for text-retrieval approaches."""

from __future__ import annotations

import csv
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol

from course_assistant.text_search.models import SearchResult


class Retriever(Protocol):
    def search(self, query: str, *, top_k: int = 5) -> list[SearchResult]: ...


@dataclass(frozen=True)
class EvaluationQuestion:
    question_id: str
    query: str
    expected_doc_id: str
    expected_page_number: int | None = None


@dataclass(frozen=True)
class RetrievalMeasurement:
    method: str
    question_id: str
    elapsed_seconds: float
    correct_source_in_top_k: bool
    top_doc_id: str
    top_page_number: int | None


def compare_retrievers(
    retrievers: Mapping[str, Retriever],
    questions: Sequence[EvaluationQuestion],
    *,
    top_k: int = 5,
    clock: Callable[[], float] = time.perf_counter,
) -> list[RetrievalMeasurement]:
    """Run the same questions through every retriever and record source recall."""

    if top_k <= 0:
        raise ValueError("top_k must be greater than zero")
    measurements: list[RetrievalMeasurement] = []
    for method, retriever in retrievers.items():
        for question in questions:
            started = clock()
            results = retriever.search(question.query, top_k=top_k)
            elapsed = clock() - started
            matched = any(
                result.doc_id == question.expected_doc_id
                and (
                    question.expected_page_number is None
                    or result.page_number == question.expected_page_number
                )
                for result in results
            )
            top = results[0] if results else None
            measurements.append(
                RetrievalMeasurement(
                    method=method,
                    question_id=question.question_id,
                    elapsed_seconds=elapsed,
                    correct_source_in_top_k=matched,
                    top_doc_id=top.doc_id if top else "",
                    top_page_number=top.page_number if top else None,
                )
            )
    return measurements


def write_comparison_csv(
    path: str | Path,
    measurements: Sequence[RetrievalMeasurement],
) -> None:
    """Save raw comparison results for the final README analysis."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(RetrievalMeasurement.__dataclass_fields__)
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(asdict(measurement) for measurement in measurements)
