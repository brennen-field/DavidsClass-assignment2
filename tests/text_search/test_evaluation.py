import pytest

from course_assistant.text_search.evaluation import (
    EvaluationQuestion,
    compare_retrievers,
    write_comparison_csv,
)
from course_assistant.text_search.models import SearchResult


class FixedRetriever:
    def __init__(self, doc_id, page_number):
        self.result = SearchResult(
            chunk_id=f"{doc_id}-{page_number}",
            doc_id=doc_id,
            doc_name=f"{doc_id}.pdf",
            page_number=page_number,
            text="Supporting source text",
            image_path=f"/{doc_id}-{page_number}.png",
            source_format="pdf",
            score=1.0,
            search_method="test",
        )

    def search(self, query, *, top_k=5):
        return [self.result][:top_k]


def test_comparison_uses_the_same_questions_and_records_source_recall(tmp_path):
    ticks = iter([1.0, 1.2, 2.0, 2.3])
    measurements = compare_retrievers(
        {
            "keyword": FixedRetriever("syllabus", 2),
            "semantic": FixedRetriever("week-5", 15),
        },
        [
            EvaluationQuestion(
                question_id="grading",
                query="How is grading determined?",
                expected_doc_id="syllabus",
                expected_page_number=2,
            )
        ],
        clock=lambda: next(ticks),
    )

    assert [item.correct_source_in_top_k for item in measurements] == [True, False]
    assert measurements[0].elapsed_seconds == pytest.approx(0.2)
    assert measurements[1].elapsed_seconds == pytest.approx(0.3)

    output = tmp_path / "comparison.csv"
    write_comparison_csv(output, measurements)
    assert "correct_source_in_top_k" in output.read_text(encoding="utf-8")
