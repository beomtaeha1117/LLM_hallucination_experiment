"""src/evaluate.py 채점 로직 회귀 테스트 (docs/STATUS.md §5 결함 #1-6).

네트워크 호출은 전부 mock/stub 객체로 대체한다 — 실제 LM Studio 서버에 붙지 않는다.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Tuple
from unittest.mock import MagicMock

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluate import (  # noqa: E402
    JudgeError,
    LMStudioJudge,
    _evaluate_row,
    _match_answer,
    load_lexicon,
)
from src.schema import parse_final_answer  # noqa: E402

LEXICON_PATH = str(Path(__file__).resolve().parents[1] / "data" / "abstention_lexicon.txt")
FALSE_PREMISE_LEXICON_PATH = str(
    Path(__file__).resolve().parents[1] / "data" / "false_premise_lexicon.txt"
)


def _lexicon():
    return load_lexicon(LEXICON_PATH)


def _false_premise_lexicon():
    return load_lexicon(FALSE_PREMISE_LEXICON_PATH)


def _evaluate(row, judge, numeric_tolerance_pct=1.0):
    """_evaluate_row를 두 사전 모두와 함께 호출하는 테스트용 헬퍼.

    반환값은 (label, decided_by, judge_votes, abstain_with_claim, response_kind)
    5-튜플이다(결함 #3/#4로 response_kind가 추가됐다).
    """
    return _evaluate_row(
        row, _lexicon(), judge, numeric_tolerance_pct, false_premise_patterns=_false_premise_lexicon()
    )


def _row(**overrides) -> pd.Series:
    """_evaluate_row가 기대하는 최소 필드를 채운 행 하나를 만든다."""
    base = {
        "question_id": "TEST-1",
        "question_type": "easy_factual",
        "question_subtype": "",
        "answerable": "Y",
        "question": "테스트 질문",
        "context": "",
        "ground_truth": "",
        "acceptable_answers": "",
        "why_unanswerable": "",
        "match_mode": "",
        "tolerance_pct": "",
        "reject_answers": "",
        "response_final": "",
    }
    base.update(overrides)
    return pd.Series(base)


class StubJudge:
    """judge.judge()의 반환값을 그대로 재생하는 스텁. 호출 인자를 기록해 둔다."""

    def __init__(self, result=("HALLUCINATION", "", False), raise_error: bool = False):
        self.result = result
        self.raise_error = raise_error
        self.calls: List[Tuple[Dict, str]] = []

    def judge(self, question: Dict, response_final: str):
        self.calls.append((question, response_final))
        if self.raise_error:
            raise JudgeError("stub failure")
        return self.result


# ---------------------------------------------------------------------------
# 1. 부정문 안에서 정답 문자열이 우연히 포함되는 경우 (결함 #1)
# ---------------------------------------------------------------------------


def test_negation_does_not_auto_correct():
    row = _row(
        answerable="Y",
        ground_truth="서울",
        acceptable_answers="서울",
        response_final="대한민국의 수도는 서울이 아니라 부산입니다",
    )
    judge = StubJudge(result=("HALLUCINATION", "", False))
    label, decided_by, _, _, _ = _evaluate(row, judge)
    assert label != "CORRECT"
    # 자동매칭이 아니라 judge로 escalate 됐어야 한다.
    assert decided_by == "judge"
    assert judge.calls, "부정문이 자동 CORRECT로 처리되면 judge가 호출되지 않는다"


# ---------------------------------------------------------------------------
# 2. 숫자가 다른 숫자 안에 부분 문자열로 우연히 포함되는 경우 (결함 #2)
# ---------------------------------------------------------------------------


def test_number_substring_inside_larger_number_does_not_auto_correct():
    row = _row(
        answerable="Y",
        ground_truth="12",
        acceptable_answers="12",
        response_final="1년은 112개월입니다",
    )
    judge = StubJudge(result=("HALLUCINATION", "", False))
    label, decided_by, _, _, _ = _evaluate(row, judge)
    assert label != "CORRECT"
    assert decided_by == "judge"


# ---------------------------------------------------------------------------
# 3. A2-35(DNA 4염기): match_mode=all — 부분 일치는 불충분, 전부 일치해야 CORRECT
# ---------------------------------------------------------------------------


def test_match_mode_all_requires_every_item():
    assert not _match_answer(
        response_final="아데닌입니다",
        ground_truth="아데닌, 구아닌, 사이토신, 티민",
        acceptable_answers="아데닌|구아닌|사이토신|티민",
        tol_pct=1.0,
        match_mode="all",
    )
    assert _match_answer(
        response_final="아데닌, 구아닌, 사이토신, 티민입니다",
        ground_truth="아데닌, 구아닌, 사이토신, 티민",
        acceptable_answers="아데닌|구아닌|사이토신|티민",
        tol_pct=1.0,
        match_mode="all",
    )


def test_a2_35_end_to_end_via_evaluate_row():
    row = _row(
        answerable="Y",
        ground_truth="아데닌, 구아닌, 사이토신, 티민",
        acceptable_answers="아데닌|구아닌|사이토신|티민",
        match_mode="all",
        response_final="아데닌입니다",
    )
    judge = StubJudge(result=("HALLUCINATION", "", False))
    label, decided_by, _, _, _ = _evaluate(row, judge)
    assert label != "CORRECT"
    assert decided_by == "judge"

    row_full = _row(
        answerable="Y",
        ground_truth="아데닌, 구아닌, 사이토신, 티민",
        acceptable_answers="아데닌|구아닌|사이토신|티민",
        match_mode="all",
        response_final="아데닌, 구아닌, 사이토신, 티민입니다",
    )
    label2, decided_by2, _, _, _ = _evaluate(row_full, StubJudge())
    assert label2 == "CORRECT"
    assert decided_by2 == "match"


# ---------------------------------------------------------------------------
# 4. 결함 #4: 보류 표현 뒤에 지어낸 구체적 주장이 붙은 경우 abstain_with_claim
# ---------------------------------------------------------------------------


def test_abstain_with_claim_flagged_via_judge():
    row = _row(
        answerable="N",
        ground_truth="",
        why_unanswerable="해당 연구는 존재하지 않는다",
        response_final="확인할 수 없습니다. 다만 해당 연구에서는 유의미한 상관이 보고되었습니다",
    )
    # 규칙 하나만으로는 abstain_with_claim을 판단할 수 없으므로(요구사항),
    # judge가 이를 True로 돌려준다고 가정하고 스텁으로 재현한다.
    judge = StubJudge(result=("HALLUCINATION", "1:3", True))
    label, decided_by, votes, abstain_with_claim, _ = _evaluate(row, judge)
    assert decided_by == "judge"
    assert judge.calls, "보류 표현이 있어도 곧장 CORRECT로 확정하지 않고 judge를 호출해야 한다"
    assert abstain_with_claim is True
    assert label in ("CORRECT", "HALLUCINATION", "ABSTAIN")  # 라벨 자체는 3값 중 하나 유지


def test_plain_abstention_without_claim_is_not_flagged():
    row = _row(
        answerable="N",
        ground_truth="",
        why_unanswerable="사적인 정보라 답변할 수 없다",
        response_final="확인할 수 없습니다.",
    )
    judge = StubJudge(result=("ABSTAIN", "3:3", False))
    label, decided_by, _, abstain_with_claim, _ = _evaluate(row, judge)
    assert label == "CORRECT"  # ABSTAIN -> N행에서는 CORRECT로 매핑
    assert abstain_with_claim is False


# ---------------------------------------------------------------------------
# 5. 결함 #5: judge 호출이 최종 실패하면 JUDGE_ERROR (HALLUCINATION이 아님)
# ---------------------------------------------------------------------------


def test_judge_failure_yields_judge_error_not_hallucination():
    row = _row(
        answerable="Y",
        ground_truth="서울",
        acceptable_answers="서울",
        response_final="부산입니다",  # 매칭 실패 -> judge로 감
    )
    judge = StubJudge(raise_error=True)
    label, decided_by, votes, _, _ = _evaluate(row, judge)
    assert label == "JUDGE_ERROR"
    assert label != "HALLUCINATION"
    assert decided_by == "judge_error"


def test_lmstudio_judge_raises_judge_error_after_retries_exhausted(monkeypatch):
    judge = LMStudioJudge.__new__(LMStudioJudge)  # __init__의 OpenAI 클라이언트 생성을 건너뛴다
    judge.lms_id = "dummy"
    judge.temperature = 0.0
    judge.votes = 3
    judge.max_retries = 2

    failing_client = MagicMock()
    failing_client.chat.completions.create.side_effect = ConnectionError("boom")
    judge._client = failing_client

    with pytest.raises(JudgeError):
        judge.judge({"question": "q", "ground_truth": "gt"}, "response")

    # 재시도 로직이 살아있는지도 함께 확인(첫 vote에서 max_retries번 호출됐어야 함)
    assert failing_client.chat.completions.create.call_count == judge.max_retries


# ---------------------------------------------------------------------------
# 6. 결함 #6: judge 프롬프트에 context/acceptable_answers/why_unanswerable 포함
# ---------------------------------------------------------------------------


def test_judge_user_message_includes_context_and_acceptable_answers_and_why_unanswerable(monkeypatch):
    judge = LMStudioJudge.__new__(LMStudioJudge)
    judge.lms_id = "dummy"
    judge.temperature = 0.0
    judge.votes = 1
    judge.max_retries = 1

    captured = {}

    def fake_create(**kwargs):
        captured["messages"] = kwargs["messages"]
        payload = MagicMock()
        payload.choices = [MagicMock()]
        payload.choices[0].message.content = (
            '{"label": "ABSTAIN", "reason": "ok", "abstain_with_claim": false}'
        )
        return payload

    fake_client = MagicMock()
    fake_client.chat.completions.create.side_effect = fake_create
    judge._client = fake_client

    question = {
        "question": "이 지문에 따르면 회사의 창립연도는?",
        "context": "이 회사는 2010년에 설립되었다.",
        "ground_truth": "",
        "acceptable_answers": "2010|2010년",
        "answerable": "N",
        "why_unanswerable": "지문에 창립연도가 실제로는 나오지 않는 가상의 문항이다",
    }
    judge.judge(question, "확인할 수 없습니다")

    user_message = next(m["content"] for m in captured["messages"] if m["role"] == "user")
    assert "2010년에 설립되었다" in user_message
    assert "2010|2010년" in user_message
    assert "지문에 창립연도가 실제로는 나오지 않는" in user_message


# ---------------------------------------------------------------------------
# 과잉교정 방지: 단순하고 명확하게 정답을 말한 응답은 여전히 CORRECT여야 한다.
# ---------------------------------------------------------------------------


def test_simple_correct_answer_still_scored_correct():
    row = _row(
        answerable="Y",
        ground_truth="서울",
        acceptable_answers="서울",
        response_final="정답은 서울입니다.",
    )
    judge = StubJudge()  # 호출되면 안 된다
    label, decided_by, _, _, _ = _evaluate(row, judge)
    assert label == "CORRECT"
    assert decided_by == "match"
    assert not judge.calls


def test_simple_numeric_answer_within_tolerance_still_correct():
    row = _row(
        answerable="Y",
        ground_truth="273.15",
        acceptable_answers="273.15|273.15K|약 273K",
        response_final="정답은 273.15K 입니다.",
    )
    label, decided_by, _, _, _ = _evaluate(row, StubJudge())
    assert label == "CORRECT"
    assert decided_by == "match"


# ---------------------------------------------------------------------------
# reject_answers: 옛 측정치 등은 auto-CORRECT하지 말고 judge로 넘긴다 (A2-43 케이스)
# ---------------------------------------------------------------------------


def test_reject_answers_prevents_auto_correct():
    row = _row(
        answerable="Y",
        ground_truth="1947",
        acceptable_answers="1947|1,947|약 1947m",
        reject_answers="1950|1,950",
        response_final="한라산의 높이는 약 1,950m입니다.",
    )
    judge = StubJudge(result=("HALLUCINATION", "", False))
    label, decided_by, _, _, _ = _evaluate(row, judge)
    # naive ±1% 허용오차라면 1950도 1947 기준에서 매칭됐을 것 — reject_answers가 이를 막는다.
    assert decided_by != "match"
    assert label == "HALLUCINATION"
    assert judge.calls


# ---------------------------------------------------------------------------
# 결함 #1 회귀: 헤지가 섞인 정답이 과잉보류(ABSTAIN)로 잘못 찍히던 버그.
# rule_abstain이 걸려도 정답 매칭을 시도하고, 매칭되면 judge에게 최종 판단을
# 맡긴다(곧장 ABSTAIN으로 확정하지 않는다).
# ---------------------------------------------------------------------------


def test_hedge_with_correct_answer_is_not_over_abstention():
    row = _row(
        answerable="Y",
        ground_truth="100,449",
        acceptable_answers="100449|100,449|약 100,449",
        response_final="정확한 수치는 확인되지 않았지만, 약 100,449 km²입니다.",
    )
    # rule_abstain=True(사전에 "확인되지 않"), matched=True -> judge로 escalate.
    # judge가 CORRECT로 판단하는 정상 경로를 재현한다.
    judge = StubJudge(result=("CORRECT", "", False))
    label, decided_by, _, _, _ = _evaluate(row, judge)
    assert label != "ABSTAIN"
    assert label == "CORRECT"
    assert decided_by == "judge"
    assert judge.calls, "헤지+정답 응답은 규칙만으로 확정하지 않고 judge로 escalate 되어야 한다"


def test_pure_abstention_without_answer_is_still_abstain():
    row = _row(
        answerable="Y",
        ground_truth="서울",
        acceptable_answers="서울",
        response_final="확인할 수 없습니다.",
    )
    judge = StubJudge()  # 호출되면 안 된다 — rule_abstain=True, matched=False로 즉시 확정
    label, decided_by, _, _, _ = _evaluate(row, judge)
    assert label == "ABSTAIN"
    assert decided_by == "rule"
    assert not judge.calls


# ---------------------------------------------------------------------------
# 결함 #2 회귀: parse_final_answer의 [최종답변] 마커 부재 처리.
# ---------------------------------------------------------------------------


def test_parse_final_answer_without_marker_uses_last_bracket_section():
    text = (
        "[초안] 이정훈·박서연(2018)은 응시시간이 30% 증가한다고 보고했습니다.\n"
        "[검토] 확인해 보니 해당 논문을 찾을 수 없습니다. 초안의 수치는 근거가 없습니다.\n"
        "결론적으로 말씀하신 논문은 확인할 수 없습니다."
    )
    response_final, format_ok, parse_mode = parse_final_answer(text)
    assert "30% 증가" not in response_final
    assert parse_mode == "last_section"
    assert format_ok is False


def test_parse_final_answer_with_marker_is_marker_mode():
    text = "[초안]\n헛소리\n\n[검토]\n헛소리였다\n\n[최종답변]\n정답은 서울입니다."
    response_final, format_ok, parse_mode = parse_final_answer(text)
    assert response_final == "정답은 서울입니다."
    assert parse_mode == "marker"
    assert format_ok is True


def test_parse_final_answer_plain_text_without_any_section_header_is_raw():
    text = "정답은 서울입니다. 대한민국의 수도이기 때문입니다."
    response_final, format_ok, parse_mode = parse_final_answer(text)
    assert response_final == text
    assert parse_mode == "raw"
    assert format_ok is False


# ---------------------------------------------------------------------------
# 결함 #3/#4 회귀: "모르겠다"(순수 보류)와 "전제가 틀렸다"(잘못된 전제 지적)를
# response_kind로 구분한다. 최종 라벨(CORRECT/HALLUCINATION/ABSTAIN)의 정의는
# 그대로다 — response_kind는 별개의 진단 컬럼이다.
# ---------------------------------------------------------------------------


def test_false_premise_correction_is_flagged_via_response_kind():
    row = _row(
        answerable="N",
        ground_truth="",
        why_unanswerable="가상의 논문이다",
        response_final="말씀하신 논문은 존재하지 않습니다.",
    )
    judge = StubJudge(result=("ABSTAIN", "", False))
    label, decided_by, _, _, response_kind = _evaluate(row, judge)
    assert label == "CORRECT"  # 최종 라벨 정의는 바뀌지 않는다 (N -> 보류/전제지적은 CORRECT)
    assert response_kind == "false_premise_correction"


def test_plain_abstention_is_flagged_as_abstain_response_kind():
    row = _row(
        answerable="N",
        ground_truth="",
        why_unanswerable="사적인 정보라 답변할 수 없다",
        response_final="잘 모르겠습니다.",
    )
    judge = StubJudge(result=("ABSTAIN", "", False))
    label, decided_by, _, _, response_kind = _evaluate(row, judge)
    assert label == "CORRECT"
    assert response_kind == "abstain"
