"""데이터 스키마 정의 및 문항 파일 검증/파싱 유틸리티."""

from __future__ import annotations

import re
from typing import Tuple

import pandas as pd

QUESTION_COLUMNS = [
    "question_id",
    "question_type",
    "question_subtype",
    "answerable",
    "question",
    "context",
    "ground_truth",
    "acceptable_answers",
    "why_unanswerable",
    "verify_action",
    "verified",
]

# question_type: 분석 단위가 되는 거친(coarse) 유형. 셀 크기를 확보해 통계 검정이
# 가능하게 하는 것이 목적이다.
COARSE_QUESTION_TYPES = [
    "easy_factual",
    "hard_factual",
    "numeric",
    "context_qa",
    "fake_paper",
    "fake_concept",
    "fake_statute",
    "false_premise",
    "unknowable",
    "context_qa_nogold",
]

# question_subtype: 세부 유형. 거친 유형은 통계 검정용 셀 크기 확보가 목적이고,
# 세부 유형은 논문 '연구 결과' 절의 유형별 분석 재료로 쓴다. 부모 유형이 아래
# 목록에 없으면(=위 리스트 대상이 아니면) question_subtype은 항상 빈 문자열이어야 한다.
QUESTION_SUBTYPES = {
    "false_premise": ["event", "anachronism", "number", "person"],
    "unknowable": ["future", "private", "subjective"],
}

RAW_COLUMNS = [
    "run_id",
    "is_mock",
    "question_id",
    "question_type",
    "question_subtype",
    "answerable",
    "question",
    "context",
    "ground_truth",
    "model_key",
    "lms_id",
    "quant",
    "engine",
    "prompt_type",
    "repeat",
    "seed",
    "temperature",
    "top_p",
    "max_tokens",
    "reasoning_effort",
    "response_raw",
    "response_final",
    "prompt_tokens",
    "completion_tokens",
    "latency_ms",
    "finish_reason",
    "format_ok",
    "parse_mode",
    "timestamp",
]

EVAL_COLUMNS = RAW_COLUMNS + [
    "label",
    "decided_by",
    "judge_votes",
    "abstain_with_claim",
    "response_kind",
    "human_label",
    "human_rater_id",
]

_FINAL_MARKER = "[최종답변]"
# [최종답변] 다음에 다른 섹션 헤더(예: 모델이 실수로 덧붙인 [참고] 등)가
# 이어지는 경우를 대비해, 응답 끝에 남은 트레일링 브래킷 섹션 헤더 한 줄을 제거한다.
_TRAILING_SECTION_RE = re.compile(r"\n\s*\[[^\[\]\n]{1,20}\]\s*$")
# 마커가 없을 때 "마지막 대괄호 섹션 헤더"를 찾는 데 쓴다(결함 #2 대응). 헤더
# 이름은 무관하다 — [초안]/[검토]/[재진술]/[배경] 등 무엇이든 텍스트에 등장하는
# 마지막 `[...]` 한 줄을 찾아 그 이후 텍스트를 취한다.
_SECTION_HEADER_RE = re.compile(r"\[[^\[\]\n]{1,20}\]")


def load_questions(path: str) -> pd.DataFrame:
    """문항 CSV를 읽고 스키마를 검증한다.

    검증 항목:
    - 필수 컬럼 존재
    - answerable ∈ {Y, N}
    - answerable == Y 행은 ground_truth가 비어있지 않음
    - answerable == N 행은 ground_truth가 비어있고 why_unanswerable이 비어있지 않음
    - question_id 유일성
    - question_type이 COARSE_QUESTION_TYPES 안에 있음
    - question_subtype이 부모 question_type에 허용된 값이거나 빈 문자열임

    검증에 실패하면 문제가 된 question_id 목록을 포함한 ValueError를 던진다.
    """
    # encoding="utf-8-sig": BOM이 있어도 없어도 투명하게 처리된다. Excel에서 저장한
    # questions_TEMPLATE.csv류(BOM 포함)를 plain utf-8로 읽으면 첫 컬럼명이
    # "﻿question_id"가 되어 컬럼 누락 오류로 오진되므로 반드시 utf-8-sig를 쓴다.
    df = pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8-sig")

    missing_cols = [c for c in QUESTION_COLUMNS if c not in df.columns]
    if missing_cols:
        raise ValueError(f"문항 파일에 필수 컬럼이 없습니다: {missing_cols}")

    errors: list[str] = []

    bad_type = df[~df["question_type"].isin(COARSE_QUESTION_TYPES)]
    if not bad_type.empty:
        errors.append(
            "question_type이 허용된 거친 유형이 아닌 question_id: "
            + ", ".join(bad_type["question_id"].tolist())
        )

    def _bad_subtype(row: pd.Series) -> bool:
        allowed = QUESTION_SUBTYPES.get(row["question_type"], [])
        subtype = row["question_subtype"].strip()
        if not allowed:
            return subtype != ""
        return subtype not in allowed

    bad_subtype = df[df.apply(_bad_subtype, axis=1)]
    if not bad_subtype.empty:
        errors.append(
            "question_subtype이 부모 question_type에 허용되지 않는 question_id: "
            + ", ".join(bad_subtype["question_id"].tolist())
        )

    bad_answerable = df[~df["answerable"].isin(["Y", "N"])]
    if not bad_answerable.empty:
        errors.append(
            "answerable이 Y/N이 아닌 question_id: "
            + ", ".join(bad_answerable["question_id"].tolist())
        )

    y_rows = df[df["answerable"] == "Y"]
    bad_y = y_rows[y_rows["ground_truth"].str.strip() == ""]
    if not bad_y.empty:
        errors.append(
            "answerable==Y인데 ground_truth가 비어있는 question_id: "
            + ", ".join(bad_y["question_id"].tolist())
        )

    n_rows = df[df["answerable"] == "N"]
    bad_n_gt = n_rows[n_rows["ground_truth"].str.strip() != ""]
    if not bad_n_gt.empty:
        errors.append(
            "answerable==N인데 ground_truth가 비어있지 않은 question_id: "
            + ", ".join(bad_n_gt["question_id"].tolist())
        )
    bad_n_why = n_rows[n_rows["why_unanswerable"].str.strip() == ""]
    if not bad_n_why.empty:
        errors.append(
            "answerable==N인데 why_unanswerable이 비어있는 question_id: "
            + ", ".join(bad_n_why["question_id"].tolist())
        )

    dup_ids = df["question_id"][df["question_id"].duplicated()].unique().tolist()
    if dup_ids:
        errors.append("question_id가 중복됩니다: " + ", ".join(dup_ids))

    if errors:
        raise ValueError("문항 파일 검증 실패:\n" + "\n".join(errors))

    return df


def parse_final_answer(text: str) -> Tuple[str, bool, str]:
    """응답 텍스트에서 채점 대상이 될 최종 텍스트를 추출한다.

    (response_final, format_ok, parse_mode)의 3-튜플을 반환한다.
    format_ok은 기존 의미 그대로 "[최종답변] 마커가 있었는가"이며(조건별 형식
    준수율 지표이므로 의미를 바꾸지 않는다), parse_mode는 어느 경로로 텍스트를
    골라냈는지를 기록하는 진단용 컬럼이다:

    - "marker": [최종답변] 마커가 있어 그 뒤를 취했다(format_ok=True).
    - "last_section": 마커는 없지만 [초안]/[검토]/[재진술]/[배경] 등 대괄호
      섹션 헤더가 있어(이름은 무관), 텍스트에 등장하는 마지막 헤더 이후를
      취했다(format_ok=False). P2/P3/P2L처럼 초안 -> 검토 절차가 있는 조건에서
      마커를 빠뜨린 경우, 검토 단계에서 스스로 철회한 초안 내용이 채점 대상에
      섞여 들어가는 것을 막기 위함이다.
    - "raw": 대괄호 섹션 헤더가 전혀 없어 전체 텍스트를 그대로 취했다
      (format_ok=False). P0/P1/P4/P5처럼 섹션 구조가 없는 조건의 정상 동작이다.
    """
    idx = text.rfind(_FINAL_MARKER)
    if idx != -1:
        after = text[idx + len(_FINAL_MARKER):]
        after = _TRAILING_SECTION_RE.sub("", after)
        return after.strip(), True, "marker"

    last_header = None
    for m in _SECTION_HEADER_RE.finditer(text):
        last_header = m
    if last_header is not None:
        after = text[last_header.end():]
        after = _TRAILING_SECTION_RE.sub("", after)
        return after.strip(), False, "last_section"

    return text.strip(), False, "raw"
