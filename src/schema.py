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
    "response_raw",
    "response_final",
    "prompt_tokens",
    "completion_tokens",
    "latency_ms",
    "finish_reason",
    "format_ok",
    "timestamp",
]

EVAL_COLUMNS = RAW_COLUMNS + [
    "label",
    "decided_by",
    "judge_votes",
    "abstain_with_claim",
    "human_label",
    "human_rater_id",
]

_FINAL_MARKER = "[최종답변]"
# [최종답변] 다음에 다른 섹션 헤더(예: 모델이 실수로 덧붙인 [참고] 등)가
# 이어지는 경우를 대비해, 응답 끝에 남은 트레일링 브래킷 섹션 헤더 한 줄을 제거한다.
_TRAILING_SECTION_RE = re.compile(r"\n\s*\[[^\[\]\n]{1,20}\]\s*$")


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


def parse_final_answer(text: str) -> Tuple[str, bool]:
    """응답 텍스트에서 마지막 [최종답변] 마커 이후 내용을 추출한다.

    마커가 없으면 (전체 텍스트.strip(), False)를 반환한다.
    마커가 있으면 (마커 이후 텍스트.strip(), True)를 반환하며, 그 뒤에 다른
    브래킷 섹션 헤더가 트레일링으로 붙어 있으면 제거한다.
    """
    idx = text.rfind(_FINAL_MARKER)
    if idx == -1:
        return text.strip(), False

    after = text[idx + len(_FINAL_MARKER):]
    after = _TRAILING_SECTION_RE.sub("", after)
    return after.strip(), True
