"""200문항 스켈레톤 CSV 생성기.

data/questions_TEMPLATE.csv 를 만든다. 연구자가 손으로 question / context /
ground_truth / acceptable_answers / why_unanswerable 을 채워 넣는 빈 틀이다.
question_type, answerable, verified, verify_action(힌트)은 미리 채워서
사람이 어떤 검증을 해야 하는지 바로 보이게 한다.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from typing import List

# src/schema.py 의 QUESTION_COLUMNS 와 반드시 일치시킨다 (컬럼명/순서 추측 금지).
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from schema import QUESTION_COLUMNS  # noqa: E402

TEMPLATE_PATH = os.path.join(
    os.path.dirname(__file__), "..", "data", "questions_TEMPLATE.csv"
)

# docs/question-design.md §3 의 200문항 구성표.
# (question_type, answerable, count, id_prefix, verify_action 힌트)
# verify_action 힌트는 §0(검증 절차)·§1(유형별 설명)의 실제 표현을 그대로 가져온 것이다.
ROW_SPEC = [
    ("easy_factual", "Y", 20, "A1", "교과서·국가기관·공식 DB 등 출처 URL 기록"),
    ("hard_factual", "Y", 50, "A2", "교과서·국가기관·공식 DB 등 출처 URL 기록"),
    ("numeric", "Y", 15, "A3", "허용오차 규칙 정할 것(예: ±1%)"),
    ("context_qa", "Y", 15, "A4", "지문 자체가 근거이므로 외부 검증 불필요"),
    ("fake_paper", "N", 15, "U1", "RISS·구글학술 검색 0건 확인 필요(우연히 존재하면 저자명·제목 교체)"),
    ("fake_concept", "N", 25, "U2", "구글·RISS 검색 0건 확인"),
    # false_premise(U3)는 아래에서 하위 4종을 순환시켜 question_subtype 을 개별 지정한다.
    # unknowable(U4)도 아래에서 하위 3종을 순환시킨다 — 두 유형 모두 이 표에는 없다.
    ("fake_statute", "N", 5, "U5", "국가법령정보센터에서 조항 존재 여부 확인 필수"),
    ("context_qa_nogold", "N", 5, "U6", "지문 자체가 근거이므로 외부 검증 불필요"),
]

# U4 unknowable 공통 검증 힌트 (docs/question-design.md §1 U4).
U4_VERIFY_HINT = "원리적으로 답변 불가함을 확인(미래 시행/비공개 정보/주관적 합의 등)"

# U3 잘못된 전제 하위 4종 (docs/question-design.md §1 U3).
# 시대착오가 검증하기 가장 쉬워서 가장 큰 비중을 준다: 8 / 12 / 7 / 8 = 35.
# subtype 값(event/anachronism/number/person)은 question_subtype 컬럼에 들어가며
# src/schema.py 의 QUESTION_SUBTYPES["false_premise"]와 반드시 일치시킨다.
U3_SUBTYPES = [
    ("event", "해당 사건·저술 등이 실제로 존재하는지 검색으로 확인 필수", 8),
    ("anachronism", "연대만 대조하면 되므로 검증이 쉬움(예: 기관 창설연도 확인)", 12),
    ("number", "전제된 수치가 사실과 다른지 교과서 등으로 확인", 7),
    ("person", "실제 인물명을 확인한 뒤 문항 속 인물이 존재하지 않는지 검색", 8),
]

# U4 답변불가 하위 3종 (docs/question-design.md §1 U4).
# subtype 값(future/private/subjective)은 question_subtype 컬럼에 들어가며
# src/schema.py 의 QUESTION_SUBTYPES["unknowable"]와 반드시 일치시킨다.
# 15문항을 균등 배분한다: 5 / 5 / 5.
U4_SUBTYPES = ["future", "private", "subjective"]


def _make_rows() -> List[dict]:
    """ROW_SPEC + U3/U4 순환 배분에 따라 200개 스켈레톤 행을 만든다."""
    rows: List[dict] = []

    for question_type, answerable, count, prefix, verify_hint in ROW_SPEC:
        for i in range(1, count + 1):
            rows.append(
                {
                    "question_id": f"{prefix}-{i:02d}",
                    "question_type": question_type,
                    "question_subtype": "",
                    "answerable": answerable,
                    "question": "",
                    "context": "",
                    "ground_truth": "",
                    "acceptable_answers": "",
                    "why_unanswerable": "",
                    "verify_action": verify_hint,
                    "verified": "NO",
                }
            )

    # U3 false_premise: 하위 4종을 8/12/7/8로 순환 배분한다 (docs §1 U3 참조).
    # verify_action은 순수 검증 힌트로 되돌리고, 하위 유형은 question_subtype으로 뺀다.
    idx = 1
    for subtype, verify_hint, count in U3_SUBTYPES:
        for _ in range(count):
            rows.append(
                {
                    "question_id": f"U3-{idx:02d}",
                    "question_type": "false_premise",
                    "question_subtype": subtype,
                    "answerable": "N",
                    "question": "",
                    "context": "",
                    "ground_truth": "",
                    "acceptable_answers": "",
                    "why_unanswerable": "",
                    "verify_action": verify_hint,
                    "verified": "NO",
                }
            )
            idx += 1

    # U4 unknowable: 하위 3종을 5/5/5로 순환 배분한다 (docs §1 U4 참조).
    idx = 1
    for subtype in U4_SUBTYPES:
        for _ in range(5):
            rows.append(
                {
                    "question_id": f"U4-{idx:02d}",
                    "question_type": "unknowable",
                    "question_subtype": subtype,
                    "answerable": "N",
                    "question": "",
                    "context": "",
                    "ground_truth": "",
                    "acceptable_answers": "",
                    "why_unanswerable": "",
                    "verify_action": U4_VERIFY_HINT,
                    "verified": "NO",
                }
            )
            idx += 1

    return rows


def write_template(path: str, force: bool) -> None:
    """스켈레톤 CSV 를 path 에 쓴다. force 가 아니면 기존 파일을 덮어쓰지 않는다."""
    if os.path.exists(path) and not force:
        raise FileExistsError(
            f"{path} 이미 존재합니다. 덮어쓰려면 --force 를 사용하세요."
        )

    rows = _make_rows()

    os.makedirs(os.path.dirname(path), exist_ok=True)
    # newline="" : csv 모듈 권장 관례. lineterminator 는 기본값 사용.
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=QUESTION_COLUMNS)
        writer.writeheader()
        for row in rows:
            # QUESTION_COLUMNS 순서대로, 빈 칸은 빈 문자열(NaN 방지)로 명시적으로 채운다.
            writer.writerow({col: row.get(col, "") for col in QUESTION_COLUMNS})

    print(f"{len(rows)}개 행을 {path} 에 썼습니다.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out", default=TEMPLATE_PATH, help="출력 CSV 경로 (기본: data/questions_TEMPLATE.csv)"
    )
    parser.add_argument(
        "--force", action="store_true", help="기존 파일이 있어도 덮어쓴다"
    )
    args = parser.parse_args()

    try:
        write_template(args.out, args.force)
    except FileExistsError as e:
        print(f"오류: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
