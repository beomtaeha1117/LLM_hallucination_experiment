"""검증 결과를 문항 파일에 반영하는 도구.

verify/worklist.html 에서 내려받은 CSV(또는 사람이 직접 채운 CSV)를 읽어,
확인된 행은 verified=YES 로 바꾸고 verify_action 에 확인 근거(출처 URL +
확인일자)를 덧붙인 새 문항 파일을 만든다.

안전장치:
- --out 이 --questions 와 같으면 거부한다 (원본 덮어쓰기 금지).
- 출처 URL이 없는 행은 verified=YES 로 바꾸지 않는다. 단, 외부 출처가
  필요 없는 유형(context_qa, context_qa_nogold)이거나
  unknowable(future/subjective) 인 행은 체크박스만으로 충분하다.
- 출력 파일을 쓰기 전에 schema.load_questions() 로 검증하고, 실패하면
  쓰지 않는다.

사용법:
    .venv/bin/python tools/apply_verification.py \
        --questions data/questions_v1_DRAFT.csv \
        --results verify/completed.csv \
        --out data/questions_v1.csv
"""

from __future__ import annotations

import argparse
import datetime
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from schema import load_questions  # noqa: E402

# 외부 출처 없이도 verified=YES 로 인정되는 question_type
NO_SOURCE_NEEDED_TYPES = {"context_qa", "context_qa_nogold"}
# 외부 출처 없이도 verified=YES 로 인정되는 unknowable 세부유형
NO_SOURCE_NEEDED_UNKNOWABLE_SUBTYPES = {"future", "subjective"}


def _no_source_required(row: pd.Series) -> bool:
    if row["question_type"] in NO_SOURCE_NEEDED_TYPES:
        return True
    if (
        row["question_type"] == "unknowable"
        and row["question_subtype"] in NO_SOURCE_NEEDED_UNKNOWABLE_SUBTYPES
    ):
        return True
    return False


def _truthy_checked(value: str) -> bool:
    return str(value).strip().upper() in ("YES", "Y", "TRUE", "1")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", required=True, help="원본 문항 CSV 경로")
    parser.add_argument("--results", required=True, help="검증 결과 CSV 경로 (worklist에서 내려받음)")
    parser.add_argument("--out", required=True, help="출력 문항 CSV 경로")
    args = parser.parse_args()

    if os.path.abspath(args.out) == os.path.abspath(args.questions):
        print("오류: --out 이 --questions 와 같은 파일입니다. 원본을 덮어쓸 수 없습니다.")
        sys.exit(1)

    questions_df = pd.read_csv(
        args.questions, dtype=str, keep_default_na=False, encoding="utf-8-sig"
    )
    results_df = pd.read_csv(
        args.results, dtype=str, keep_default_na=False, encoding="utf-8-sig"
    )

    results_by_id = {row["question_id"]: row for _, row in results_df.iterrows()}

    today = datetime.date.today().isoformat()

    newly_verified = []
    refused = []

    for idx in questions_df.index:
        row = questions_df.loc[idx]
        qid = row["question_id"]
        if row["verified"] == "YES":
            continue
        if qid not in results_by_id:
            continue

        result_row = results_by_id[qid]
        checked = _truthy_checked(result_row.get("checked", result_row.get("verified_by_human", "")))
        source_url = str(result_row.get("source_url", "")).strip()

        if not checked:
            continue

        if not source_url and not _no_source_required(row):
            refused.append(
                (qid, row["question_type"], "출처 URL 없음 (외부 출처가 필요한 유형)")
            )
            continue

        confirm_note = f"확인: {source_url} ({today})" if source_url else f"확인: 검토완료 ({today})"
        existing_action = row["verify_action"]
        new_action = f"{existing_action} | {confirm_note}" if existing_action else confirm_note

        questions_df.at[idx, "verified"] = "YES"
        questions_df.at[idx, "verify_action"] = new_action
        newly_verified.append(qid)

    if refused:
        print("=== 거부된 행 (verified=YES로 바뀌지 않음) ===")
        for qid, qtype, reason in refused:
            print(f"  {qid} ({qtype}): {reason}")
        print()

    # 출력 전 스키마 검증
    tmp_path = args.out + ".tmp_check"
    questions_df.to_csv(tmp_path, index=False, encoding="utf-8-sig")
    try:
        load_questions(tmp_path)
    except ValueError as e:
        os.remove(tmp_path)
        print("오류: 갱신된 문항 파일이 스키마 검증을 통과하지 못해 쓰지 않았습니다.")
        print(str(e))
        sys.exit(1)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    os.replace(tmp_path, args.out)

    total = len(questions_df)
    still_no = int((questions_df["verified"] != "YES").sum())
    print(f"=== 요약 ===")
    print(f"새로 검증됨: {len(newly_verified)}")
    print(f"여전히 NO: {still_no} / {total}")
    print(f"거부됨: {len(refused)}")
    print()
    print("question_type 별 남은 verified != YES 건수:")
    remaining = questions_df[questions_df["verified"] != "YES"]
    for qtype, count in remaining["question_type"].value_counts().sort_index().items():
        print(f"  {qtype:<20} {count}")

    print(f"\n출력: {args.out}")


if __name__ == "__main__":
    main()
