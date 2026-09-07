"""문항 CSV 검증 리포트 도구.

src.schema.load_questions() 를 돌려서 스키마 위반을 사람이 읽기 좋은 형태로
보고한다. question_type 별 행수, Y/N 균형, verified=NO 잔여 건수를 함께 낸다.
"""

from __future__ import annotations

import argparse
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from schema import load_questions  # noqa: E402


def print_report(path: str) -> None:
    """path 의 문항 파일을 로드해 검증 리포트를 출력한다."""
    print(f"=== 문항 파일 검증: {path} ===")

    try:
        df = load_questions(path)
    except ValueError as e:
        # load_questions 는 문제가 된 question_id 를 모아 ValueError 하나로 던진다.
        print("검증 실패:")
        print(str(e))
        # 스키마 위반이 있어도 원본을 읽어서 참고용 통계는 최대한 보여준다.
        try:
            df = pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8-sig")
        except Exception as read_err:  # noqa: BLE001 - 진단용 최대한 넓게 캐치
            print(f"(원본 파일도 읽지 못했습니다: {read_err})")
            return
        print("--- 참고: 원본 파일 기준 통계 (검증 실패 상태) ---")
    else:
        print("검증 통과: 스키마 오류 없음.")

    total = len(df)
    print(f"\n총 행수: {total}")

    print("\nquestion_type 별 행수:")
    for qtype, count in df["question_type"].value_counts().sort_index().items():
        print(f"  {qtype:<25} {count}")

    if "answerable" in df.columns:
        y_count = int((df["answerable"] == "Y").sum())
        n_count = int((df["answerable"] == "N").sum())
        other = total - y_count - n_count
        print(f"\nanswerable 균형: Y={y_count}, N={n_count}", end="")
        if other:
            print(f", 그 외/오류={other}")
        else:
            print()

    if "verified" in df.columns:
        not_verified = int((df["verified"] != "YES").sum())
        print(f"\nverified != YES (아직 검증 안 됨): {not_verified} / {total}")

    hits = check_fewshot_contamination(df)
    if hits:
        print("\n[경고] few-shot 예시와 동일한 문항이 있습니다 (P5 조건이 오염됩니다):")
        for h in hits:
            print("  -", h)
    else:
        print("\nfew-shot 예시 오염: 없음")

    print()



def check_fewshot_contamination(df, prompt_path: str = "prompts/P5_fewshot.txt") -> list:
    """few-shot 예시로 쓰인 질문이 문항 세트에 들어 있는지 검사한다.

    예시가 평가 문항과 겹치면 P5 조건에서만 정답이 프롬프트 안에 이미 들어 있는
    셈이 되어 조건 간 비교가 무의미해진다(오염).
    """
    import os
    import re as _re

    if not os.path.exists(prompt_path):
        return []
    text = open(prompt_path, encoding="utf-8").read()
    examples = [m.strip() for m in _re.findall(r"^질문:\s*(.+)$", text, _re.MULTILINE)]

    def norm(x: object) -> str:
        return _re.sub(r"[\s?.!,]", "", str(x))

    ex_norm = {norm(e) for e in examples}
    return [
        f"{row.question_id}: {row.question}"
        for row in df.itertuples()
        if norm(getattr(row, "question", "")) in ex_norm
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", help="검증할 문항 CSV 경로")
    args = parser.parse_args()
    print_report(args.path)


if __name__ == "__main__":
    main()
