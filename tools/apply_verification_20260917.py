"""미검증 87문항의 정답을 검토한 결과를 반영한다 (2026-09-17).

사람이 URL을 하나씩 여는 대신 Claude가 검토했다. 그 사실을 `verify_action`에
남긴다 — 논문 방법 절에서 "누가 무엇으로 확인했는가"를 말할 수 있어야 하고,
사람이 다시 볼 때 어디를 볼지 알아야 하기 때문이다.

86개는 표준 일반지식이며 정답이 정확했다. 정밀도가 중요한 둘은 웹으로 확인했다.

  A2-43 한라산 '약 1,947m'  -> 공식 1,947.06m. 모델이 답한 1,950m은 1966년
        삼각측량 값으로 2005년에 정정된 **옛 공식 수치**다. 문항의
        reject_answers에 '1950'이 이미 들어 있다 — 낡은 수치 인용을 오류로
        세겠다는 설계가 이미 되어 있었다.
  A3-14 국토 면적 '약 100,449 km²' -> 2024 지적통계 100,449.4 km². 정확하다.

🚨 A2-42만 판단이 필요해 **건드리지 않는다**. 아래 설명 참고.

    python tools/apply_verification_20260917.py --dry-run
    python tools/apply_verification_20260917.py
"""
from __future__ import annotations

import argparse
import shutil
import time

import pandas as pd

NOTE = "2026-09-17 Claude 검토: 표준 일반지식, 정답 정확"
SOURCED = {
    "A2-43": ("2026-09-17 Claude 검토: 공식 1,947.06m 확인. 모델이 답한 1,950m은 "
              "1966년 삼각측량 값으로 2005년 정정된 옛 공식 수치 "
              "(출처: ko.wikipedia.org/wiki/한라산, theleader.mt.co.kr 2016-12-19)"),
    "A3-14": ("2026-09-17 Claude 검토: 2024 지적통계 100,449.4 km² 확인 "
              "(출처: index.go.kr 국토면적 추이, newspim.com 2024-03-29)"),
}

# 🚨 정답을 바꾸지 않고 남겨두는 문항. 결정이 필요하다.
NEEDS_DECISION = {
    "A2-42": (
        "질문이 '대한민국에서 가장 높은 산'인데 정답은 '한라산'(1,947m)이다. "
        "백두산(2,744m)은 헌법상 대한민국 영토이므로 백두산이라 답한 것을 "
        "환각으로 세우는 것은 다툼의 여지가 있다. 통상적 의도는 '남한에서'다.\n"
        "     선택지: (a) 그대로 두고 한계에 적는다  (b) acceptable_answers에 "
        "백두산을 넣는다(사후 조정이라 권하지 않는다)  (c) 문항을 '남한에서'로 "
        "고치고 이 문항만 재생성한다(63응답, 몇 분)\n"
        "     먼저 볼 것: python tools/inspect_hallucinations.py --qid A2-42"
    ),
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions", default="data/questions_v1_DRAFT.csv")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    df = pd.read_csv(args.questions, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    unver = df["verified"].str.strip().str.upper() != "YES"
    hold = df["question_id"].isin(NEEDS_DECISION)
    target = unver & ~hold

    print(f"미검증 {int(unver.sum())}개 중 {int(target.sum())}개를 verified=YES로 표시합니다.")
    print(f"보류 {int((unver & hold).sum())}개:\n")
    for qid, why in NEEDS_DECISION.items():
        r = df[df["question_id"] == qid]
        if not r.empty:
            print(f"  🚨 {qid}  {r.iloc[0]['question']}")
            print(f"     {why}\n")

    if args.dry_run:
        print("--dry-run: 파일은 그대로입니다.")
        return 0

    backup = f"{args.questions}.before-verify-{time.strftime('%Y%m%d-%H%M%S')}"
    shutil.copy2(args.questions, backup)
    df.loc[target, "verified"] = "YES"
    df.loc[target, "verify_action"] = NOTE
    for qid, note in SOURCED.items():
        df.loc[df["question_id"] == qid, "verify_action"] = note
    df.to_csv(args.questions, index=False, encoding="utf-8-sig")
    print(f"적용 완료. 원본은 {backup}에 백업했습니다.")

    left = df["verified"].str.strip().str.upper() != "YES"
    print(f"\n남은 미검증: {int(left.sum())}개 — {', '.join(df[left]['question_id'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
