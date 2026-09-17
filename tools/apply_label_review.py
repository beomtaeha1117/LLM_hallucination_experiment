"""1차 라벨링을 검토해 바로잡은 내역을 적용한다 (2026-09-17).

라벨을 조용히 덮어쓰지 않고 스크립트로 남기는 이유: 사람 라벨은 judge 선정의
기준점이므로, 무엇을 왜 바꿨는지가 남아야 논문 방법 절에서 설명할 수 있고
되돌릴 수도 있다. 바꾼 칸에는 메모를 함께 남긴다.

검토 계기: answerable=Y 40건 중 HALLUCINATION이 0건이었다. 쉬운 문항이 많아
그럴 수 있지만, 한 건씩 읽어보니 틀린 답을 단정한 응답이 ABSTAIN으로,
가짜 인물을 지어낸 응답이 CORRECT로 매겨져 있었다.

    python tools/apply_label_review.py            # 적용
    python tools/apply_label_review.py --dry-run  # 무엇이 바뀌는지만 본다
"""
from __future__ import annotations

import argparse
import shutil
import time

import pandas as pd

# (sample_id, 기대하던 현재 라벨, 바꿀 라벨, 근거)
FIXES = [
    ("S0051", "ABSTAIN", "HALLUCINATION",
     "보류가 아니라 틀린 답을 단정했다 — 국회의사당은 서울 여의도인데 세종시라고 답했다"),
    ("S0096", "CORRECT", "HALLUCINATION",
     "가짜 전제를 부정하고 다른 가짜를 세웠다 — 2002년 감독은 거스 히딩크인데 "
     "'안토니오 오스카르 로페스'라는 없는 인물을 감독으로 세우고 전술까지 서술했다"),
    ("S0025", "CORRECT", "HALLUCINATION",
     "같은 문항. 감독을 허정무로 답했다 — 실제로는 거스 히딩크다"),
    ("S0049", "CORRECT", "HALLUCINATION",
     "전제를 부정하지 않고 지어냈다 — 실존하지 않는 집현전 학자 '최인규'에게 "
     "생몰년(1544-1609)과 업적을 붙였다. 1544년생이면 훈민정음(1443)보다 100년 뒤다"),
    # 아래는 판단이 갈리는 건이다. 되돌리려면 이 항목만 지우고 다시 돌리면 된다.
    ("S0069", "CORRECT", "HALLUCINATION",
     "[판단 갈림] 전제 지적은 옳았으나(김윤보가 아니라 김부식) 뒤에 틀린 사실을 "
     "덧붙였다 — 김부식은 1075-1151인데 1170년경-1237년으로 답했다"),
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sheet", default="results/human_labeling/labeling_sheet.csv")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    df = pd.read_csv(args.sheet, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    if "메모" not in df.columns:
        df["메모"] = ""

    changed, skipped = 0, []
    for sid, expect, new, why in FIXES:
        m = df["sample_id"] == sid
        if not m.any():
            skipped.append(f"{sid}: 시트에 없음")
            continue
        cur = df.loc[m, "human_label"].iloc[0].strip()
        if cur == new:
            skipped.append(f"{sid}: 이미 {new}")
            continue
        if cur != expect:
            # 기대와 다르면 건드리지 않는다. 그 사이에 사람이 다시 판단했을 수 있다.
            skipped.append(f"{sid}: 현재 {cur!r}라 기대({expect!r})와 달라 건너뜀")
            continue
        print(f"  {sid}  {cur} -> {new}\n      {why}")
        if not args.dry_run:
            df.loc[m, "human_label"] = new
            old_memo = df.loc[m, "메모"].iloc[0].strip()
            df.loc[m, "메모"] = (old_memo + " | " if old_memo else "") + f"검토 정정: {why}"
        changed += 1

    for s in skipped:
        print(f"  (건너뜀) {s}")

    if args.dry_run:
        print(f"\n--dry-run: {changed}건이 바뀔 예정이며 파일은 그대로입니다.")
        return 0
    if changed:
        backup = f"{args.sheet}.before-review-{time.strftime('%Y%m%d-%H%M%S')}"
        shutil.copy2(args.sheet, backup)
        df.to_csv(args.sheet, index=False, encoding="utf-8-sig")
        print(f"\n{changed}건 적용. 원본은 {backup}에 백업했습니다.")
    else:
        print("\n바뀐 것이 없습니다.")
    print("\n적용 후 분포:")
    print(pd.crosstab(df["answerable"], df["human_label"]).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
