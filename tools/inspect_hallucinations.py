"""환각으로 찍힌 응답이 **진짜 오답인지 채점이 엄격한 것인지** 눈으로 가른다.

미검증 문항을 줄 세워 보니 정답 자체는 대부분 명백히 맞았다(몰, 근정전, 1443년…).
그러면 환각 딱지의 출처는 셋 중 하나다.

  (a) 모델이 진짜 틀렸다        -> 연구가 재려는 것. 그대로 둔다.
  (b) 반올림/표기가 다를 뿐이다 -> 채점 문제. acceptable_answers나 tolerance를 손본다.
                                   ("약 1,950미터" vs 정답 "약 1,947m")
  (c) 정답이 틀렸다             -> 문항을 고친다.

셋은 응답을 읽어야 갈린다. 서로 다른 오답을 묶어 보여주므로, 같은 오답이 여러 번
반복되면 (b)일 가능성이 높다.

    python tools/inspect_hallucinations.py --top 8
"""
from __future__ import annotations

import argparse
import glob
from collections import Counter

import pandas as pd


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions", default="data/questions_v1_DRAFT.csv")
    ap.add_argument("--top", type=int, default=8, help="환각이 많은 문항 몇 개를 볼 것인가")
    ap.add_argument("--variants", type=int, default=5, help="문항당 서로 다른 오답 몇 개")
    ap.add_argument("--qid", default=None, help="특정 문항만 본다")
    args = ap.parse_args()

    q = pd.read_csv(args.questions, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    ev = pd.concat([pd.read_csv(f, dtype=str, keep_default_na=False, encoding="utf-8-sig")
                    for f in sorted(glob.glob("results/main_*/evaluated.csv"))], ignore_index=True)
    h = ev[ev["label"] == "HALLUCINATION"]

    if args.qid:
        targets = [args.qid]
    else:
        unver = set(q[q["verified"].str.strip().str.upper() != "YES"]["question_id"])
        targets = [qid for qid, _ in Counter(h[h["question_id"].isin(unver)]["question_id"])
                   .most_common(args.top)]

    for qid in targets:
        row = q[q["question_id"] == qid]
        if row.empty:
            print(f"\n!! {qid} 문항을 찾지 못했습니다")
            continue
        r = row.iloc[0]
        sub = h[h["question_id"] == qid]
        print("\n" + "=" * 70)
        print(f"{qid}  [{r['question_type']}]  환각 {len(sub)}건")
        print(f"  문항: {r['question']}")
        print(f"  정답: {r['ground_truth']!r}")
        if str(r.get("acceptable_answers", "")).strip():
            print(f"  허용 표현: {r['acceptable_answers']!r}")
        else:
            print("  허용 표현: (없음)  <- 표기가 조금만 달라도 judge로 넘어간다")
        print("=" * 70)
        # 서로 다른 오답을 묶는다. 같은 답이 반복되면 채점 문제일 가능성이 높다.
        texts = sub["response_final"].fillna("").apply(lambda t: " ".join(str(t).split())[:150])
        for txt, n in Counter(texts).most_common(args.variants):
            models = ", ".join(sorted(set(sub[texts == txt]["model_key"])))
            print(f"\n  [{n}건 / {models}]")
            print(f"    {txt}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
