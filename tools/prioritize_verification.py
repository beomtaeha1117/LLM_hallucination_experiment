"""미검증 문항 중 **정답이 틀렸을 때 실제로 피해가 나는 것**부터 줄 세운다.

87문항을 전부 검증할 시간이 없을 때 쓴다. 전제는 이렇다.

정답(ground_truth)이 틀리면 모델의 맞는 답이 HALLUCINATION으로 찍힌다. 그러므로
위험한 문항은 **환각으로 찍힌 응답이 많은 문항**이다. 반대로 세 모델 아홉 응답이
전부 CORRECT로 나온 문항은 정답이 아홉 번 독립적으로 모델과 일치한 것이라
틀렸을 가능성이 낮다 — 증명은 아니지만, 한정된 시간을 어디에 쓸지 정하는
근거로는 충분하다.

`decided_by`도 함께 본다. `match`로 끝난 행은 정답 문자열이 응답에 그대로
들어 있었다는 뜻이라 특히 안전하다.

    python tools/prioritize_verification.py
"""
from __future__ import annotations

import argparse
import glob

import pandas as pd


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions", default="data/questions_v1_DRAFT.csv")
    ap.add_argument("--top", type=int, default=25)
    args = ap.parse_args()

    q = pd.read_csv(args.questions, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    unver = q[q["verified"].str.strip().str.upper() != "YES"]
    print(f"미검증 문항 {len(unver)}개 / 전체 {len(q)}개\n")

    files = sorted(glob.glob("results/main_*/evaluated.csv"))
    if not files:
        print("!! results/main_*/evaluated.csv 가 없습니다.")
        return 1
    ev = pd.concat([pd.read_csv(f, dtype=str, keep_default_na=False, encoding="utf-8-sig")
                    [["question_id", "model_key", "label", "decided_by"]] for f in files],
                   ignore_index=True)

    ev = ev[ev["question_id"].isin(set(unver["question_id"]))]
    g = ev.groupby("question_id").agg(
        n=("label", "size"),
        hall=("label", lambda s: int((s == "HALLUCINATION").sum())),
        match=("decided_by", lambda s: int((s == "match").sum())),
    ).reset_index()
    g["hall_rate"] = g["hall"] / g["n"]
    d = unver.merge(g, on="question_id", how="left").fillna({"n": 0, "hall": 0, "match": 0, "hall_rate": 0})

    clean = d[(d["hall"] == 0) & (d["match"] > 0)]
    risky = d[d["hall"] > 0].sort_values("hall", ascending=False)
    quiet = d[(d["hall"] == 0) & (d["match"] == 0)]

    print("=" * 66)
    print(f"[1] 검증해야 할 것 — 환각이 찍힌 미검증 문항 : {len(risky)}개")
    print("    정답이 틀렸다면 여기서 맞는 답이 환각으로 잘못 찍혔을 수 있다.")
    print("=" * 66)
    for _, r in risky.head(args.top).iterrows():
        print(f"\n  {r['question_id']}  [{r['question_type']}]  "
              f"환각 {int(r['hall'])}/{int(r['n'])}  (match {int(r['match'])})")
        print(f"    문항: {str(r['question'])[:70]}")
        print(f"    정답: {str(r['ground_truth'])[:70]}")
    if len(risky) > args.top:
        print(f"\n  ... 외 {len(risky) - args.top}개")

    print("\n" + "=" * 66)
    print(f"[2] 나중에 봐도 되는 것 — 환각 0건, 정답 문자열이 응답에 직접 일치 : {len(clean)}개")
    print("    아홉 응답이 정답과 맞았다. 틀렸을 가능성이 낮다.")
    print("=" * 66)
    print("  " + ", ".join(clean["question_id"].tolist()))

    if len(quiet):
        print("\n" + "=" * 66)
        print(f"[3] 애매한 것 — 환각 0건이지만 문자열 일치도 없음 : {len(quiet)}개")
        print("    judge가 의미로 맞다고 본 것이다. [1] 다음 순서.")
        print("=" * 66)
        print("  " + ", ".join(quiet["question_id"].tolist()))

    print(f"\n\n요약: {len(d)}개 중 **{len(risky)}개**만 먼저 보면 된다.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
