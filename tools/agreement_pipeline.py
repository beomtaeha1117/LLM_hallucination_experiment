"""사람 라벨을 **파이프라인이 실제로 붙인 라벨**과 대조한다.

judge 단독 성능이 아니라 데이터에 실제로 들어간 라벨의 신뢰도를 재야 한다.
이유가 둘이다.

1. 파이프라인은 judge만 쓰지 않는다. 본실험에서 절반 가까이가 규칙 매칭(`match`)
   으로 끝나고 judge까지 가지 않는다. judge에 100건을 전부 밀어 넣고 잰 kappa는
   실제로 저장된 라벨의 신뢰도가 아니다(한라산 1,950m가 judge에서는 환각으로,
   파이프라인에서는 수치 허용오차 안이라 CORRECT로 갈린다).
2. 라벨링 시트에는 `why_unanswerable`과 `acceptable_answers`가 없다. 사람이 읽을
   것만 담았기 때문인데, judge는 그 두 칸을 보고 판정한다. 시트만 넘기면 judge가
   눈을 가린 채 판정한 셈이 되어 성능이 실제보다 낮게 나온다.

`labeling_key.csv`가 sample_id와 원래 행을 이어주므로 GPU 없이 계산된다.

    python tools/agreement_pipeline.py
"""
from __future__ import annotations

import argparse
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.agreement import cohen_kappa, interpret_kappa, LABELS  # noqa: E402

JOIN = ["run_id", "model_key", "prompt_type", "question_id", "repeat"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="results/human_labeling")
    ap.add_argument("--results", default="results")
    args = ap.parse_args()

    sheet = pd.read_csv(f"{args.dir}/labeling_sheet.csv", dtype=str,
                        keep_default_na=False, encoding="utf-8-sig")
    key = pd.read_csv(f"{args.dir}/labeling_key.csv", dtype=str,
                      keep_default_na=False, encoding="utf-8-sig")
    h = sheet[["sample_id", "human_label", "question_type", "answerable"]].merge(
        key, on="sample_id", how="inner")

    # 표본이 어느 run에서 왔든 그 run의 evaluated.csv에서 실제 라벨을 가져온다
    frames = []
    for run_id in sorted(h["run_id"].unique()):
        p = os.path.join(args.results, run_id, "evaluated.csv")
        if not os.path.exists(p):
            print(f"!! {p} 가 없습니다. 판정(evaluate)을 먼저 돌리십시오.")
            return 1
        e = pd.read_csv(p, dtype=str, keep_default_na=False, encoding="utf-8-sig")
        frames.append(e[JOIN + ["label", "decided_by", "abstain_with_claim"]])
    ev = pd.concat(frames, ignore_index=True)

    d = h.merge(ev, on=JOIN, how="left")
    missing = d["label"].isna() | (d["label"] == "")
    if missing.any():
        print(f"!! evaluated.csv에서 찾지 못한 표본 {int(missing.sum())}건 — 제외합니다")
        d = d[~missing]

    print(f"대조 {len(d)}건\n")
    print("파이프라인이 무엇으로 정했나:")
    print("  " + d["decided_by"].value_counts().to_string().replace("\n", "\n  "))

    a, b = d["label"].tolist(), d["human_label"].tolist()
    print(f"\n혼동행렬 (행=파이프라인, 열=사람):")
    print(pd.crosstab(d["label"], d["human_label"]).reindex(
        index=LABELS, columns=LABELS, fill_value=0).to_string())
    k = cohen_kappa(a, b, LABELS)
    print(f"\n전체 일치 {sum(x == y for x, y in zip(a, b)) / len(a):.3f}")
    print(f"Cohen's kappa = {k:.4f}  [{interpret_kappa(k)}]")

    # 이 연구에서 가장 비싼 칸: 사람이 환각이라 본 것을 파이프라인이 놓친 자리
    miss = d[(d["human_label"] == "HALLUCINATION") & (d["label"] != "HALLUCINATION")]
    tot_h = int((d["human_label"] == "HALLUCINATION").sum())
    print(f"\n환각 재현율: {tot_h - len(miss)}/{tot_h} "
          f"({(tot_h - len(miss)) / tot_h:.3f}) — 사람이 환각이라 본 것 중 파이프라인이 잡은 비율")
    if len(miss):
        awc = miss["abstain_with_claim"].str.strip().str.lower().isin(["true", "1"])
        print(f"  놓친 {len(miss)}건 중 abstain_with_claim=True : {int(awc.sum())}건")
        print("  -> 이 값이 크면 judge는 알아봤고 라벨 규칙이 버린 것이다.")
        print("     그 경우 재판정 없이 라벨만 다시 계산하면 된다.")
        print("\n  놓친 표본:", ", ".join(miss["sample_id"].tolist()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
