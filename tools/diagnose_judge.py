"""judge가 사람과 어긋난 자리를 하나씩 들여다본다.

kappa 하나로는 judge를 바꿀지 프롬프트를 고칠지 정할 수 없다. 2026-09-17에
devstral의 kappa가 0.371로 나왔는데, 사람이 환각이라 본 22건 중 judge가 잡은
것은 7건뿐이었다. 그 15건이 **어떤 종류로** 어긋났는지에 따라 할 일이 다르다.

특히 확인해야 할 것: evaluate.py는 answerable=N 행에서 judge가 ABSTAIN을 내면
abstain_with_claim이 참이어도 CORRECT로 확정한다. "확인할 수 없습니다. 다만
…였습니다"처럼 보류하고 지어낸 응답이 여기로 샌다. 15건이 대부분 이 경우라면
judge를 바꿀 일이 아니라 라벨 규칙을 고칠 일이다.

    python tools/diagnose_judge.py
"""
from __future__ import annotations

import argparse
import glob
import os

import pandas as pd


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sheet", default="results/human_labeling/labeling_sheet.csv")
    ap.add_argument("--judge", default=None)
    ap.add_argument("--show", type=int, default=6, help="응답 본문을 보여줄 건수")
    args = ap.parse_args()

    jf = args.judge
    if not jf:
        c = sorted(glob.glob(os.path.join(os.path.dirname(args.sheet), "judge_*.csv")))
        if not c:
            raise SystemExit("judge_*.csv를 찾지 못했습니다. --judge로 지정하십시오.")
        jf = c[0]

    h = pd.read_csv(args.sheet, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    j = pd.read_csv(jf, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    d = h.merge(j, on="sample_id", how="inner")
    print(f"{args.sheet}\n{jf}\n합쳐진 행 {len(d)}\n")

    d["agree"] = d["auto_label"] == d["human_label"]
    print("전체 일치:", f"{d['agree'].mean():.3f}", f"({d['agree'].sum()}/{len(d)})\n")

    miss = d[(d["auto_label"] == "CORRECT") & (d["human_label"] == "HALLUCINATION")]
    print(f"=== judge는 CORRECT, 사람은 HALLUCINATION : {len(miss)}건 ===")
    print("   (judge가 환각을 놓친 자리. 이 연구에서 가장 비싼 오류다)\n")

    if len(miss):
        # 핵심 질문: 이 15건에서 judge의 원본 라벨과 abstain_with_claim은 무엇이었나
        if "judge_label_raw" in miss.columns:
            print("judge 원본 라벨 (매핑 전):")
            print("  " + miss["judge_label_raw"].value_counts().to_string().replace("\n", "\n  "))
        if "abstain_with_claim" in miss.columns:
            awc = miss["abstain_with_claim"].str.strip().str.lower().isin(["true", "1"])
            print(f"\nabstain_with_claim=True : {int(awc.sum())} / {len(miss)}건")
            print("  -> 이 값이 크면 judge는 '보류하고 지어냈다'를 알아봤는데")
            print("     evaluate.py가 그 플래그를 라벨에 반영하지 않아 샌 것이다.")
            print("     그 경우 고칠 것은 judge가 아니라 라벨 규칙이다.")
        print("\n유형별:")
        print("  " + miss["question_type"].value_counts().to_string().replace("\n", "\n  "))

        print(f"\n--- 놓친 응답 {min(args.show, len(miss))}건 ---")
        for _, r in miss.head(args.show).iterrows():
            awc = r.get("abstain_with_claim", "")
            print(f"\n[{r['sample_id']}] {r['question_type']} / answerable={r['answerable']}"
                  f" / judge원본={r.get('judge_label_raw','?')} / awc={awc}")
            print(f"  질문: {str(r['question'])[:80]}")
            resp = str(r["response_final"]).replace("\n", " ")
            print(f"  응답: {resp[:300]}{'...' if len(resp) > 300 else ''}")
            if str(r.get("reason", "")).strip():
                print(f"  judge 근거: {str(r['reason'])[:160]}")

    over = d[(d["auto_label"] == "HALLUCINATION") & (d["human_label"] == "CORRECT")]
    print(f"\n\n=== judge는 HALLUCINATION, 사람은 CORRECT : {len(over)}건 ===")
    for _, r in over.iterrows():
        print(f"  [{r['sample_id']}] {r['question_type']}: "
              f"{str(r['response_final']).replace(chr(10),' ')[:140]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
