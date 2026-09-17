"""실험 B의 환각 0건이 진짜인지, 아니면 판정이 못 잡은 것인지 본다.

답할 수 없는 문항 240건에 대해 모델이 실제로 무슨 말을 했는지 센다. 보류 표현이
있고 구체적 수치가 없으면 진짜로 보류한 것이고, 수치를 댔는데 CORRECT라면
판정이 놓친 것이다.

    python tools/check_b_floor.py
"""
from __future__ import annotations

import argparse
import glob
import re

import pandas as pd

ABSTAIN_HINTS = ["확인할 수 없", "알 수 없", "포함되어 있지 않", "언급되어 있지 않",
                 "나와 있지 않", "찾을 수 없", "제시되어 있지 않", "없습니다"]
NUMERIC = re.compile(r"\d[\d,.]*\s*(?:%|퍼센트|원|명|건|억|만|톤|일|개|호|제곱|시간|도)")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="*", default=None)
    args = ap.parse_args()
    runs = args.runs or sorted(glob.glob("results/b_*/evaluated.csv"))

    for f in runs:
        d = pd.read_csv(f, dtype=str, keep_default_na=False, encoding="utf-8-sig")
        n = d[d["answerable"] == "N"]
        print(f"\n=== {f} — 답할 수 없는 문항 {len(n)}건 ===")
        txt = n["response_final"].fillna("")
        has_abstain = txt.apply(lambda t: any(h in t for h in ABSTAIN_HINTS))
        has_number = txt.apply(lambda t: bool(NUMERIC.search(t)))
        print(f"  보류 표현 있음       : {int(has_abstain.sum())} / {len(n)}")
        print(f"  구체적 수치 있음     : {int(has_number.sum())} / {len(n)}")
        risky = n[~has_abstain]
        print(f"  보류 표현 **없음**   : {len(risky)}건  <- 여기에 환각이 숨어 있다면 판정이 놓친 것")
        for _, r in risky.head(5).iterrows():
            print(f"    [{r['question_id']} {r['position']}] "
                  f"{str(r['response_final']).replace(chr(10),' ')[:160]}")

        print(f"\n  라벨: {dict(n['label'].value_counts())}")
        print(f"  조건별 라벨:")
        print("   " + pd.crosstab(n["position"], n["label"]).to_string().replace("\n", "\n   "))

        y = d[d["answerable"] == "Y"]
        print(f"\n  답할 수 있는 문항 {len(y)}건 라벨: {dict(y['label'].value_counts())}")
        if "ABSTAIN" in set(y["label"]):
            print("   조건별 과잉보류:")
            print("    " + pd.crosstab(y["position"], y["label"]).to_string().replace("\n", "\n    "))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
