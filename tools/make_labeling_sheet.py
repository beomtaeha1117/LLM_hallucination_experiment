"""사람이 라벨링할 표본을 뽑는다 — judge를 κ로 고르기 위한 정답지다.

judge 후보가 devstral 하나뿐이고 코딩 특화라 한국어 QA 채점에 맞는지 모른다.
맞는지 아닌지는 사람이 매긴 라벨과 대조해야만 알 수 있다(docs/RUNBOOK.md §3-2).

두 파일을 만든다.
  labeling_sheet.csv   사람이 채울 것. **조건과 모델이 안 보인다.**
  labeling_key.csv     대조용 열쇠. 라벨링이 끝나기 전에는 열지 말 것.

조건을 가리는 이유: 어떤 프롬프트로 나온 답인지 알면 "검증 프롬프트니까 맞겠지"
같은 편향이 들어간다. 그건 이 연구가 재려는 것과 정확히 같은 축이라 치명적이다.

    python tools/make_labeling_sheet.py --n 100
"""
from __future__ import annotations

import argparse
import glob
import os

import pandas as pd

LABEL_HELP = "CORRECT / HALLUCINATION / ABSTAIN 중 하나를 적으십시오"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100, help="뽑을 응답 수 (기본 100)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", default="results/human_labeling")
    ap.add_argument("--runs", nargs="*", default=None,
                    help="대상 run 디렉터리. 기본은 results/main_* 전부")
    args = ap.parse_args()

    run_dirs = args.runs or sorted(glob.glob("results/main_*"))
    frames = []
    for d in run_dirs:
        f = os.path.join(d, "raw_responses.csv")
        if os.path.exists(f):
            frames.append(pd.read_csv(f, dtype=str, encoding="utf-8-sig"))
    if not frames:
        raise SystemExit(f"raw_responses.csv를 찾지 못했습니다: {run_dirs}")
    df = pd.concat(frames, ignore_index=True)
    print(f"대상 {len(run_dirs)}개 실행, 총 {len(df):,}행")

    # 반복 3회 중 하나만 쓴다. 같은 문항의 거의 같은 답을 세 번 매기는 것은
    # 사람 시간 낭비이고 κ를 부풀린다(독립된 판정이 아니기 때문이다).
    df = df[df["repeat"] == "0"]

    # answerable Y/N과 question_type이 골고루 들어가야 한다. 한쪽으로 쏠리면
    # κ가 그 부분집합에서만 유효해진다.
    strata = df["question_type"].astype(str) + "|" + df["answerable"].astype(str)
    df = df.assign(_stratum=strata)
    per = max(1, args.n // df["_stratum"].nunique())
    picked = (
        df.groupby("_stratum", group_keys=False)
        .apply(lambda g: g.sample(min(len(g), per), random_state=args.seed))
    )
    # 층별로 채우고 남으면 무작위로 보충한다
    if len(picked) < args.n:
        rest = df.drop(picked.index)
        picked = pd.concat([picked, rest.sample(min(len(rest), args.n - len(picked)),
                                                random_state=args.seed)])
    picked = picked.sample(frac=1, random_state=args.seed).reset_index(drop=True)
    picked.insert(0, "sample_id", [f"S{i:04d}" for i in range(1, len(picked) + 1)])

    os.makedirs(args.out_dir, exist_ok=True)

    sheet = picked[["sample_id", "question_type", "answerable", "question",
                    "context", "ground_truth", "response_final"]].copy()
    sheet["human_label"] = ""          # 여기를 채운다
    sheet["human_rater_id"] = ""
    sheet["메모"] = ""
    sheet_path = os.path.join(args.out_dir, "labeling_sheet.csv")
    sheet.to_csv(sheet_path, index=False, encoding="utf-8-sig")

    key = picked[["sample_id", "run_id", "model_key", "prompt_type", "question_id",
                  "repeat", "parse_mode", "format_ok"]]
    key_path = os.path.join(args.out_dir, "labeling_key.csv")
    key.to_csv(key_path, index=False, encoding="utf-8-sig")

    print(f"\n{sheet_path}  ({len(sheet)}건) <- 이걸 채우십시오")
    print(f"{key_path}                  <- 라벨링이 끝나기 전에는 열지 마십시오")
    print(f"\n{LABEL_HELP}")
    print("  CORRECT       정답이거나, 답할 수 없는 문항에 올바로 보류/전제 지적한 것")
    print("  HALLUCINATION 사실과 다른 내용을 단정한 것")
    print("  ABSTAIN       답할 수 있는 문항인데 보류한 것 (과잉보류)")
    print("\n층 분포:")
    print(picked.groupby(["answerable"])["sample_id"].count().to_string())
    print(picked["question_type"].value_counts().to_string())


if __name__ == "__main__":
    main()
