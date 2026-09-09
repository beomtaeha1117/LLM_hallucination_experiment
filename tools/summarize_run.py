"""완료된 실행의 조건별 수치를 raw_responses.csv에서 직접 낸다.

실행을 다시 돌리지 않고 보고 싶을 때 쓴다.

    python tools/summarize_run.py results/main_qwen35
"""
from __future__ import annotations

import os
import sys

import pandas as pd


def main(run_dir: str) -> None:
    raw = os.path.join(run_dir, "raw_responses.csv")
    df = pd.read_csv(raw, dtype=str, encoding="utf-8-sig")
    df["ok"] = df["format_ok"].astype(str).str.lower().isin(["true", "1"])
    df["trunc"] = df["finish_reason"].astype(str) == "length"
    ct = pd.to_numeric(df["completion_tokens"], errors="coerce")

    print(f"{raw}  —  총 {len(df):,}행\n")
    print(f"{'조건':<6}{'n':>7}{'format_ok':>11}{'잘림':>7}{'토큰중앙값':>11}  parse_mode")
    for c in df["prompt_type"].drop_duplicates():
        g = df[df["prompt_type"] == c]
        modes = g["parse_mode"].value_counts().to_dict()
        modes_s = " ".join(f"{k}={v}" for k, v in modes.items())
        med = ct[g.index].median()
        print(f"{c:<6}{len(g):>7}{g['ok'].mean():>11.3f}{int(g['trunc'].sum()):>7}{med:>11.0f}  {modes_s}")

    print(f"\n전체 format_ok {df['ok'].mean():.3f}, 잘림 {int(df['trunc'].sum())}건 "
          f"({df['trunc'].mean() * 100:.2f}%)")

    # 계획한 응답 수를 못 채운 칸이 있는지 — 실패로 빠진 자리를 눈에 보이게 한다.
    # 모델 수를 빼먹으면 음수가 나온다(mock은 모델 3종이라 조건당 27x3x3=243이다).
    expected = (
        df["model_key"].nunique()
        * df["question_id"].nunique()
        * df["repeat"].nunique()
    )
    short = {c: expected - len(df[df["prompt_type"] == c]) for c in df["prompt_type"].unique()}
    missing = {c: n for c, n in short.items() if n}
    if missing:
        print(f"\n!! 계획({expected}건)보다 모자란 조건: " +
              ", ".join(f"{c} {n}건" for c, n in missing.items()))
    failed = os.path.join(run_dir, "failed_responses.csv")
    if os.path.exists(failed):
        f = pd.read_csv(failed, dtype=str)
        print(f"\n실패 기록 {len(f)}건 ({failed}):")
        for _, r in f.iterrows():
            print(f"  {r['prompt_type']:<4} {r['question_id']:<8} repeat={r['repeat']}  {str(r['error'])[:90]}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "results/main_qwen35")
