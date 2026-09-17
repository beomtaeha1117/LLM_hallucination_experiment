"""사람 라벨을 저장소에 백업할 수 있는 최소 형태로 뽑는다.

사람 라벨은 이 연구에서 **유일하게 다시 만들 수 없는 자산**이다. 응답은 14시간이면
재생성되고 판정은 7시간이면 다시 붙지만, 사람이 100건을 매긴 것은 복구가 없다.
그런데 `results/`가 .gitignore에 걸려 있어 그 파일은 기계 한 대에만 있다.

`results/`를 여는 것은 답이 아니다 — 저장소가 공개라 응답 전문이 올라간다.
대신 **라벨과 그 라벨이 어느 행의 것인지**만 뽑는다. 응답도 질문도 들어가지 않으므로
공개돼도 문제가 없고, 원본이 날아가도 `results/`를 다시 만든 뒤 sample_id로 이어붙이면
라벨이 그대로 복원된다.

출력은 `data/human_labels/` — 이미 추적되는 자리라 .gitignore를 고칠 필요가 없다.

    python tools/export_human_labels.py
"""
from __future__ import annotations

import argparse
import os

import pandas as pd

LABEL_COLS = ["sample_id", "human_label", "human_rater_id", "메모"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="results/human_labeling")
    ap.add_argument("--out", default="data/human_labels")
    args = ap.parse_args()

    sheet_p = os.path.join(args.dir, "labeling_sheet.csv")
    key_p = os.path.join(args.dir, "labeling_key.csv")
    if not os.path.exists(sheet_p):
        print(f"!! {sheet_p} 가 없습니다.")
        return 1

    os.makedirs(args.out, exist_ok=True)
    sheet = pd.read_csv(sheet_p, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    labels = sheet[[c for c in LABEL_COLS if c in sheet.columns]].copy()
    n = int((labels["human_label"].str.strip() != "").sum())
    labels.to_csv(f"{args.out}/labels.csv", index=False, encoding="utf-8-sig")
    print(f"{args.out}/labels.csv  ({len(labels)}행, 채워진 라벨 {n}건)")

    if os.path.exists(key_p):
        # 라벨만 있으면 어느 행의 것인지 알 수 없다. 이 대응표가 있어야 복원된다.
        # 모델·조건·문항 id는 연구 메타데이터이지 응답 내용이 아니다.
        key = pd.read_csv(key_p, dtype=str, keep_default_na=False, encoding="utf-8-sig")
        key.to_csv(f"{args.out}/key.csv", index=False, encoding="utf-8-sig")
        print(f"{args.out}/key.csv      ({len(key)}행) — sample_id가 어느 응답인지")

    # 들어가면 안 되는 것이 섞이지 않았는지 확인한다
    leaked = [c for c in ("response_final", "question", "context", "ground_truth")
              if c in labels.columns]
    if leaked:
        print(f"!! 응답/문항 내용이 섞였습니다: {leaked}")
        return 1
    print("\n응답 본문과 문항 내용은 들어 있지 않습니다. 공개 저장소에 올려도 됩니다.")
    print("\n복원하려면:")
    print("  results/human_labeling/labeling_sheet.csv 를 다시 만든 뒤")
    print("  data/human_labels/labels.csv 를 sample_id로 병합하면 라벨이 돌아옵니다.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
