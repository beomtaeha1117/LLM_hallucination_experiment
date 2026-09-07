"""인간 검증용 층화 표본 추출기 (docs/research-design.md §7 stage 4).

results/evaluated.csv 에서 (prompt_type, question_type) 셀 층화로 n건을 뽑아
results/human_sample.csv(전체, auto_label 포함)와
results/human_sample_blind.csv(블라인드, auto_label/decided_by 제거)를 만든다.

블라인드 파일만 실제 평가자에게 준다 — 자동판정 라벨을 보여주면 Cohen's κ가
전제하는 두 판정의 독립성이 깨지기 때문이다.
"""

from __future__ import annotations

import argparse
import math
import random
from typing import Dict, List, Tuple

import pandas as pd
import yaml

EVAL_PATH_DEFAULT = "results/evaluated.csv"

# results/human_sample*.csv 에 담을 컬럼. evaluated.csv 의 'label' 컬럼을
# 'auto_label' 로 이름을 바꿔서 담는다 (evaluated.csv 실제 헤더 확인 결과,
# 스펙에서 가정한 'auto_label' 이 아니라 'label' 이었음).
SOURCE_COLUMNS = [
    "question_id",
    "question_type",
    "question_subtype",
    "answerable",
    "prompt_type",
    "model_key",
    "repeat",
    "question",
    "context",
    "ground_truth",
    "response_final",
    "label",  # -> auto_label 로 rename
    "decided_by",
]

FULL_OUTPUT_COLUMNS = [
    "sample_id",
    "question_id",
    "question_type",
    "question_subtype",
    "answerable",
    "prompt_type",
    "model_key",
    "repeat",
    "question",
    "context",
    "ground_truth",
    "response_final",
    "auto_label",
    "decided_by",
    "human_label",
    "human_note",
]

BLIND_DROP_COLUMNS = ["auto_label", "decided_by"]


def _stratified_allocate(sizes: Dict[Tuple[str, str], int], n: int) -> Dict[Tuple[str, str], int]:
    """(prompt_type, question_type) 층별 표본 크기를 정한다.

    비례 배분을 기본으로 하되, 비어있지 않은 층에는 최소 2건을 보장하고
    최대 잉여분은 최대잔여법(largest remainder method)으로 나눈다.
    각 층 배분은 그 층의 실제 행수를 넘지 못한다.
    """
    total = sum(sizes.values())
    if total == 0:
        return {}

    # 최소 2건 보장 (해당 층 크기가 2 미만이면 그 층 크기로 캡).
    alloc = {k: min(2, v) for k, v in sizes.items()}
    assigned = sum(alloc.values())

    if assigned >= n:
        # 최소보장 총합이 이미 n 이상이면, 큰 층부터 초과분을 덜어낸다.
        # (n=400, 5조건 x 15유형 정도의 표에서는 거의 발생하지 않지만 방어적으로 처리)
        excess = assigned - n
        order = sorted(alloc.keys(), key=lambda k: sizes[k], reverse=True)
        i = 0
        while excess > 0 and order:
            k = order[i % len(order)]
            if alloc[k] > 0:
                alloc[k] -= 1
                excess -= 1
            i += 1
            if i > 10000:  # 안전장치
                break
        return alloc

    remaining = n - assigned
    # 최대잔여법: 남은 몫을 반복적으로 분수 나머지가 큰 층부터 1개씩 배분.
    while remaining > 0:
        candidates = [k for k in sizes if alloc[k] < sizes[k]]
        if not candidates:
            break  # 모든 층이 이미 자기 크기만큼 배분됨 (n이 total을 넘는 경우는 상위에서 처리)
        fracs = {k: (n * sizes[k] / total) - alloc[k] for k in candidates}
        order = sorted(candidates, key=lambda k: fracs[k], reverse=True)
        progressed = False
        for k in order:
            if remaining <= 0:
                break
            if alloc[k] < sizes[k]:
                alloc[k] += 1
                remaining -= 1
                progressed = True
        if not progressed:
            break

    return alloc


def _rebalance_by_prompt_type(
    sample: pd.DataFrame,
    pool: pd.DataFrame,
    n: int,
    rng: random.Random,
) -> pd.DataFrame:
    """prompt_type(조건) 간 표본 수를 n/5 의 ±10% 이내로 재조정한다.

    비례 층화만 하면 질문유형 분포가 조건마다 달라 조건별 표본 수가
    한쪽으로 쏠릴 수 있다. docs/experiment-plan.md §1 이 경고하듯,
    이러면 조건 간 측정된 차이가 인위적으로 부풀려진다. 그래서 층화 이후
    조건별 총량을 강제로 맞춘다.
    """
    prompt_types = sorted(sample["prompt_type"].unique())
    k = len(prompt_types)
    target = n / k
    lower = target * 0.9
    upper = target * 1.1

    sample = sample.copy()
    already_selected_ids = set(zip(sample["question_id"], sample["prompt_type"], sample["model_key"], sample["repeat"]))

    for _ in range(len(prompt_types) * 3):  # 반복해서 수렴시킨다 (보통 1~2회면 충분)
        counts = sample["prompt_type"].value_counts().to_dict()
        counts = {pt: counts.get(pt, 0) for pt in prompt_types}

        over = {pt: c for pt, c in counts.items() if c > upper}
        under = {pt: c for pt, c in counts.items() if c < lower}

        if not over and not under:
            break

        # 초과 조건: 표본에서 무작위로 덜어낸다(가능하면 층이 완전히 비지 않게).
        for pt, c in over.items():
            drop_n = int(round(c - target))
            drop_n = min(drop_n, c - 1) if c > 0 else 0
            if drop_n <= 0:
                continue
            idxs = sample.index[sample["prompt_type"] == pt].tolist()
            rng.shuffle(idxs)
            to_drop = idxs[:drop_n]
            sample = sample.drop(index=to_drop)

        # 부족 조건: 아직 뽑히지 않은 같은 prompt_type 행에서 채운다.
        for pt, c in under.items():
            need_n = int(round(target - c))
            if need_n <= 0:
                continue
            candidates = pool[
                (pool["prompt_type"] == pt)
                & (~pool.set_index(["question_id", "prompt_type", "model_key", "repeat"]).index.isin(already_selected_ids))
            ]
            if candidates.empty:
                continue
            take_n = min(need_n, len(candidates))
            picked = candidates.sample(n=take_n, random_state=rng.randint(0, 2**31 - 1))
            sample = pd.concat([sample, picked], ignore_index=False)
            already_selected_ids.update(
                zip(picked["question_id"], picked["prompt_type"], picked["model_key"], picked["repeat"])
            )

    return sample.reset_index(drop=True)


def build_sample(eval_path: str, n: int, seed: int) -> pd.DataFrame:
    """evaluated.csv 로부터 층화 표본 데이터프레임을 만들어 반환한다."""
    df = pd.read_csv(eval_path, dtype=str, keep_default_na=False, encoding="utf-8-sig")

    missing = [c for c in SOURCE_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"evaluated.csv 에 예상 컬럼이 없습니다: {missing}")

    total_rows = len(df)
    rng = random.Random(seed)

    if n >= total_rows:
        print(
            f"경고: 요청한 n={n} 이 사용 가능한 전체 행수({total_rows})보다 크거나 같습니다. "
            "전체 행을 표본으로 사용합니다."
        )
        sampled = df.copy()
    else:
        groups = df.groupby(["prompt_type", "question_type"])
        sizes = {key: len(g) for key, g in groups}
        alloc = _stratified_allocate(sizes, n)

        parts = []
        for key, count in alloc.items():
            if count <= 0:
                continue
            g = groups.get_group(key)
            picked = g.sample(n=min(count, len(g)), random_state=rng.randint(0, 2**31 - 1))
            parts.append(picked)
        sampled = pd.concat(parts, ignore_index=False) if parts else df.iloc[0:0]

        # prompt_type(조건) 간 ±10% 균형 재조정.
        sampled = _rebalance_by_prompt_type(sampled, df, n, rng)

    sampled = sampled[SOURCE_COLUMNS].rename(columns={"label": "auto_label"})
    sampled = sampled.sample(frac=1.0, random_state=seed).reset_index(drop=True)  # 셀 순서 섞기

    sampled.insert(0, "sample_id", [f"S{i+1:04d}" for i in range(len(sampled))])
    sampled["human_label"] = ""
    sampled["human_note"] = ""

    return sampled[FULL_OUTPUT_COLUMNS]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.yaml", help="config.yaml 경로 (참조용)")
    parser.add_argument("--eval-path", default=EVAL_PATH_DEFAULT, help="results/evaluated.csv 경로")
    parser.add_argument("--n", type=int, default=400, help="표본 크기 (기본 400)")
    parser.add_argument("--out", default="results/human_sample.csv", help="전체본 출력 경로")
    parser.add_argument("--seed", type=int, default=42, help="랜덤 시드")
    args = parser.parse_args()

    # config.yaml 은 이 실험의 유일한 통제변인 출처이므로 존재 여부만 확인하고
    # (스펙에 정의되지 않은 표본 관련 항목이 없으면) 그대로 진행한다.
    try:
        with open(args.config, "r", encoding="utf-8") as f:
            yaml.safe_load(f)
    except FileNotFoundError:
        print(f"경고: config 파일 {args.config} 을 찾지 못했습니다. 계속 진행합니다.")

    sample = build_sample(args.eval_path, args.n, args.seed)

    full_out = args.out
    blind_out = full_out.replace(".csv", "_blind.csv")
    if blind_out == full_out:  # .csv 로 안 끝나는 경우 방어
        blind_out = full_out + "_blind.csv"

    sample.to_csv(full_out, index=False, encoding="utf-8-sig")

    # 블라인드본: auto_label/decided_by 제거 + 다시 섞기(순서로 auto_label 유추 방지).
    blind = sample.drop(columns=BLIND_DROP_COLUMNS)
    blind = blind.sample(frac=1.0, random_state=args.seed + 1).reset_index(drop=True)
    blind.to_csv(blind_out, index=False, encoding="utf-8-sig")

    print(f"\n전체본: {full_out} ({len(sample)}행)")
    print(f"블라인드본: {blind_out} ({len(blind)}행, 컬럼 제거: {BLIND_DROP_COLUMNS})")

    print("\n조건(prompt_type)별 달성 표본 수:")
    counts = sample["prompt_type"].value_counts().sort_index()
    target = len(sample) / max(len(counts), 1)
    for pt, c in counts.items():
        pct = (c - target) / target * 100 if target else 0.0
        print(f"  {pt:<6} {c:>5}  (목표 대비 {pct:+.1f}%)")


if __name__ == "__main__":
    main()
