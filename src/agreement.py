"""Cohen's kappa 일치도 계산기 (docs/research-design.md §7 stage 4).

블라인드 표본(human_sample_blind.csv, 사람이 human_label 을 채워 넣은 버전)과
전체본(human_sample.csv, auto_label 을 보존한 원본)을 합쳐서
(auto_label, human_label) 간 Cohen's kappa 를 계산한다.
두 번째 평가자 파일이 주어지면 (rater1, rater2) 간 kappa 도 계산한다.
"""

from __future__ import annotations

import argparse
from typing import Dict, List

import pandas as pd

# requirements.txt 에 scikit-learn 이 없으므로 새 의존성을 넣지 않고 직접 구현한다.
LABELS = ["CORRECT", "HALLUCINATION", "ABSTAIN"]

INTERPRETATION_BANDS = [
    (float("-inf"), 0.0, "poor (< 0)"),
    (0.0, 0.20, "slight (0-0.20)"),
    (0.20, 0.40, "fair (0.21-0.40)"),
    (0.40, 0.60, "moderate (0.41-0.60)"),
    (0.60, 0.80, "substantial (0.61-0.80)"),
    (0.80, 1.01, "almost perfect (0.81-1.00)"),
]


def interpret_kappa(k: float) -> str:
    """kappa 값을 표준 해석 구간 문자열로 바꾼다."""
    for lo, hi, label in INTERPRETATION_BANDS:
        if lo < k <= hi or (lo == float("-inf") and k <= hi):
            return label
    return "unknown"


def cohen_kappa(a: List[str], b: List[str], labels: List[str]) -> float:
    """Cohen's kappa 를 직접 계산한다 (관측 일치율 - 우연 일치율) / (1 - 우연 일치율).

    scikit-learn 없이 구현: 각 라벨의 두 평가자 주변확률을 곱해 우연 일치율을 구한다.
    """
    n = len(a)
    if n == 0:
        return float("nan")

    po = sum(1 for x, y in zip(a, b) if x == y) / n

    a_counts = {lab: 0 for lab in labels}
    b_counts = {lab: 0 for lab in labels}
    for x in a:
        a_counts[x] = a_counts.get(x, 0) + 1
    for y in b:
        b_counts[y] = b_counts.get(y, 0) + 1

    pe = sum((a_counts.get(lab, 0) / n) * (b_counts.get(lab, 0) / n) for lab in labels)

    if pe == 1.0:
        return 1.0  # 완전히 우연히도 100% 일치가 기대되는 퇴화 케이스
    return (po - pe) / (1 - pe)


def confusion_matrix(a: List[str], b: List[str], labels: List[str]) -> pd.DataFrame:
    """행=a(예: auto_label), 열=b(예: human_label) 혼동행렬."""
    mat = pd.DataFrame(0, index=labels, columns=labels)
    for x, y in zip(a, b):
        if x in labels and y in labels:
            mat.loc[x, y] += 1
    return mat


def per_label_agreement(a: List[str], b: List[str], labels: List[str]) -> Dict[str, float]:
    """라벨별 일치율: 해당 라벨을 (a 또는 b)가 준 경우 중 서로 일치한 비율."""
    rates: Dict[str, float] = {}
    for lab in labels:
        relevant = [(x, y) for x, y in zip(a, b) if x == lab or y == lab]
        if not relevant:
            rates[lab] = float("nan")
            continue
        agree = sum(1 for x, y in relevant if x == y)
        rates[lab] = agree / len(relevant)
    return rates


def _report_pair(name_a: str, series_a: pd.Series, name_b: str, series_b: pd.Series) -> None:
    """두 라벨 시리즈 사이의 kappa 리포트를 출력한다."""
    a = series_a.astype(str).tolist()
    b = series_b.astype(str).tolist()
    labels = sorted(set(a) | set(b) | set(LABELS))

    print(f"\n=== {name_a} vs {name_b} (n={len(a)}) ===")

    cm = confusion_matrix(a, b, labels)
    print("혼동행렬 (행=" + name_a + ", 열=" + name_b + "):")
    print(cm.to_string())

    k = cohen_kappa(a, b, labels)
    print(f"\nCohen's kappa = {k:.4f}  [{interpret_kappa(k)}]")

    print("\n라벨별 일치율:")
    rates = per_label_agreement(a, b, labels)
    for lab, rate in rates.items():
        if rate != rate:  # NaN
            print(f"  {lab:<15} 해당 라벨 등장 없음")
        else:
            print(f"  {lab:<15} {rate:.3f}")


def load_full_with_human(full_path: str, blind_filled_path: str) -> pd.DataFrame:
    """전체본(auto_label 보존)과 사람이 채운 블라인드본을 sample_id 로 합친다."""
    full = pd.read_csv(full_path, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    blind = pd.read_csv(blind_filled_path, dtype=str, keep_default_na=False, encoding="utf-8-sig")

    if "sample_id" not in full.columns or "sample_id" not in blind.columns:
        raise ValueError("두 파일 모두 sample_id 컬럼이 있어야 합니다.")
    if "human_label" not in blind.columns:
        raise ValueError(f"{blind_filled_path} 에 human_label 컬럼이 없습니다.")

    merged = full[["sample_id", "auto_label"]].merge(
        blind[["sample_id", "human_label"]], on="sample_id", how="inner"
    )
    merged = merged[(merged["auto_label"].str.strip() != "") & (merged["human_label"].str.strip() != "")]
    return merged


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full", default="results/human_sample.csv", help="auto_label 이 보존된 전체본 경로")
    parser.add_argument("rater1", help="1번 평가자가 채운 블라인드 파일 경로 (human_label 채워짐)")
    parser.add_argument("rater2", nargs="?", default=None, help="(선택) 2번 평가자가 채운 블라인드 파일 경로")
    args = parser.parse_args()

    merged1 = load_full_with_human(args.full, args.rater1)
    _report_pair("auto_label", merged1["auto_label"], f"human_label({args.rater1})", merged1["human_label"])

    if args.rater2:
        merged2 = load_full_with_human(args.full, args.rater2)
        # 두 평가자 사이 kappa 계산을 위해 sample_id 기준으로 다시 합친다.
        full = pd.read_csv(args.full, dtype=str, keep_default_na=False, encoding="utf-8-sig")
        r1 = pd.read_csv(args.rater1, dtype=str, keep_default_na=False, encoding="utf-8-sig")[["sample_id", "human_label"]].rename(
            columns={"human_label": "human_label_1"}
        )
        r2 = pd.read_csv(args.rater2, dtype=str, keep_default_na=False, encoding="utf-8-sig")[["sample_id", "human_label"]].rename(
            columns={"human_label": "human_label_2"}
        )
        both = r1.merge(r2, on="sample_id", how="inner")
        both = both[
            (both["human_label_1"].str.strip() != "") & (both["human_label_2"].str.strip() != "")
        ]
        _report_pair("rater1", both["human_label_1"], "rater2", both["human_label_2"])


if __name__ == "__main__":
    main()
