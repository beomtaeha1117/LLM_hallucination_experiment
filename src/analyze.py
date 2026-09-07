"""요약 지표 + 통계 검정 + 그래프 생성 스테이지."""

from __future__ import annotations

import argparse
import logging
import warnings
from typing import Dict, List, Tuple

import matplotlib
import numpy as np
import pandas as pd
import yaml

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402

logger = logging.getLogger("analyze")
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

# 그래프·표에서의 조건 표시 순서 선호도. 여기 없는 조건도 반드시 살아남아야 한다.
# 이 목록을 고정 목록으로 쓰면 새 조건(P4, P5 …)이 Categorical 변환에서 NaN이 되어
# 결과에서 조용히 사라진다. 실제로 그 버그가 있었다.
_CONDITIONS_PREFERRED = ["P0", "P1", "P2", "P3", "P2L", "P4", "P5"]


def _order_conditions(present) -> list:
    """실제 데이터에 존재하는 조건을 선호 순서대로 정렬하고, 미지의 조건은 뒤에 붙인다."""
    present = list(dict.fromkeys(present))
    known = [c for c in _CONDITIONS_PREFERRED if c in present]
    unknown = sorted(c for c in present if c not in _CONDITIONS_PREFERRED)
    return known + unknown
_FACTORIAL_CONDITIONS = ["P0", "P1", "P2", "P3"]  # P2L 제외 (2x2 요인설계용)


# ---------------------------------------------------------------------------
# 한글 폰트
# ---------------------------------------------------------------------------

def _setup_korean_font() -> None:
    candidates = ["AppleGothic", "NanumGothic"]
    available = {f.name for f in font_manager.fontManager.ttflist}
    for name in candidates:
        if name in available:
            plt.rcParams["font.family"] = name
            plt.rcParams["axes.unicode_minus"] = False
            return
    logger.warning("한글 폰트(AppleGothic/NanumGothic)를 찾지 못했습니다. 기본 폰트로 대체합니다 — 그래프의 한글이 깨질 수 있습니다.")
    plt.rcParams["axes.unicode_minus"] = False


# ---------------------------------------------------------------------------
# Wilson CI
# ---------------------------------------------------------------------------

def _wilson_ci(count: int, nobs: int, alpha: float = 0.05) -> Tuple[float, float]:
    if nobs == 0:
        return (float("nan"), float("nan"))
    from statsmodels.stats.proportion import proportion_confint

    low, high = proportion_confint(count, nobs, alpha=alpha, method="wilson")
    return float(low), float(high)


def _rate_with_ci(count: int, nobs: int, alpha: float) -> Dict[str, float]:
    rate = count / nobs if nobs else float("nan")
    low, high = _wilson_ci(count, nobs, alpha)
    return {"rate": rate, "ci_low": low, "ci_high": high}


# ---------------------------------------------------------------------------
# 요약 지표
# ---------------------------------------------------------------------------

def _predicted_abstain(row: pd.Series) -> bool:
    """지표 계산용: '이 응답이 보류였는가'를 answerable에 따라 통일적으로 정의.

    N 문항은 별도 ABSTAIN 라벨이 없고 CORRECT가 곧 보류/전제지적이므로,
    N행에서는 label==CORRECT를 '보류'로 취급한다.
    """
    if row["answerable"] == "Y":
        return row["label"] == "ABSTAIN"
    return row["label"] == "CORRECT"


def compute_summary(df: pd.DataFrame, alpha: float) -> pd.DataFrame:
    df = df.copy()
    df["completion_tokens"] = pd.to_numeric(df["completion_tokens"], errors="coerce")
    df["format_ok_bool"] = df["format_ok"].astype(str).str.lower().isin(["true", "1"])
    df["predicted_abstain"] = df.apply(_predicted_abstain, axis=1)

    rows = []
    for (model_key, prompt_type), g in df.groupby(["model_key", "prompt_type"]):
        n = len(g)
        y = g[g["answerable"] == "Y"]
        n_g = g[g["answerable"] == "N"]

        halluc = _rate_with_ci(int((g["label"] == "HALLUCINATION").sum()), n, alpha)
        acc = _rate_with_ci(int((y["label"] == "CORRECT").sum()), len(y), alpha)
        correct_abstain = _rate_with_ci(int((n_g["label"] == "CORRECT").sum()), len(n_g), alpha)
        over_abstain = _rate_with_ci(int((y["label"] == "ABSTAIN").sum()), len(y), alpha)

        # answer_rate: 실제로 내용을 답한 비율. Y는 ABSTAIN이 아니면 실답변,
        # N은 HALLUCINATION(지어냄)이어야 실답변이다 (CORRECT는 보류/전제지적이므로 실답변이 아님).
        real_answer = int((y["label"] != "ABSTAIN").sum()) + int((n_g["label"] == "HALLUCINATION").sum())
        answer = _rate_with_ci(real_answer, n, alpha)

        # abstention precision/recall/F1: positive = unanswerable(N)
        pred_abstain = g["predicted_abstain"]
        actual_n = g["answerable"] == "N"
        tp = int((pred_abstain & actual_n).sum())
        fp = int((pred_abstain & ~actual_n).sum())
        fn = int((~pred_abstain & actual_n).sum())
        precision = _rate_with_ci(tp, tp + fp, alpha)
        recall = _rate_with_ci(tp, tp + fn, alpha)
        f1 = (2 * precision["rate"] * recall["rate"] / (precision["rate"] + recall["rate"])) if (
            precision["rate"] + recall["rate"] > 0
        ) else float("nan")

        format_ok = _rate_with_ci(int(g["format_ok_bool"].sum()), n, alpha)

        rows.append(
            {
                "model_key": model_key,
                "prompt_type": prompt_type,
                "n": n,
                "hallucination_rate": halluc["rate"],
                "hallucination_rate_ci_low": halluc["ci_low"],
                "hallucination_rate_ci_high": halluc["ci_high"],
                "accuracy_at_answerable": acc["rate"],
                "accuracy_at_answerable_ci_low": acc["ci_low"],
                "accuracy_at_answerable_ci_high": acc["ci_high"],
                "correct_abstention_rate": correct_abstain["rate"],
                "correct_abstention_rate_ci_low": correct_abstain["ci_low"],
                "correct_abstention_rate_ci_high": correct_abstain["ci_high"],
                "over_abstention_rate": over_abstain["rate"],
                "over_abstention_rate_ci_low": over_abstain["ci_low"],
                "over_abstention_rate_ci_high": over_abstain["ci_high"],
                "answer_rate": answer["rate"],
                "answer_rate_ci_low": answer["ci_low"],
                "answer_rate_ci_high": answer["ci_high"],
                "abstention_precision": precision["rate"],
                "abstention_precision_ci_low": precision["ci_low"],
                "abstention_precision_ci_high": precision["ci_high"],
                "abstention_recall": recall["rate"],
                "abstention_recall_ci_low": recall["ci_low"],
                "abstention_recall_ci_high": recall["ci_high"],
                "abstention_f1": f1,
                "mean_completion_tokens": g["completion_tokens"].mean(),
                "format_ok_rate": format_ok["rate"],
                "format_ok_rate_ci_low": format_ok["ci_low"],
                "format_ok_rate_ci_high": format_ok["ci_high"],
            }
        )

    summary = pd.DataFrame(rows)
    cat_type = pd.CategoricalDtype(
        categories=_order_conditions(summary["prompt_type"].unique()), ordered=True
    )
    summary["prompt_type"] = summary["prompt_type"].astype(cat_type)
    summary = summary.sort_values(["model_key", "prompt_type"]).reset_index(drop=True)
    summary["prompt_type"] = summary["prompt_type"].astype(str)
    return summary


# ---------------------------------------------------------------------------
# 통계 검정
# ---------------------------------------------------------------------------

def _majority_vote_indicator(sub: pd.DataFrame) -> int:
    """한 (question, condition) 셀의 여러 repeat 응답을 다수결로 0/1 환각 지표로 집계."""
    vals = (sub["label"] == "HALLUCINATION").astype(int)
    return int(vals.mean() >= 0.5)


def _run_gee(df: pd.DataFrame, formula_cols: List[str], out_lines: List[str], title: str) -> None:
    out_lines.append(f"--- {title} ---")
    try:
        import statsmodels.api as sm
        import statsmodels.formula.api as smf

        work = df.copy()
        work["hallucination"] = (work["label"] == "HALLUCINATION").astype(int)
        formula = "hallucination ~ " + " * ".join(["uncertainty", "verification"])
        if "completion_tokens_z" in formula_cols:
            formula += " + completion_tokens_z"

        model = smf.gee(
            formula,
            groups="question_id",
            data=work,
            family=sm.families.Binomial(),
            cov_struct=sm.cov_struct.Exchangeable(),
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = model.fit()

        out_lines.append(result.summary().as_text())
        out_lines.append("")
        out_lines.append("오즈비 (OR)와 95% CI:")
        conf = result.conf_int()
        for name in result.params.index:
            coef = result.params[name]
            or_val = float(np.exp(coef))
            ci_low, ci_high = float(np.exp(conf.loc[name, 0])), float(np.exp(conf.loc[name, 1]))
            p_val = float(result.pvalues[name])
            out_lines.append(f"  {name}: OR={or_val:.3f} (95% CI {ci_low:.3f}-{ci_high:.3f}), p={p_val:.4f}")
    except Exception as exc:  # noqa: BLE001 - 통계 실패가 파이프라인을 깨면 안 됨
        out_lines.append(f"[오류] GEE 적합 실패: {exc!r}")
    out_lines.append("")


def _cochran_q(mat: np.ndarray) -> Tuple[float, float, int]:
    """Cochran's Q. mat: rows=subjects, cols=treatments, values in {0,1}."""
    n, k = mat.shape
    col_sums = mat.sum(axis=0)
    row_sums = mat.sum(axis=1)
    numerator = k * (k - 1) * np.sum((col_sums - col_sums.mean()) ** 2)
    denominator = k * row_sums.sum() - np.sum(row_sums ** 2)
    if denominator == 0:
        raise ValueError("분모가 0입니다 (모든 subject가 전 조건에서 동일한 결과) — Q 계산 불가")
    q = numerator / denominator
    df = k - 1
    from scipy.stats import chi2

    p = float(1 - chi2.cdf(q, df))
    return float(q), p, df


# 2×2 밖의 비교군(P2L, P4, P5)에 대한 **사전 계획된** 대조.
# 요인분석(GEE, Cochran's Q, P0-P3 쌍별 McNemar)과는 별개의 검정 가족으로 다루고
# 각각 자기 가족 안에서 Bonferroni 보정한다. 사후에 눈에 띄는 쌍을 골라 검정하는 것이
# 아니라 설계 단계에서 정해둔 대조이므로 이렇게 나누는 것이 정당하다.
# 근거는 docs/experiment-plan.md 의 P2L / P4 / P5 절.
_PLANNED_CONTRASTS = [
    ("P0", "P2L"),   # 출력 길이만 늘렸을 때의 효과 (길이 교란 확인)
    ("P2", "P2L"),   # 자기검증 vs 길이만 증가 — P2 효과가 길이 때문인지 가른다
    ("P0", "P4"),    # 금지 지시에 효과가 있는가
    ("P1", "P4"),    # 금지 vs 행동 지침
    ("P0", "P5"),    # 예시에 효과가 있는가
    ("P1", "P5"),    # 예시 vs 지시
]


def _mcnemar_pairs(cell, pairs, alpha, out_lines, title) -> None:
    """지정한 조건 쌍들에 대해 McNemar 검정을 하고 Bonferroni 보정 결과를 적는다."""
    from statsmodels.stats.contingency_tables import mcnemar

    out_lines.append(title)
    usable = [(a, b) for a, b in pairs if a in cell.columns and b in cell.columns]
    skipped = [(a, b) for a, b in pairs if (a, b) not in usable]
    if not usable:
        out_lines.append("  (해당 조건의 데이터가 없어 건너뜀)")
        out_lines.append("")
        return
    alpha_corrected = alpha / len(usable)
    out_lines.append(f"보정된 alpha = {alpha} / {len(usable)} = {alpha_corrected:.5f}")
    for a, b in usable:
        try:
            both = cell[[a, b]].dropna()
            tbl = np.zeros((2, 2))
            for _, r in both.iterrows():
                tbl[int(r[a]), int(r[b])] += 1
            result = mcnemar(tbl, exact=(tbl.sum() < 25))
            sig = "유의함" if result.pvalue < alpha_corrected else "유의하지 않음"
            out_lines.append(
                f"  {a} vs {b}: statistic={result.statistic:.4f}, p={result.pvalue:.4f} ({sig})"
            )
        except Exception as inner_exc:  # noqa: BLE001
            out_lines.append(f"  {a} vs {b}: [오류] {inner_exc!r}")
    for a, b in skipped:
        out_lines.append(f"  {a} vs {b}: [건너뜀] 데이터에 없는 조건")
    out_lines.append("")


def _run_cochran_and_mcnemar(df: pd.DataFrame, out_lines: List[str]) -> None:
    out_lines.append("--- Cochran's Q (P0-P3, 질문별 다수결 환각 지표) ---")
    try:
        sub = df[df["prompt_type"].isin(_FACTORIAL_CONDITIONS)]
        cell = (
            sub.groupby(["question_id", "prompt_type"])
            .apply(_majority_vote_indicator, include_groups=False)
            .unstack("prompt_type")
        )
        cell = cell.dropna()
        cell = cell[_FACTORIAL_CONDITIONS]
        mat = cell.to_numpy(dtype=float)
        q, p, dfree = _cochran_q(mat)
        out_lines.append(f"Q={q:.4f}, df={dfree}, p={p:.4f} (n_questions={mat.shape[0]})")
    except Exception as exc:  # noqa: BLE001
        out_lines.append(f"[오류] Cochran's Q 계산 실패: {exc!r}")
    out_lines.append("")

    # (1) 요인설계 안의 쌍별 비교
    try:
        from itertools import combinations

        _mcnemar_pairs(
            cell,
            list(combinations(_FACTORIAL_CONDITIONS, 2)),
            0.05,
            out_lines,
            "--- 쌍별 McNemar: 요인조건 P0-P3 (Bonferroni 보정) ---",
        )
    except Exception as exc:  # noqa: BLE001
        out_lines.append(f"[오류] 요인조건 McNemar 계산 실패: {exc!r}")
        out_lines.append("")

    # (2) 2×2 밖 비교군에 대한 사전 계획 대조
    try:
        all_cell = (
            df.groupby(["question_id", "prompt_type"])
            .apply(_majority_vote_indicator, include_groups=False)
            .unstack("prompt_type")
            .dropna(how="all")
        )
        _mcnemar_pairs(
            all_cell,
            _PLANNED_CONTRASTS,
            0.05,
            out_lines,
            "--- 사전 계획 대조 McNemar: 비교군 P2L/P4/P5 (Bonferroni 보정) ---",
        )
    except Exception as exc:  # noqa: BLE001
        out_lines.append(f"[오류] 계획 대조 McNemar 계산 실패: {exc!r}")
        out_lines.append("")


def run_stats(df: pd.DataFrame, alpha: float) -> str:
    out_lines: List[str] = []
    out_lines.append("=" * 70)
    out_lines.append("통계 분석 결과 (mock 데이터인 경우 실제 결론으로 사용 금지)")
    out_lines.append("=" * 70)
    out_lines.append("")

    work = df.copy()
    work["uncertainty"] = work["prompt_type"].isin(["P1", "P3"]).astype(int)
    work["verification"] = work["prompt_type"].isin(["P2", "P3"]).astype(int)
    work["completion_tokens"] = pd.to_numeric(work["completion_tokens"], errors="coerce")

    for model_key, g in work.groupby("model_key"):
        out_lines.append("#" * 70)
        out_lines.append(f"# 모델: {model_key}")
        out_lines.append("#" * 70)
        out_lines.append("")

        factorial = g[g["prompt_type"].isin(_FACTORIAL_CONDITIONS)]
        out_lines.append("[연구질문 3] GEE: hallucination ~ uncertainty * verification (P0-P3, 상호작용항 확인)")
        _run_gee(factorial, [], out_lines, f"{model_key} - GEE (factorial only)")

        with_p2l = g.copy()
        mean_tok = with_p2l["completion_tokens"].mean()
        std_tok = with_p2l["completion_tokens"].std()
        with_p2l["completion_tokens_z"] = (
            (with_p2l["completion_tokens"] - mean_tok) / std_tok if std_tok and std_tok > 0 else 0.0
        )
        out_lines.append("[길이 교란 통제] GEE: hallucination ~ uncertainty * verification + completion_tokens_z (P0-P3-P2L 전체)")
        _run_gee(with_p2l, ["completion_tokens_z"], out_lines, f"{model_key} - GEE (with length covariate)")

        _run_cochran_and_mcnemar(g, out_lines)

    return "\n".join(out_lines)


# ---------------------------------------------------------------------------
# 그래프
# ---------------------------------------------------------------------------

_MODEL_MARKERS = {"qwen35": "o", "gemma4": "s", "gptoss20": "^"}
_MODEL_COLORS = {"qwen35": "#3B82C4", "gemma4": "#D9822B", "gptoss20": "#4C9A6A"}


def _title(base: str, run_id: str, is_mock: bool) -> str:
    if is_mock:
        return f"{base}\n[run_id={run_id}] MOCK DATA — NOT REAL RESULTS"
    return f"{base}\n[run_id={run_id}]"


def _grouped_bar(
    summary: pd.DataFrame,
    value_col: str,
    ci_low_col: str,
    ci_high_col: str,
    ylabel: str,
    title: str,
    out_path: str,
) -> None:
    models = sorted(summary["model_key"].unique())
    conditions = _order_conditions(summary["prompt_type"].unique())
    x = np.arange(len(conditions))
    width = 0.8 / max(len(models), 1)

    fig, ax = plt.subplots(figsize=(9, 5.5))
    for i, model_key in enumerate(models):
        sub = summary[summary["model_key"] == model_key].set_index("prompt_type")
        sub = sub.reindex(conditions)
        vals = sub[value_col].to_numpy(dtype=float)
        lo = sub[ci_low_col].to_numpy(dtype=float)
        hi = sub[ci_high_col].to_numpy(dtype=float)
        yerr = np.vstack([vals - lo, hi - vals])
        yerr = np.nan_to_num(yerr, nan=0.0)
        ax.bar(
            x + i * width,
            vals,
            width=width,
            yerr=yerr,
            capsize=3,
            label=model_key,
            color=_MODEL_COLORS.get(model_key),
        )
    ax.set_xticks(x + width * (len(models) - 1) / 2)
    ax.set_xticklabels(conditions)
    ax.set_ylabel(ylabel)
    ax.set_ylim(0, 1.0)
    ax.set_title(title)
    ax.legend(title="model")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _abstention_tradeoff_bar(summary: pd.DataFrame, title: str, out_path: str) -> None:
    models = sorted(summary["model_key"].unique())
    conditions = _order_conditions(summary["prompt_type"].unique())
    x = np.arange(len(conditions))
    n_series = len(models) * 2
    width = 0.8 / max(n_series, 1)

    fig, ax = plt.subplots(figsize=(10, 6))
    slot = 0
    for model_key in models:
        sub = summary[summary["model_key"] == model_key].set_index("prompt_type").reindex(conditions)
        color = _MODEL_COLORS.get(model_key)
        ax.bar(
            x + slot * width,
            sub["correct_abstention_rate"].to_numpy(dtype=float),
            width=width,
            label=f"{model_key} - correct_abstention",
            color=color,
            alpha=0.9,
        )
        slot += 1
        ax.bar(
            x + slot * width,
            sub["over_abstention_rate"].to_numpy(dtype=float),
            width=width,
            label=f"{model_key} - over_abstention",
            color=color,
            alpha=0.4,
            hatch="//",
        )
        slot += 1
    ax.set_xticks(x + width * (n_series - 1) / 2)
    ax.set_xticklabels(conditions)
    ax.set_ylabel("rate")
    ax.set_ylim(0, 1.0)
    ax.set_title(title)
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _tradeoff_scatter(summary: pd.DataFrame, title: str, out_path: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 7))
    for _, row in summary.iterrows():
        marker = _MODEL_MARKERS.get(row["model_key"], "x")
        color = _MODEL_COLORS.get(row["model_key"])
        ax.scatter(
            row["hallucination_rate"],
            row["accuracy_at_answerable"],
            marker=marker,
            color=color,
            s=90,
            edgecolors="black",
            linewidths=0.5,
        )
        ax.annotate(
            f"{row['model_key']}/{row['prompt_type']}",
            (row["hallucination_rate"], row["accuracy_at_answerable"]),
            fontsize=7,
            xytext=(4, 4),
            textcoords="offset points",
        )
    ax.set_xlabel("Hallucination Rate (낮을수록 좋음)")
    ax.set_ylabel("Accuracy@Answerable (높을수록 좋음)")
    ax.set_title(title)
    ax.annotate(
        "▲ up-left = better",
        xy=(0.02, 0.96),
        xycoords="axes fraction",
        fontsize=9,
        ha="left",
        va="top",
        bbox=dict(boxstyle="round", fc="white", ec="gray"),
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def make_graphs(summary: pd.DataFrame, run_id: str, is_mock: bool) -> None:
    _setup_korean_font()

    _grouped_bar(
        summary,
        "hallucination_rate",
        "hallucination_rate_ci_low",
        "hallucination_rate_ci_high",
        "Hallucination Rate",
        _title("환각률 (Hallucination Rate)", run_id, is_mock),
        "graphs/hallucination_rate.png",
    )
    _grouped_bar(
        summary,
        "accuracy_at_answerable",
        "accuracy_at_answerable_ci_low",
        "accuracy_at_answerable_ci_high",
        "Accuracy@Answerable",
        _title("정답률 (Accuracy@Answerable)", run_id, is_mock),
        "graphs/accuracy.png",
    )
    _abstention_tradeoff_bar(
        summary,
        _title("보류율: 정답보류 vs 과잉보류", run_id, is_mock),
        "graphs/abstention_rate.png",
    )
    _tradeoff_scatter(
        summary,
        _title("환각률-정답률 트레이드오프", run_id, is_mock),
        "graphs/tradeoff.png",
    )


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------

def run(config_path: str) -> None:
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    run_id = config["run_id"]
    is_mock = bool(config.get("mock", False))
    alpha = config.get("analysis", {}).get("alpha", 0.05)

    df = pd.read_csv("results/evaluated.csv", dtype=str, keep_default_na=False, encoding="utf-8-sig")
    logger.info("evaluated.csv %d행 로드", len(df))

    summary = compute_summary(df, alpha)
    summary.to_csv("results/summary.csv", index=False)
    logger.info("results/summary.csv 작성 완료 (%d행)", len(summary))

    stats_text = run_stats(df, alpha)
    with open("results/stats.txt", "w", encoding="utf-8") as f:
        f.write(stats_text)
    logger.info("results/stats.txt 작성 완료")

    make_graphs(summary, run_id, is_mock)
    logger.info("그래프 4종 작성 완료 (graphs/)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
