"""요약 지표 + 통계 검정 + 그래프 생성 스테이지."""

from __future__ import annotations

import argparse
import logging
import os
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
# 길이 교란 통제 GEE 전용 조건 집합. P4/P5는 여기 포함하지 않는다 — 포함하면
# with_p2l이 "P0-P3-P2L 전체"라는 주석과 달리 P4/P5까지 끌고 들어와 P0의
# uncertainty=0/verification=0 셀에 조용히 섞여 들어간다 (defect 11).
_LENGTH_CONTROL_CONDITIONS = _FACTORIAL_CONDITIONS + ["P2L"]


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
    """Wilson CI 버전. 12,600건이 독립이 아니라는(문항당 3회 반복) 문제를
    무시하므로 실제 CI보다 좁게 나온다. 삭제하지 않고 참고/폴백용으로 남겨둔다 —
    summary.csv/그래프는 아래 클러스터 부트스트랩 버전을 쓴다 (defect 12)."""
    rate = count / nobs if nobs else float("nan")
    low, high = _wilson_ci(count, nobs, alpha)
    return {"rate": rate, "ci_low": low, "ci_high": high}


# ---------------------------------------------------------------------------
# 문항(question_id) 클러스터 부트스트랩 CI (defect 12)
# ---------------------------------------------------------------------------
#
# 12,600개 응답은 서로 독립이 아니다 — 문항당 3회 반복이 상관되어 있으므로
# Wilson CI처럼 응답 단위로 풀링하면 실제보다 CI가 좁게 나온다. 대신
# question_id를 리샘플링 단위로 삼는 클러스터 부트스트랩을 쓴다: 관측된 유니크
# question_id 개수만큼 question_id를 복원추출하고, 뽑힌 question_id에 속한
# 모든 행을 모아 지표를 계산하는 것을 n_boot회 반복해 백분위수로 CI를 낸다.


def _cluster_bootstrap_ci(
    df: pd.DataFrame,
    indicator_col: str,
    question_col: str = "question_id",
    n_boot: int = 2000,
    alpha: float = 0.05,
    rng_seed: int = 12345,
) -> Tuple[float, float]:
    if df.empty:
        return (float("nan"), float("nan"))

    grouped = {
        qid: sub[indicator_col].to_numpy(dtype=float)
        for qid, sub in df.groupby(question_col)
    }
    qid_list = list(grouped.keys())
    n_q = len(qid_list)
    if n_q == 0:
        return (float("nan"), float("nan"))

    rng = np.random.default_rng(rng_seed)
    rates = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        sampled_qids = rng.choice(qid_list, size=n_q, replace=True)
        vals = np.concatenate([grouped[q] for q in sampled_qids])
        rates[b] = vals.mean() if vals.size else float("nan")

    low, high = np.nanpercentile(rates, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(low), float(high)


def _rate_with_ci_cluster(
    sub: pd.DataFrame,
    indicator_col: str,
    alpha: float,
    question_col: str = "question_id",
    n_boot: int = 2000,
    rng_seed: int = 12345,
) -> Dict[str, float]:
    """`_rate_with_ci`의 클러스터 버전. count/nobs 대신, 지표(0/1) 컬럼과
    question_id를 담은 부분 데이터프레임을 받는다 — 클러스터링에는 풀링된
    카운트가 아니라 문항별 분해가 필요하기 때문이다."""
    if sub.empty:
        return {"rate": float("nan"), "ci_low": float("nan"), "ci_high": float("nan")}
    rate = float(sub[indicator_col].astype(float).mean())
    low, high = _cluster_bootstrap_ci(
        sub, indicator_col, question_col=question_col, n_boot=n_boot, alpha=alpha, rng_seed=rng_seed
    )
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

        # defect 12: 응답 단위 풀링(Wilson) 대신 question_id 클러스터 부트스트랩으로
        # CI를 낸다 — 문항당 3회 반복이 상관되어 있으므로, 지표(0/1) 컬럼 + question_id를
        # 담은 부분 데이터프레임을 만들어 _rate_with_ci_cluster에 넘긴다.
        halluc = _rate_with_ci_cluster(
            g.assign(_ind=(g["label"] == "HALLUCINATION")), "_ind", alpha
        )
        acc = _rate_with_ci_cluster(
            y.assign(_ind=(y["label"] == "CORRECT")), "_ind", alpha
        )
        correct_abstain = _rate_with_ci_cluster(
            n_g.assign(_ind=(n_g["label"] == "CORRECT")), "_ind", alpha
        )
        over_abstain = _rate_with_ci_cluster(
            y.assign(_ind=(y["label"] == "ABSTAIN")), "_ind", alpha
        )

        # answer_rate: 실제로 내용을 답한 비율. Y는 ABSTAIN이 아니면 실답변,
        # N은 HALLUCINATION(지어냄)이어야 실답변이다 (CORRECT는 보류/전제지적이므로 실답변이 아님).
        real_answer = int((y["label"] != "ABSTAIN").sum()) + int((n_g["label"] == "HALLUCINATION").sum())
        real_answer_df = pd.concat(
            [
                y.assign(_ind=(y["label"] != "ABSTAIN")),
                n_g.assign(_ind=(n_g["label"] == "HALLUCINATION")),
            ],
            ignore_index=True,
        )
        answer = _rate_with_ci_cluster(real_answer_df, "_ind", alpha)

        # abstention precision/recall/F1: positive = unanswerable(N)
        pred_abstain = g["predicted_abstain"]
        actual_n = g["answerable"] == "N"
        precision_sub = g.loc[pred_abstain].copy()
        precision_sub["_ind"] = actual_n.loc[pred_abstain]
        precision = _rate_with_ci_cluster(precision_sub, "_ind", alpha)
        recall_sub = g.loc[actual_n].copy()
        recall_sub["_ind"] = pred_abstain.loc[actual_n]
        recall = _rate_with_ci_cluster(recall_sub, "_ind", alpha)
        f1 = (2 * precision["rate"] * recall["rate"] / (precision["rate"] + recall["rate"])) if (
            precision["rate"] + recall["rate"] > 0
        ) else float("nan")

        format_ok = _rate_with_ci_cluster(g.assign(_ind=g["format_ok_bool"]), "_ind", alpha)

        # 결함 #2: parse_mode 분포(marker/last_section/raw 비율). 조건별로 마커를
        # 얼마나 빠뜨렸는지, 빠뜨렸을 때 last_section 경로로 얼마나 구제됐는지를
        # 본다. 구버전 raw_responses.csv에는 parse_mode 컬럼이 없을 수 있어
        # 안전하게 처리한다.
        parse_mode_col = g["parse_mode"] if "parse_mode" in g.columns else pd.Series([""] * len(g), index=g.index)
        parse_mode_rate = (
            parse_mode_col.value_counts(normalize=True).to_dict() if len(g) else {}
        )

        # 결함 #3/#4: answerable=N 행에서 false_premise_correction 비율. 기존
        # 지표 6종(correct_abstention_rate 포함)의 계산식은 건드리지 않는다 —
        # 이건 그 내부를 더 쪼갠 진단용 보조 지표다.
        if "response_kind" in n_g.columns and len(n_g):
            false_premise_correction = _rate_with_ci_cluster(
                n_g.assign(_ind=(n_g["response_kind"] == "false_premise_correction")), "_ind", alpha
            )
        else:
            false_premise_correction = {"rate": float("nan"), "ci_low": float("nan"), "ci_high": float("nan")}

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
                "parse_mode_marker_rate": parse_mode_rate.get("marker", 0.0),
                "parse_mode_last_section_rate": parse_mode_rate.get("last_section", 0.0),
                "parse_mode_raw_rate": parse_mode_rate.get("raw", 0.0),
                "false_premise_correction_rate": false_premise_correction["rate"],
                "false_premise_correction_rate_ci_low": false_premise_correction["ci_low"],
                "false_premise_correction_rate_ci_high": false_premise_correction["ci_high"],
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

        with_p2l = g[g["prompt_type"].isin(_LENGTH_CONTROL_CONDITIONS)].copy()
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


def _title(base: str, run_id: str, is_mock: bool, note_ci: bool = False) -> str:
    if is_mock:
        title = f"{base}\n[run_id={run_id}] MOCK DATA — NOT REAL RESULTS"
    else:
        title = f"{base}\n[run_id={run_id}]"
    if note_ci:
        # defect 12: CI가 Wilson(응답 단위 독립 가정)이 아니라 문항 클러스터
        # 부트스트랩임을 그래프에서도 알 수 있게 한다.
        title += "\n(CI: 문항 클러스터 부트스트랩)"
    return title


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

    # defect 10: graphs/ 아래 최상위에 계속 쌓이면 run을 거듭할수록 예전 run의
    # PNG가 새 run의 PNG와 뒤섞인다 — results/<run_id>/와 마찬가지로 run별
    # 디렉터리로 분리한다.
    graphs_dir = f"graphs/{run_id}"
    os.makedirs(graphs_dir, exist_ok=True)

    _grouped_bar(
        summary,
        "hallucination_rate",
        "hallucination_rate_ci_low",
        "hallucination_rate_ci_high",
        "Hallucination Rate",
        _title("환각률 (Hallucination Rate)", run_id, is_mock, note_ci=True),
        f"{graphs_dir}/hallucination_rate.png",
    )
    _grouped_bar(
        summary,
        "accuracy_at_answerable",
        "accuracy_at_answerable_ci_low",
        "accuracy_at_answerable_ci_high",
        "Accuracy@Answerable",
        _title("정답률 (Accuracy@Answerable)", run_id, is_mock, note_ci=True),
        f"{graphs_dir}/accuracy.png",
    )
    _abstention_tradeoff_bar(
        summary,
        _title("보류율: 정답보류 vs 과잉보류", run_id, is_mock, note_ci=True),
        f"{graphs_dir}/abstention_rate.png",
    )
    _tradeoff_scatter(
        summary,
        _title("환각률-정답률 트레이드오프", run_id, is_mock),
        f"{graphs_dir}/tradeoff.png",
    )


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------

def _load_evaluated_for_run(run_id: str, is_mock: bool) -> pd.DataFrame:
    """defect 7: 이 run만의 evaluated 데이터를 읽는다.

    evaluate.py는 아직 고칠 수 없어 (경로 하드코딩) results/evaluated.csv에
    모든 run을 계속 합쳐 쓴다. 그래서:
    1) 먼저 미래의 run별 evaluate.py 출력을 가정하고 results/<run_id>/evaluated.csv를
       찾는다.
    2) 없으면 현재의 합쳐진 results/evaluated.csv로 폴백하되, run_id와 is_mock이
       모두 일치하는 행만 남긴다. is_mock은 CSV에 문자열로 저장되어 있으므로
       (run_experiment.py의 csv.DictWriter가 Python bool의 str() — "True"/"False" —
       를 그대로 쓴다) 대소문자를 정규화해 비교한다.
    필터링 결과가 비면, 통계/그래프가 조용히 빈 채로 나오는 대신 여기서 바로
    에러를 낸다.
    """
    per_run_path = f"results/{run_id}/evaluated.csv"
    if os.path.exists(per_run_path):
        df = pd.read_csv(per_run_path, dtype=str, keep_default_na=False, encoding="utf-8-sig")
        logger.info("%s 에서 %d행 로드 (run별 evaluated.csv)", per_run_path, len(df))
        return df

    fallback_path = "results/evaluated.csv"
    df = pd.read_csv(fallback_path, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    logger.info(
        "%s 를 찾지 못해 %s 를 읽은 뒤 run_id=%s, is_mock=%s 로 필터링합니다.",
        per_run_path,
        fallback_path,
        run_id,
        is_mock,
    )
    mask_run = df["run_id"].astype(str).str.strip() == str(run_id)
    mask_mock = df["is_mock"].astype(str).str.strip().str.lower() == str(is_mock).lower()
    filtered = df[mask_run & mask_mock].reset_index(drop=True)
    logger.info(
        "필터링 후 %d행 (필터 전 %d행, path=%s, run_id==%s AND is_mock==%s)",
        len(filtered),
        len(df),
        fallback_path,
        run_id,
        is_mock,
    )
    if filtered.empty:
        raise RuntimeError(
            f"'{fallback_path}'에서 run_id='{run_id}' AND is_mock={is_mock} 조건에 맞는 "
            f"행을 찾지 못했습니다 (필터 전 총 {len(df)}행). run_id 오타, mock 설정 불일치, "
            f"혹은 evaluate 단계가 이 run에 대해 아직 실행되지 않았을 가능성을 확인하십시오."
        )
    return filtered


def run(config_path: str) -> None:
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    run_id = config["run_id"]
    is_mock = bool(config.get("mock", False))
    alpha = config.get("analysis", {}).get("alpha", 0.05)

    df = _load_evaluated_for_run(run_id, is_mock)

    os.makedirs(f"results/{run_id}", exist_ok=True)

    # JUDGE_ERROR는 판정 실패이지 관측이 아니다 (evaluate.py 결함 #5 대응 라벨).
    # 분모에 남겨두면 judge 백엔드가 죽은 만큼 환각률/답변률이 왜곡된다.
    # 여기서 걷어내되, 몇 건이었는지는 반드시 남긴다 — 조용히 사라지면
    # judge가 절반쯤 실패한 실행을 정상 실행으로 착각하게 된다.
    n_judge_error = int((df["label"] == "JUDGE_ERROR").sum()) if "label" in df.columns else 0
    if n_judge_error:
        pct = 100.0 * n_judge_error / len(df)
        logger.warning(
            "JUDGE_ERROR %d행(%.2f%%)을 분석에서 제외합니다. judge 백엔드 상태를 확인하십시오.",
            n_judge_error, pct,
        )
        if pct >= 5.0:
            raise RuntimeError(
                f"JUDGE_ERROR가 {pct:.1f}%로 5%를 넘습니다({n_judge_error}/{len(df)}행). "
                "판정이 대량 실패한 실행을 분석하면 안 됩니다. judge를 고치고 evaluate를 다시 돌리십시오."
            )
        df = df[df["label"] != "JUDGE_ERROR"].copy()

    summary = compute_summary(df, alpha)
    summary_path = f"results/{run_id}/summary.csv"
    summary.to_csv(summary_path, index=False)
    logger.info("%s 작성 완료 (%d행)", summary_path, len(summary))

    stats_text = run_stats(df, alpha)
    stats_path = f"results/{run_id}/stats.txt"
    with open(stats_path, "w", encoding="utf-8") as f:
        f.write(stats_text)
    logger.info("%s 작성 완료", stats_path)

    make_graphs(summary, run_id, is_mock)
    logger.info("그래프 4종 작성 완료 (graphs/%s/)", run_id)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
