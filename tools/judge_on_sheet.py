"""라벨링 표본에 judge 후보를 돌려서 κ 비교용 auto_label을 만든다.

judge 선정은 "사람 라벨 vs judge 라벨"의 Cohen's kappa로 한다(docs/RUNBOOK.md §3-2).
사람 라벨링은 몇 시간이 걸리지만 judge 쪽은 사람 라벨과 무관하므로 **먼저 돌려둘 수
있다.** 그래야 라벨링이 끝나는 순간 κ가 바로 나온다.

후보를 여러 개 재려면 --lms-id로 바꿔가며 돌리고 --out을 다르게 준다.

    python tools/judge_on_sheet.py --lms-id devstral-small-2-24b-instruct-2512
    python -m src.agreement --full results/human_labeling/judge_devstral.csv \\
        results/human_labeling/labeling_sheet.csv
"""
from __future__ import annotations

import argparse
import csv
import logging
import os
import re

import sys

import pandas as pd
import yaml
from tqdm import tqdm

# tools/ 에서 직접 실행해도 src 를 찾게 한다 (다른 tools 는 src 를 안 쓴다).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.evaluate import LMStudioJudge  # noqa: E402
from src.logging_setup import quiet_http_logs  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sheet", default="results/human_labeling/labeling_sheet.csv")
    ap.add_argument("--config", default="config_main_qwen35.yaml", help="server 설정만 쓴다")
    ap.add_argument("--lms-id", default=None, help="judge 모델 id (기본: config의 judge)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--votes", type=int, default=1)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    quiet_http_logs()

    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    lms_id = args.lms_id or cfg["judge"]["lms_id"]
    out = args.out or os.path.join(
        os.path.dirname(args.sheet),
        "judge_" + re.sub(r"[^A-Za-z0-9]+", "_", lms_id).strip("_")[:40] + ".csv",
    )

    sheet = pd.read_csv(args.sheet, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    print(f"{args.sheet}  {len(sheet)}건")
    print(f"judge: {lms_id}  (votes={args.votes})")
    print(f"출력: {out}\n")

    srv = cfg["server"]
    judge = LMStudioJudge(
        base_url=srv["base_url"], api_key=srv.get("api_key", "lm-studio"),
        lms_id=lms_id, temperature=cfg["judge"].get("temperature", 0.0),
        votes=args.votes, timeout_s=srv.get("timeout_s", 300),
        max_retries=srv.get("max_retries", 3),
    )

    # 응답 하나가 실행 전체를 죽이지 않게 한다 — 생성 단계에서 겪은 그대로다.
    rows, failed = [], 0
    with open(out, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=["sample_id", "auto_label", "judge_label_raw",
                                          "abstain_with_claim", "reason"])
        w.writeheader()
        for _, r in tqdm(sheet.iterrows(), total=len(sheet), desc="judging"):
            try:
                label, reason, awc = judge.judge(r.to_dict(), r.get("response_final", ""))
            except Exception as exc:  # noqa: BLE001
                failed += 1
                label, reason, awc = "", f"{type(exc).__name__}: {exc}"[:300], False
            # 🚨 judge의 원본 라벨을 그대로 auto_label로 쓰면 안 된다.
            # evaluate.py는 answerable=N 행에서 judge가 ABSTAIN을 내도 CORRECT로
            # 매핑한다(답할 수 없는 문항에서는 보류가 곧 정답이기 때문이다).
            # 매핑 없이 κ를 재면 사람이 CORRECT라 적은 것과 어긋나 실제보다 훨씬
            # 낮은 κ가 나온다. 첫 실행에서 100건 중 ABSTAIN이 54건이었는데
            # 표본의 N 문항이 60개였다 — 거의 전부 이 경우였다.
            mapped = label
            if str(r.get("answerable", "")).strip() == "N" and label in ("CORRECT", "ABSTAIN"):
                mapped = "CORRECT"
            w.writerow({"sample_id": r["sample_id"], "auto_label": mapped,
                        "judge_label_raw": label,
                        "abstain_with_claim": awc, "reason": reason})
            f.flush()
            rows.append(mapped)

    got = pd.Series([x for x in rows if x])
    print(f"\n완료. 실패 {failed}건")
    if not got.empty:
        print("judge 라벨 분포:")
        print(got.value_counts().to_string())
    print(f"\n사람 라벨링이 끝나면:\n"
          f"  python -m src.agreement --full {out} {args.sheet}")


if __name__ == "__main__":
    main()
