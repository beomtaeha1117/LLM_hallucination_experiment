#!/usr/bin/env bash
# 남은 것을 한 번에 돌린다 — 실험 B 판정, 일치도 재확인, 다섯 실행 분석.
#
#   bash tools/finish_all.sh
#   tail -f finish_all.log
#
# LM Studio에 judge(devstral)가 ctx 8192로 올라와 있어야 한다. 사전 점검이 막는다.
# 중간에 끊겨도 같은 명령으로 이어서 한다 — 판정은 행 단위로 resume한다.
set -u
cd "$(dirname "$0")/.."
LOG=finish_all.log
say() { echo "" | tee -a "$LOG"; echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOG"; }
run() { "$@" 2>&1 | tee -a "$LOG"; return "${PIPESTATUS[0]}"; }

say "=== 0/4  사전 점검 — judge가 ctx 8192로 올라와 있는가 ==="
# 주모델은 판정에 쓰지 않으므로 judge 줄만 본다.
if ! python tools/preflight.py --config config_b_qwen35.yaml 2>&1 | tee -a "$LOG" \
     | grep -qE "devstral.*state=loaded.*context=8192"; then
  say "!! judge(devstral)가 ctx 8192로 로드돼 있지 않습니다."
  say "   LM Studio에서 devstral-small-2-24b-instruct-2512을 Context Length 8192로 올린 뒤"
  say "   다시 실행하십시오. JIT에 맡기면 393216으로 올라와 스왑에 걸립니다."
  exit 1
fi

say "=== 1/4  실험 B 판정 (960건, 약 20분) ==="
for cfg in config_b_qwen35.yaml config_b_gemma4.yaml; do
  say "--- $cfg ---"
  run python -m src.evaluate --config "$cfg" || { say "!! $cfg 판정 중단. 같은 명령으로 이어서 돌리십시오."; exit 1; }
done

say "=== 2/4  사람 라벨 대조 재확인 (GPU 없음) ==="
run python tools/agreement_pipeline.py || say "  (대조 실패 — 치명적이지 않으므로 계속합니다)"

say "=== 3/4  abstain_with_claim을 라벨에 반영하면 득실이 어떤가 ==="
run python - <<'PY'
import glob
import pandas as pd

# 환각 재현율이 0.727이고 놓친 것 중 일부는 judge가 abstain_with_claim을 세웠는데
# 라벨 규칙이 버린 것이다. 그 플래그를 반영하면 재현율은 오르지만 CORRECT 쪽에
# 거짓 양성이 생길 수 있다. 득실을 표로 본다 — 규칙을 바꾸기 전에 볼 숫자다.
J = ["run_id", "model_key", "prompt_type", "question_id", "repeat"]
h = pd.read_csv("results/human_labeling/labeling_sheet.csv", dtype=str, keep_default_na=False)
k = pd.read_csv("results/human_labeling/labeling_key.csv", dtype=str, keep_default_na=False)
ev = pd.concat([pd.read_csv(f, dtype=str, keep_default_na=False)[J + ["label", "abstain_with_claim"]]
                for f in sorted(glob.glob("results/main_*/evaluated.csv"))], ignore_index=True)
d = h[["sample_id", "human_label"]].merge(k, on="sample_id").merge(ev, on=J)
d["awc"] = d["abstain_with_claim"].str.strip().str.lower().isin(["true", "1"])

print(pd.crosstab([d["label"], d["awc"]], d["human_label"]).to_string())

flip = d[(d["label"] == "CORRECT") & d["awc"]]
gain = int((flip["human_label"] == "HALLUCINATION").sum())
loss = int((flip["human_label"] == "CORRECT").sum())
tot_h = int((d["human_label"] == "HALLUCINATION").sum())
now = int(((d["label"] == "HALLUCINATION") & (d["human_label"] == "HALLUCINATION")).sum())
print(f"\nCORRECT이면서 abstain_with_claim=True인 행 {len(flip)}건을 HALLUCINATION으로 바꾸면:")
print(f"  제대로 잡는 것 +{gain}건, 잘못 뒤집는 것 +{loss}건")
print(f"  환각 재현율 {now}/{tot_h} ({now/tot_h:.3f}) -> {now+gain}/{tot_h} ({(now+gain)/tot_h:.3f})")
print("\n  gain > loss 면 규칙을 바꿀 값어치가 있다. 반대면 지금 라벨을 유지하고")
print("  과소추정임을 논문 한계에 적는 편이 낫다(현재 RUNBOOK §3-5의 판단).")
PY

say "=== 4/4  분석 — 다섯 실행 ==="
for cfg in config_main_qwen35 config_main_gemma4 config_main_gptoss20 config_b_qwen35 config_b_gemma4; do
  say "--- $cfg ---"
  run python -m src.analyze --config "$cfg.yaml" || say "  !! $cfg 분석 실패 — 계속합니다"
done

say "=== 끝 ==="
say "결과: results/<run_id>/summary.csv, stats.txt / graphs/<run_id>/"
say "다음: 문항 87개 검증(verify/worklist.html) -> 해당 행 재판정 -> 최종 분석"
