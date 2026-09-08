#!/usr/bin/env bash
# 파일럿을 관문으로 두고, 통과하면 본실험까지 이어서 돌린다.
#
# 시간이 없을 때 한 번 걸어두고 자리를 뜨기 위한 것이다. 파일럿에서 잘림이
# 많이 남으면 본실험을 시작하지 않고 멈춘다 — 6시간을 잘린 응답으로 채우느니
# 아침에 원인을 보는 편이 낫다.
#
#   bash tools/gate_and_run.sh
#
# 진행 상황:  tail -f gate_run.log
set -u
cd "$(dirname "$0")/.."
LOG=gate_run.log
MAIN_CFG=config_main_qwen35.yaml
PILOT_CFG=config_pilot_real.yaml
TRUNC_MAX=2   # 파일럿 189건 중 이보다 많이 잘리면 본실험을 시작하지 않는다

say() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOG"; }

say "=== 0/3  사전 점검 (모델 로드 상태와 컨텍스트) ==="
if ! python tools/preflight.py --config "$PILOT_CFG" 2>&1 | tee -a "$LOG" | grep -q "통과"; then
  say "!! 사전 점검 실패. LM Studio에서 모델을 ctx 8192로 직접 로드한 뒤 다시 실행하십시오."
  exit 1
fi

say "=== 1/3  파일럿 (189응답, 약 20~25분) ==="
python -m src.run_experiment --config "$PILOT_CFG" 2>&1 | tee -a "$LOG"
[ "${PIPESTATUS[0]}" -eq 0 ] || { say "!! 파일럿이 실패했습니다. 중단합니다."; exit 1; }

say "=== 2/3  관문 판정 (잘림이 ${TRUNC_MAX}건 이하인가) ==="
TRUNC=$(python - <<'PY'
import glob, pandas as pd
f = sorted(glob.glob("results/pilot_003/raw_responses.csv"))
print(0 if not f else int((pd.read_csv(f[0], dtype=str)["finish_reason"] == "length").sum()))
PY
)
say "파일럿 잘림: ${TRUNC}건 (허용 ${TRUNC_MAX}건)"
if [ "$TRUNC" -gt "$TRUNC_MAX" ]; then
  say "!! 잘림이 아직 많습니다. 본실험을 시작하지 않고 멈춥니다 —"
  say "   max_tokens를 더 올려야 하는지 아침에 보십시오. 파일럿 결과는 남아 있습니다."
  exit 2
fi

say "=== 3/3  본실험 시작 (4,200응답, 약 5.5~6.5시간) ==="
say "중간에 끊겨도 같은 명령으로 이어서 합니다(resume)."
python -m src.run_experiment --config "$MAIN_CFG" 2>&1 | tee -a "$LOG"
[ "${PIPESTATUS[0]}" -eq 0 ] || { say "!! 본실험이 중단됐습니다. 같은 명령으로 이어서 돌리십시오."; exit 1; }

say "=== 끝. 생성 완료 — 판정(judge)은 아직입니다. ==="
say "결과: results/main_qwen35/raw_responses.csv"
