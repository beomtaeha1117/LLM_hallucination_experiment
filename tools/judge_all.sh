#!/usr/bin/env bash
# 세 모델의 응답을 순서대로 판정한다. 한 번 걸어두고 자리를 뜨기 위한 것이다.
#
#   bash tools/judge_all.sh
#   tail -f judge_all.log
#
# 판정은 모델 하나만 쓰므로(judge) 중간에 모델을 바꿀 일이 없다.
# 죽어도 같은 명령으로 이어서 한다 — evaluate가 행 단위로 resume한다.
set -u
cd "$(dirname "$0")/.."
LOG=judge_all.log
say() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOG"; }

say "=== 사전 점검 ==="
if ! python tools/preflight.py --config config_main_qwen35.yaml 2>&1 | tee -a "$LOG" | grep -q "통과"; then
  say "!! 주모델이 로드돼 있어야 통과합니다. judge만 쓸 것이므로 이 점검은 건너뜁니다."
fi

for cfg in config_main_qwen35.yaml config_main_gemma4.yaml config_main_gptoss20.yaml; do
  say "=== 판정: $cfg ==="
  python -m src.evaluate --config "$cfg" 2>&1 | tee -a "$LOG"
  if [ "${PIPESTATUS[0]}" -ne 0 ]; then
    say "!! $cfg 판정이 중단됐습니다. 같은 명령으로 이어서 돌리십시오."
    exit 1
  fi
done

say "=== 판정 끝. 다음은 analyze입니다. ==="
