"""실행 직전 점검 — 모델이 올라와 있는가, 컨텍스트가 8192인가.

전원을 껐다 켜거나 모델이 언로드되면 JIT 로딩이 모델을 자동으로 올리는데, 그때
컨텍스트가 기본값(262144)으로 잡힌다. 통제변인이 어긋난 채로 13시간이 돌아가고
로그에는 아무 흔적도 남지 않는다. 그걸 시작 전에 잡는다.

    python tools/preflight.py --config config_pilot_real.yaml
"""
from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request

import yaml

EXPECTED_CTX = 8192


def _get(url: str):
    try:
        with urllib.request.urlopen(url, timeout=10) as r:
            return json.loads(r.read().decode("utf-8"))
    except (urllib.error.URLError, json.JSONDecodeError, OSError) as e:
        return {"__error__": str(e)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config_pilot_real.yaml")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    base = cfg["server"]["base_url"].rstrip("/")
    root = base[: -len("/v1")] if base.endswith("/v1") else base
    want = [m["lms_id"] for m in cfg["models"]]
    if cfg.get("judge"):
        want.append(cfg["judge"]["lms_id"])

    print(f"config: {args.config}")
    print(f"필요한 모델: {', '.join(want)}\n")

    # LM Studio 확장 API. OpenAI 호환 /v1/models와 달리 로드 상태와 컨텍스트를 준다.
    # 다만 LM Studio 버전에 따라 없을 수 있으므로, 없으면 없다고 말하고 넘어간다.
    data = _get(f"{root}/api/v0/models")
    if "__error__" in data or "data" not in data:
        print("!! LM Studio 확장 API(/api/v0/models)를 읽지 못했습니다.")
        print(f"   {data.get('__error__', data)}")
        print("   컨텍스트 길이를 자동으로 확인할 수 없습니다 —")
        print("   LM Studio 화면에서 Context Length가 8192인지 직접 보십시오.")
        return 2

    by_id = {m.get("id"): m for m in data["data"]}
    bad = False
    for mid in want:
        m = by_id.get(mid)
        if m is None:
            print(f"  [없음]   {mid}  — 이 서버에 없는 모델입니다. config의 lms_id를 확인하십시오.")
            bad = True
            continue
        state = m.get("state", "unknown")
        ctx = m.get("loaded_context_length", m.get("max_context_length"))
        mark = "OK  "
        if state != "loaded":
            mark = "미로드"
        elif ctx is not None and int(ctx) != EXPECTED_CTX:
            mark = "!!ctx"
            bad = True
        print(f"  [{mark}] {mid}  state={state}  context={ctx}")

    print()
    if bad:
        print("!! 그대로 돌리지 마십시오.")
        print(f"   컨텍스트가 {EXPECTED_CTX}가 아니면 통제변인이 어긋난 실행이 됩니다.")
        print("   LM Studio에서 모델을 언로드하고 Context Length 8192로 다시 로드하십시오")
        print("   (JIT 로딩이 자동으로 올리면 기본값 262144로 잡힙니다).")
        return 1

    if any(by_id.get(m, {}).get("state") != "loaded" for m in want[:1]):
        print("주모델이 아직 안 올라와 있습니다. JIT에 맡기지 말고 직접 로드하십시오.")
        return 1

    print("통과. 실행해도 됩니다.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
