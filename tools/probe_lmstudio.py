"""LM Studio 실서버 프로브 — 파일럿 진입 전에 딱 한 번 돌린다.

이 스크립트는 **추측하지 않는다.** 코드가 확인할 수 없는 것 세 가지를 실제 서버에
물어보고, 되는 것과 안 되는 것을 그대로 출력한다.

  1. seed가 실제로 반영되는가   <- 안 되면 3회 반복이 같은 응답이 되어 실험이 무너진다
  2. thinking을 끄는 방법이 무엇인가 <- config에는 thinking:false가 있지만 파라미터명이 미검증이다
  3. 실제 처리 속도가 얼마인가   <- 18~35시간이라는 추정치에는 아직 근거가 없다

사용법:
    LM Studio에서 모델을 하나 로드하고 로컬 서버를 켠 뒤,
    python tools/probe_lmstudio.py --model <LM Studio에 뜨는 모델 id>

    모델 id를 모르면 --model 없이 돌리면 목록을 보여주고 끝난다.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from typing import Any, Dict, List, Optional

try:
    from openai import OpenAI
except ImportError:
    sys.exit("openai 패키지가 없습니다. source .venv/bin/activate 후 다시 실행하십시오.")

PROMPT = "대한민국의 수도는 어디인가? 한 문장으로 답하십시오."
LONG_PROMPT = (
    "다음 주제로 한국어 설명문을 작성하십시오: 도시의 대중교통 분담률이 도시 규모에 따라 "
    "어떻게 달라질 수 있는지, 가능한 요인을 들어 설명하시오."
)
THINK_TAG_RE = re.compile(r"<think>|</think>|<thinking>|◁think▷", re.IGNORECASE)


def _call(client: OpenAI, model: str, prompt: str, **kwargs: Any):
    return client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=kwargs.pop("max_tokens", 256),
        **kwargs,
    )


def probe_seed(client: OpenAI, model: str) -> None:
    print("\n" + "=" * 68)
    print("[1] seed가 반영되는가  (temperature 0.7 고정, seed만 변경)")
    print("=" * 68)
    print("판정 기준: 같은 seed 두 번 -> 같아야 한다 / 다른 seed -> 달라야 한다.")
    print("둘 다 같으면 seed가 무시되고 있는 것이고, 그러면 3회 반복이 의미를 잃는다.\n")
    try:
        a1 = _call(client, model, PROMPT, temperature=0.7, top_p=0.9, seed=1).choices[0].message.content
        a2 = _call(client, model, PROMPT, temperature=0.7, top_p=0.9, seed=1).choices[0].message.content
        b1 = _call(client, model, PROMPT, temperature=0.7, top_p=0.9, seed=2).choices[0].message.content
    except Exception as exc:
        print(f"  !! 호출 실패: {type(exc).__name__}: {exc}")
        return

    same_seed_identical = a1 == a2
    diff_seed_identical = a1 == b1
    print(f"  같은 seed(1) 두 번 동일:  {same_seed_identical}")
    print(f"  다른 seed(1 vs 2) 동일:   {diff_seed_identical}")
    if same_seed_identical and not diff_seed_identical:
        print("  => seed 정상 반영. 설계대로 반복마다 seed만 바꾸면 된다. OK")
    elif same_seed_identical and diff_seed_identical:
        print("  => !! seed와 무관하게 항상 같은 응답이다. temperature가 무시되고 있을 수 있다.")
        print("        이대로면 3회 반복이 전부 같은 응답이 된다. 설계 재검토 필요.")
    elif not same_seed_identical:
        print("  => !! 같은 seed인데 응답이 다르다. seed가 무시되고 있다.")
        print("        반복은 여전히 변동을 주지만 재현이 불가능해진다. 논문에 그대로 밝힐 것.")


def probe_thinking(client: OpenAI, model: str) -> None:
    print("\n" + "=" * 68)
    print("[2] thinking(추론 모드)을 끄는 방법")
    print("=" * 68)
    print("config.yaml에 generation.thinking: false가 있지만 실제 파라미터명은 미검증이다.")
    print("아래 후보를 하나씩 던져보고, 서버가 받아들이는지 + 추론 흔적이 사라지는지 본다.\n")

    candidates: List[Dict[str, Any]] = [
        {"label": "(대조군) 아무 것도 안 보냄", "kwargs": {}},
        {"label": "extra_body={'thinking': False}", "kwargs": {"extra_body": {"thinking": False}}},
        {"label": "extra_body={'enable_thinking': False}", "kwargs": {"extra_body": {"enable_thinking": False}}},
        {"label": "extra_body={'chat_template_kwargs': {'enable_thinking': False}}",
         "kwargs": {"extra_body": {"chat_template_kwargs": {"enable_thinking": False}}}},
        {"label": "reasoning_effort='none'", "kwargs": {"extra_body": {"reasoning_effort": "none"}}},
        {"label": "reasoning_effort='low'", "kwargs": {"extra_body": {"reasoning_effort": "low"}}},
        {"label": "프롬프트에 /no_think 접미사", "kwargs": {"_prompt_suffix": " /no_think"}},
    ]

    for c in candidates:
        kwargs = dict(c["kwargs"])
        suffix = kwargs.pop("_prompt_suffix", "")
        try:
            resp = _call(client, model, LONG_PROMPT + suffix, temperature=0.7, **kwargs)
            msg = resp.choices[0].message
            text = msg.content or ""
            has_tag = bool(THINK_TAG_RE.search(text))
            # 일부 서버는 추론을 별도 필드로 분리한다
            reasoning_field = None
            for attr in ("reasoning", "reasoning_content"):
                if getattr(msg, attr, None):
                    reasoning_field = attr
                    break
            ct = getattr(resp.usage, "completion_tokens", "?")
            verdict = "추론흔적 있음" if (has_tag or reasoning_field) else "추론흔적 없음"
            extra = f", 별도필드={reasoning_field}" if reasoning_field else ""
            print(f"  [수락] {c['label']:<58} completion_tokens={ct:<5} {verdict}{extra}")
        except Exception as exc:
            print(f"  [거부] {c['label']:<58} {type(exc).__name__}: {str(exc)[:90]}")

    print("\n  읽는 법: [거부]는 그 파라미터를 서버가 안 받는다는 뜻이다.")
    print("  [수락] 중에서 대조군보다 completion_tokens가 뚜렷이 줄고 추론흔적이 사라진 것이 답이다.")
    print("  전부 대조군과 같으면 이 모델은 애초에 추론 모드가 아니거나 끌 수 없는 것이다.")
    print("  !! 어느 것도 확실하지 않으면 config의 thinking 설정을 지우고,")
    print("     '추론 모드를 끄지 못했다'고 논문 한계에 쓰는 편이 낫다. 지어내지 말 것.")


def probe_throughput(client: OpenAI, model: str, n: int) -> None:
    print("\n" + "=" * 68)
    print(f"[3] 처리 속도 실측 ({n}회 호출, max_tokens=512)")
    print("=" * 68)
    lat: List[float] = []
    toks: List[int] = []
    for i in range(n):
        t0 = time.monotonic()
        try:
            r = _call(client, model, LONG_PROMPT, temperature=0.7, top_p=0.9, max_tokens=512, seed=i + 1)
        except Exception as exc:
            print(f"  !! {i+1}번째 호출 실패: {type(exc).__name__}: {exc}")
            return
        dt = time.monotonic() - t0
        lat.append(dt)
        toks.append(getattr(r.usage, "completion_tokens", 0) or 0)
        print(f"  {i+1}/{n}: {dt:6.2f}초, completion_tokens={toks[-1]}")

    mean_lat = sum(lat) / len(lat)
    mean_tok = sum(toks) / len(toks)
    tps = mean_tok / mean_lat if mean_lat else 0.0
    print(f"\n  평균 지연: {mean_lat:.2f}초/응답")
    print(f"  평균 생성: {mean_tok:.0f} 토큰  ->  약 {tps:.1f} tok/s")
    print("\n  이 속도로 추정한 소요 시간 (생성만, judge 별도):")
    for label, n_resp in [
        ("실험 A 축소안  200x5x3x2 = 6,000", 6000),
        ("실험 A 원안    200x7x3x3 = 12,600", 12600),
        ("실험 B         40x4x3x2 =  960", 960),
        ("파일럿          20x7x1x1 =  140", 140),
    ]:
        hours = n_resp * mean_lat / 3600
        print(f"    {label:<34} 약 {hours:6.1f} 시간")
    print("\n  judge까지 더해야 총 시간이다. judge는 응답 1건당 1회 호출된다(votes=1).")
    print("  !! 실험 A 원안이 감당 안 되는 시간이면 조건이나 모델을 줄일 것. 문항을 줄이지 말 것.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://localhost:1234/v1")
    ap.add_argument("--api-key", default="lm-studio")
    ap.add_argument("--model", default=None, help="LM Studio에 로드된 모델 id")
    ap.add_argument("--throughput-n", type=int, default=5)
    args = ap.parse_args()

    client = OpenAI(base_url=args.base_url, api_key=args.api_key, timeout=300.0)

    print("=" * 68)
    print(f"LM Studio 프로브  base_url={args.base_url}")
    print("=" * 68)
    try:
        models = client.models.list()
        ids = [m.id for m in models.data]
    except Exception as exc:
        sys.exit(f"서버에 붙지 못했습니다: {type(exc).__name__}: {exc}\n"
                 "LM Studio에서 Local Server를 켰는지 확인하십시오.")

    print("이 서버에서 쓸 수 있는 모델 (JIT 로딩이 켜져 있으면 로드 안 된 것도 나온다):")
    for i in ids:
        print(f"  - {i}")
    if not args.model:
        print("\n--model <id> 를 붙여 다시 실행하십시오.")
        return
    if args.model not in ids:
        print(f"\n!! '{args.model}'가 목록에 없습니다. 그래도 진행합니다(별칭일 수 있음).")

    probe_seed(client, args.model)
    probe_thinking(client, args.model)
    probe_throughput(client, args.model, args.throughput_n)

    print("\n" + "=" * 68)
    print("끝. 이 출력을 통째로 복사해서 돌려주십시오 — config와 docs에 반영하겠습니다.")
    print("=" * 68)


if __name__ == "__main__":
    main()
