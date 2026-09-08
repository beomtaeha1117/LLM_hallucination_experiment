"""format_ok가 낮은 이유를 찾는다 — 모델의 성향인가, 프롬프트/파서의 버그인가.

파일럿에서 P4/P5의 format_ok가 0.5 부근으로 나왔다. 그 자체가 결과일 수도 있지만
(조건이 형식 준수를 떨어뜨린다), 마커를 쓰긴 썼는데 표기가 미묘하게 달라
파서가 못 잡는 것이라면 그건 버그다. 둘을 갈라놓기 전에는 어느 쪽도 주장할 수 없다.

    python tools/diagnose_format.py results/pilot_001/raw_responses.csv
"""
from __future__ import annotations

import re
import sys
from collections import Counter

import pandas as pd

MARKER = "[최종답변]"

# 파서가 요구하는 정확한 표기에서 조금씩 어긋난 형태들. 이게 많이 잡히면
# 낮은 format_ok는 모델의 성향이 아니라 표기 불일치다.
NEAR_MISS = {
    "공백 삽입 [최종 답변]": re.compile(r"\[\s*최종\s+답변\s*\]"),
    "전각 괄호 【최종답변】": re.compile(r"【\s*최종답변\s*】"),
    "괄호 없음 최종답변:": re.compile(r"(?<!\[)최종\s*답변\s*[:：]"),
    "볼드 **[최종답변]**": re.compile(r"\*\*\s*\[?최종답변\]?\s*\*\*"),
    "영문 Final Answer": re.compile(r"(?i)final\s*answer"),
    "다른 괄호 (최종답변)": re.compile(r"[(（]\s*최종답변\s*[)）]"),
}


def main(path: str) -> None:
    df = pd.read_csv(path, dtype=str, encoding="utf-8-sig")
    df["format_ok"] = df["format_ok"].astype(str).str.lower().isin(["true", "1"])
    print(f"총 {len(df)}행\n")

    for cond, g in df.groupby("prompt_type", sort=False):
        bad = g[~g["format_ok"]]
        rate = g["format_ok"].mean()
        print(f"=== {cond}  format_ok={rate:.3f}  실패 {len(bad)}/{len(g)} ===")
        if bad.empty:
            print()
            continue

        # 1) 실패한 응답이 마커 비슷한 것이라도 썼는가
        hits = Counter()
        for t in bad["response_raw"].fillna(""):
            for name, rx in NEAR_MISS.items():
                if rx.search(t):
                    hits[name] += 1
        if hits:
            print("  마커 유사 표기 (파서가 못 잡는 것):")
            for name, n in hits.most_common():
                print(f"    {n:3d}건  {name}")
        else:
            print("  마커 유사 표기: 없음 — 아예 안 쓴 것으로 보인다")

        # 2) parse_mode 분포와 잘림 여부
        print("  parse_mode:", dict(Counter(bad["parse_mode"].fillna(""))))
        print("  finish_reason:", dict(Counter(bad["finish_reason"].fillna(""))))
        ct = pd.to_numeric(bad["completion_tokens"], errors="coerce")
        print(f"  completion_tokens: 중앙값 {ct.median():.0f}, 최대 {ct.max():.0f}, "
              f"512 도달 {(ct >= 512).sum()}건")

        # 3) 실제로 어떻게 끝나는지 — 눈으로 봐야 아는 것이 있다
        print("  실패 응답 2건의 마지막 200자:")
        for t in bad["response_raw"].fillna("").head(2):
            tail = t[-200:].replace("\n", "\\n")
            print(f"    ...{tail}")
        print()

    # 전체 요약: 잘려서 실패한 것과 형식을 안 지켜서 실패한 것은 원인이 다르다
    bad_all = df[~df["format_ok"]]
    ct = pd.to_numeric(bad_all["completion_tokens"], errors="coerce")
    n_trunc = int((ct >= 512).sum())
    print("=" * 60)
    print(f"전체 실패 {len(bad_all)}건 중 max_tokens(512)에 걸린 것: {n_trunc}건")
    if n_trunc > len(bad_all) * 0.3:
        print("  !! 상당수가 잘려서 실패했다. 형식 문제가 아니라 max_tokens가 모자란 것이므로")
        print("     프롬프트를 고치기 전에 max_tokens를 올려서 다시 볼 것.")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "results/pilot_001/raw_responses.csv")
