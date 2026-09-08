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


# 실패 응답에 실제로 등장한 대괄호 토큰을 센다. 미리 떠올린 후보 목록만으로는
# 놓친다는 것을 pilot_002에서 확인했다.
BRACKET_RE = re.compile(r"\[([^\[\]\n]{1,20})\]")


def _looks_like_marker(tok: str) -> bool:
    """[최종답변]에서 한두 글자 어긋난 것을 마커 오타로 본다."""
    target = "최종답변"
    t = tok.replace(" ", "")
    if t == target:
        return False  # 정확히 맞으면 파서가 이미 잡았을 것이다
    if len(t) != len(target):
        return False
    return sum(a != b for a, b in zip(t, target)) <= 2


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

        # 위 목록은 내가 미리 떠올린 형태만 잡는다. pilot_002에서 모델이 쓴 것은
        # "[최정답변]"이었고(최종 -> 최정 오타) 목록에 없어서 "유사 표기 없음"으로
        # 잘못 보고됐다. 그래서 떠올리는 대신 실제로 나온 대괄호 토큰을 전부 센다.
        seen = Counter()
        for t in bad["response_raw"].fillna(""):
            for tok in BRACKET_RE.findall(t):
                seen[tok.strip()] += 1
        if seen:
            print("  실패 응답에 실제로 나온 대괄호 토큰:")
            for tok, n in seen.most_common(8):
                flag = "  <- 마커 오타로 보인다" if _looks_like_marker(tok) else ""
                print(f"    {n:3d}건  [{tok}]{flag}")
        else:
            print("  대괄호 토큰이 하나도 없다 — 형식을 아예 안 쓴 것이다")

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

    # 조건이 다른데 실패율이 똑같이 나오면, 원인이 조건이 아니라 문항일 수 있다.
    # 실제로 pilot_002에서 P0/P1/P4/P5가 전부 0.741(=20/27)로 같았다.
    fails = {c: set(g[~g["format_ok"]]["question_id"]) for c, g in df.groupby("prompt_type")}
    conds = [c for c in fails if fails[c]]
    if len(conds) >= 2:
        common = set.intersection(*(fails[c] for c in conds))
        union = set.union(*(fails[c] for c in conds))
        print("=" * 60)
        print("조건 간 실패 문항 겹침")
        print(f"  실패가 있는 조건 {len(conds)}개, 실패 문항 합집합 {len(union)}개")
        print(f"  모든 조건에서 실패한 문항: {len(common)}개")
        if common:
            print("  -> 조건과 무관하게 실패하는 문항이다. 원인은 프롬프트가 아니라 문항 쪽이다:")
            sub = df[df["question_id"].isin(common)].drop_duplicates("question_id")
            for _, r in sub.iterrows():
                q = str(r.get("question", ""))[:50].replace("\n", " ")
                print(f"     {r['question_id']:8} [{r.get('question_type','')}] {q}")
        else:
            print("  -> 겹치는 문항이 없다. 조건별로 다른 문항이 실패하고 있다.")
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
