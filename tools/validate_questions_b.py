"""실험 B 문항 파일을 기준 문서(docs/experiment-b-position.md §3)에 맞춰 검증한다.

여기서 걸러야 할 것은 오타가 아니라 **실험을 무효로 만드는 불균형**이다.
근거가 자료1에 쏠리면 "지시 위치 효과"라고 보고한 것 안에 "근거 위치 효과"가
섞이고, 지문이 짧으면 앞·중간·뒤가 사실상 같은 자리가 된다.

    python tools/validate_questions_b.py data/questions_b.csv
"""
from __future__ import annotations

import sys

import pandas as pd

MIN_CHARS, MAX_CHARS = 420, 480
PARAGRAPHS = 3


def main(path: str) -> int:
    df = pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    errs, warns = [], []
    print(f"=== 실험 B 문항 검증: {path} ===\n총 {len(df)}행\n")

    need = ["question_id", "doc_a", "doc_b", "question", "answerable",
            "evidence_doc", "evidence_char_offset", "evidence_para"]
    miss = [c for c in need if c not in df.columns]
    if miss:
        print(f"필수 컬럼 없음: {miss}")
        return 1

    for _, r in df.iterrows():
        qid, a, b = r["question_id"], r["doc_a"], r["doc_b"]
        for name, doc in (("doc_a", a), ("doc_b", b)):
            n = len(doc)
            if not (MIN_CHARS <= n <= MAX_CHARS):
                errs.append(f"{qid} {name} 길이 {n}자 (요구 {MIN_CHARS}~{MAX_CHARS})")
            paras = [p for p in doc.split("\n\n") if p.strip()]
            if len(paras) != PARAGRAPHS:
                errs.append(f"{qid} {name} 문단 {len(paras)}개 (요구 {PARAGRAPHS})")
        if a and b:
            diff = abs(len(a) - len(b)) / max(len(a), len(b))
            if diff > 0.10:
                errs.append(f"{qid} 두 자료 길이 차 {diff*100:.1f}% (요구 10% 이내)")

        ans = str(r["answerable"]).strip()
        ev = str(r["evidence_doc"]).strip()
        if ans == "Y":
            if ev not in ("1", "2"):
                errs.append(f"{qid} answerable=Y인데 evidence_doc={ev!r} (1 또는 2여야 함)")
            else:
                # 근거가 실제로 그 자료 안에 있는지 오프셋으로 확인한다
                doc = a if ev == "1" else b
                try:
                    off = int(str(r["evidence_char_offset"]).strip())
                    if not (0 <= off < len(doc)):
                        errs.append(f"{qid} evidence_char_offset {off}이 자료{ev} 범위(0~{len(doc)-1}) 밖")
                except ValueError:
                    errs.append(f"{qid} evidence_char_offset이 정수가 아님: {r['evidence_char_offset']!r}")
        elif ans == "N":
            if ev not in ("", "none"):
                errs.append(f"{qid} answerable=N인데 evidence_doc={ev!r} (none이어야 함)")
        else:
            errs.append(f"{qid} answerable={ans!r} (Y 또는 N)")

    # 🚨 10:10 균형. 이게 깨지면 지시 위치 효과에 근거 위치 효과가 섞인다.
    y = df[df["answerable"].str.strip() == "Y"]
    n = df[df["answerable"].str.strip() == "N"]
    c1 = int((y["evidence_doc"].str.strip() == "1").sum())
    c2 = int((y["evidence_doc"].str.strip() == "2").sum())
    print(f"answerable Y={len(y)} / N={len(n)}   (요구 20/20)")
    print(f"근거 위치  자료1={c1} / 자료2={c2}   (요구 10/10)\n")
    if len(y) != 20 or len(n) != 20:
        errs.append(f"answerable 분포 {len(y)}/{len(n)} (요구 20/20)")
    if c1 != c2:
        errs.append(f"근거 위치 불균형 {c1}:{c2} — 위치 효과에 근거 위치가 섞인다")

    dup = df[df["question_id"].duplicated()]["question_id"].tolist()
    if dup:
        errs.append(f"question_id 중복: {dup}")

    if warns:
        print("경고:")
        for w in warns:
            print(f"  {w}")
    if errs:
        print(f"오류 {len(errs)}건:")
        for e in errs:
            print(f"  {e}")
        return 1
    print("검증 통과.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "data/questions_b.csv"))
