# -*- coding: utf-8 -*-
"""지문 정의(domains_*.py)에서 data/questions_b.csv를 만든다.

근거 오프셋을 손으로 적지 않고 근거 문장을 찾아 계산하는 이유: 지문을 한 글자만
고쳐도 손으로 적은 오프셋은 조용히 틀어진다. 문장을 못 찾으면 여기서 멈춘다.

    python data/b_src/build_questions_b.py
"""
from __future__ import annotations

import csv
import importlib.util
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.schema import QUESTION_COLUMNS  # noqa: E402

COLUMNS = QUESTION_COLUMNS + [
    "doc_a", "doc_b", "evidence_doc", "evidence_char_offset", "evidence_para", "b_type",
]


def _load(name: str):
    path = pathlib.Path(__file__).with_name(name)
    spec = importlib.util.spec_from_file_location(path.stem, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.DOMAINS


def _para_index(doc: str, offset: int) -> int:
    """오프셋이 몇 번째 문단(1부터)에 있는지."""
    pos, idx = 0, 1
    for para in doc.split("\n\n"):
        if pos + len(para) >= offset:
            return idx
        pos += len(para) + 2
        idx += 1
    return idx - 1


def main() -> int:
    domains = _load("domains_01_05.py") + _load("domains_06_10.py")
    rows = []
    for d in domains:
        for q in d["questions"]:
            qid = f"{d['id']}-{q['suffix']}"
            answerable = q["answerable"]
            if answerable == "Y":
                doc = d["doc_a"] if q["doc"] == 1 else d["doc_b"]
                off = doc.find(q["ev"])
                if off < 0:
                    print(f"!! {qid}: 근거 문장을 자료{q['doc']}에서 찾지 못했습니다.")
                    return 1
                ev_doc, ev_off, ev_para = str(q["doc"]), str(off), str(_para_index(doc, off))
                gt, acc, why = q["gt"], q.get("acc", ""), ""
                # 실험 A의 거친 유형 목록을 그대로 쓴다 — 분석 코드가 이 값을 본다.
                qtype = "context_qa"
                b_type = f"answerable_doc{q['doc']}"
            else:
                ev_doc, ev_off, ev_para = "none", "", ""
                gt, acc, why = "", "", q["why"]
                qtype = "context_qa_nogold"
                b_type = f"unanswerable_{q['kind']}"
            rows.append({
                "question_id": qid, "question_type": qtype, "question_subtype": "",
                "answerable": answerable, "question": q["q"], "context": "",
                "ground_truth": gt, "acceptable_answers": acc, "why_unanswerable": why,
                # 허구 세계라 외부 검증이 필요 없다. 검증 항목은 지문의 내적 일관성뿐이고,
                # 그건 tools/validate_questions_b.py가 본다(기준 문서 §3.1).
                "verify_action": "허구 지문 — 외부 검증 불필요, 내적 일관성만 확인",
                "verified": "YES",
                "doc_a": d["doc_a"], "doc_b": d["doc_b"],
                "evidence_doc": ev_doc, "evidence_char_offset": ev_off,
                "evidence_para": ev_para, "b_type": b_type,
            })

    out = ROOT / "data" / "questions_b.csv"
    with open(out, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)
    print(f"{out}  {len(rows)}행")
    from collections import Counter
    print("b_type:", dict(Counter(r["b_type"] for r in rows)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
