"""실험 B — 같은 지시를 입력의 어디에 넣느냐(FRONT/MID/BACK/NOINSTR).

기준 문서는 docs/experiment-b-position.md다. 코드가 어긋나면 코드를 고친다.

실험 A와 섞이지 않게 별도 모듈로 둔다. A는 지시의 **내용**을 바꾸고(P0~P5,
system 프롬프트), B는 지시의 **자리**만 바꾼다(user 메시지 안에서). system
프롬프트는 4조건 모두 P0_baseline으로 동일하다 — 그래야 조작이 위치 하나로
남는다.
"""

from __future__ import annotations

import os
from typing import Dict, Tuple

POSITIONS = ["FRONT", "MID", "BACK", "NOINSTR"]
INSTRUCTION_ID = "B-INSTR-01"
_INSTRUCTION_PATH = "prompts/_b_instruction.txt"

# 지문과 지시를 마커로 가른다. MID 지시가 자료의 일부로 읽히면 실험이 무효다.
_DOC_A = "<자료 1>\n{doc_a}\n</자료 1>"
_DOC_B = "<자료 2>\n{doc_b}\n</자료 2>"
_INSTR = "[지시]\n{instruction}\n[/지시]"
_QUESTION = "[질문]\n{question}"


def load_instruction(path: str = _INSTRUCTION_PATH) -> str:
    if not os.path.exists(path):
        raise FileNotFoundError(f"실험 B 지시문이 없습니다: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()


def build_user_message(
    position: str, question: Dict, instruction: str
) -> Tuple[str, int]:
    """조건별 user 메시지와, 그 안에서 지시가 시작하는 문자 인덱스를 돌려준다.

    질문은 **항상 마지막**이다. BACK의 '뒤'는 지문 뒤·질문 앞이지 메시지의
    절대 끝이 아니다(기준 문서 §1).

    문자 인덱스를 같이 주는 이유: FRONT/MID/BACK이라는 이름이 아니라 실제
    프롬프트 안 위치가 조작된 것이므로, 조건별로 그 값이 갈리는지 데이터에서
    확인할 수 있어야 한다. 토큰 인덱스는 모델 토크나이저가 있어야 정확하므로
    tools/probe_b_position.py에서 실제 서버로 따로 잰다(기준 문서 §5, §9).
    """
    if position not in POSITIONS:
        raise ValueError(f"알 수 없는 position: {position} (가능: {POSITIONS})")

    doc_a = _DOC_A.format(doc_a=str(question.get("doc_a", "")).strip())
    doc_b = _DOC_B.format(doc_b=str(question.get("doc_b", "")).strip())
    instr = _INSTR.format(instruction=instruction)
    q = _QUESTION.format(question=str(question.get("question", "")).strip())

    if position == "FRONT":
        parts = [instr, doc_a, doc_b, q]
    elif position == "MID":
        parts = [doc_a, instr, doc_b, q]
    elif position == "BACK":
        parts = [doc_a, doc_b, instr, q]
    else:  # NOINSTR — 지시가 없다. 위치 요인의 네 번째 수준이 아니다.
        parts = [doc_a, doc_b, q]

    message = "\n\n".join(parts)
    instr_char_index = message.find(instr) if position != "NOINSTR" else -1
    return message, instr_char_index


def position_metadata(position: str, question: Dict, message: str, instr_char_index: int) -> Dict:
    """응답 행에 남길 B 전용 컬럼 (기준 문서 §5)."""
    doc_a = str(question.get("doc_a", ""))
    doc_b = str(question.get("doc_b", ""))
    return {
        "position": position,
        "instruction_id": INSTRUCTION_ID if position != "NOINSTR" else "",
        "evidence_doc": str(question.get("evidence_doc", "") or "none"),
        "evidence_char_offset": str(question.get("evidence_char_offset", "")),
        "doc_a_chars": len(doc_a),
        "doc_b_chars": len(doc_b),
        "instr_char_index": instr_char_index,
        # 비율은 조건이 실제로 갈리는지 한눈에 보기 위한 것이다. 토큰 인덱스의
        # 대용이지 대체물이 아니다 — 토큰 쪽은 probe_b_position.py로 확인한다.
        "instr_char_ratio": round(instr_char_index / len(message), 4) if instr_char_index >= 0 else "",
    }
