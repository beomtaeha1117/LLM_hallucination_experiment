"""LM Studio API 클라이언트와, 네트워크 없이 파이프라인을 검증하기 위한 MockClient."""

from __future__ import annotations

import hashlib
import logging
import random
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional

logger = logging.getLogger("client")


@dataclass
class Completion:
    """LLM 응답 한 건. text는 파싱 전 원문(response_raw)이다."""

    text: str
    prompt_tokens: int
    completion_tokens: int
    latency_ms: float
    finish_reason: str


class LMStudioClient:
    """LM Studio의 OpenAI 호환 `/v1/chat/completions` 엔드포인트 래퍼.

    `openai` 패키지는 클래스 내부에서 지연 임포트한다 — mock 실행 시에는
    이 클래스를 아예 인스턴스화하지 않으므로 openai가 설치되지 않아도 된다.
    """

    def __init__(
        self,
        base_url: str,
        api_key: str,
        timeout_s: float = 180.0,
        max_retries: int = 3,
    ) -> None:
        from openai import OpenAI  # 지연 임포트

        self._client = OpenAI(base_url=base_url, api_key=api_key, timeout=timeout_s)
        self.max_retries = max_retries
        self._reasoning_leak_logged = False

    def complete(
        self,
        *,
        model_key: str,
        lms_id: str,
        prompt_type: str,
        system_prompt: str,
        user_message: str,
        question: Optional[Dict[str, Any]] = None,
        repeat: int = 0,
        seed: int = 0,
        temperature: float = 0.7,
        top_p: float = 0.9,
        max_tokens: int = 512,
        reasoning_effort: Optional[str] = None,
    ) -> Completion:
        """실제 LM Studio 서버에 요청을 보낸다. 지수 백오프로 재시도한다.

        `reasoning_effort`: config의 generation.reasoning_effort 값을 그대로 전달받는다
        (None이면 아무 것도 보내지 않는다). 2026-09-08 프로브에서 qwen3.6-35b-a3b에
        대해 "none"만이 응답에서 reasoning_content를 없앴다 — extra_body의 thinking /
        enable_thinking / chat_template_kwargs 와 프롬프트 /no_think 접미사는 서버가
        받아들이기는 하되(거부되지 않는다) 추론 흔적이 그대로 남았다. 자세한 근거는
        docs/RUNBOOK.md의 프로브 절을 볼 것.
        """

        backoff = 1.0
        last_exc: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                t0 = time.monotonic()
                request_kwargs: Dict[str, Any] = dict(
                    model=lms_id,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_message},
                    ],
                    temperature=temperature,
                    top_p=top_p,
                    max_tokens=max_tokens,
                    seed=seed,
                )
                if reasoning_effort is not None:
                    # reasoning_effort는 프로브로 검증된 유일한 경로다 (2026-09-08,
                    # qwen3.6-35b-a3b). 다만 검증은 그 한 모델에서만 했다 — gemma-4-12b,
                    # gpt-oss-20b에서도 같은지는 아래 누출 검사가 실행 중에 확인한다.
                    request_kwargs["reasoning_effort"] = reasoning_effort
                resp = self._client.chat.completions.create(**request_kwargs)
                latency_ms = (time.monotonic() - t0) * 1000.0
                choice = resp.choices[0]
                text = choice.message.content or ""
                # 조건이 조용히 어긋나는 것을 막는다: reasoning을 껐다고 선언했는데
                # 서버가 여전히 reasoning_content를 돌려주면 그 모델에서는 안 꺼진
                # 것이고, 그대로 두면 통제변인이 모델마다 달라진 채로 실험이 끝난다.
                if (
                    reasoning_effort == "none"
                    and getattr(choice.message, "reasoning_content", None)
                    and not self._reasoning_leak_logged
                ):
                    logger.warning(
                        "%s: reasoning_effort='none'을 보냈는데도 응답에 reasoning_content가 "
                        "남아 있습니다. 이 모델에서는 추론 모드가 꺼지지 않은 것이므로 "
                        "조건이 모델 간에 달라집니다 — 논문 한계에 반드시 적을 것.",
                        lms_id,
                    )
                    self._reasoning_leak_logged = True
                usage = resp.usage
                prompt_tokens = getattr(usage, "prompt_tokens", 0) if usage else 0
                completion_tokens = getattr(usage, "completion_tokens", 0) if usage else 0
                finish_reason = choice.finish_reason or "unknown"
                return Completion(
                    text=text,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    latency_ms=latency_ms,
                    finish_reason=finish_reason,
                )
            except Exception as exc:  # noqa: BLE001 - 재시도 후 마지막에 재발생
                last_exc = exc
                if attempt < self.max_retries - 1:
                    time.sleep(backoff)
                    backoff *= 2
        assert last_exc is not None
        raise last_exc


# ---------------------------------------------------------------------------
# Mock client
# ---------------------------------------------------------------------------

_ABSTAIN_LEXICON_PHRASES = [
    "확인할 수 없습니다. 관련 근거를 찾지 못했습니다.",
    "해당 내용은 자료가 없어 답변드리기 어렵습니다.",
    "제공된 정보만으로는 알 수 없습니다.",
    "그러한 사실은 확인되지 않습니다.",
    "질문에 포함된 전제가 잘못된 것으로 보입니다.",
    "관련 문헌을 찾을 수 없어 답변할 수 없습니다.",
]
_ABSTAIN_UNUSUAL_PHRASES = [
    "제가 답변드릴 만한 근거를 가지고 있지 않습니다.",
    "이 부분은 제 지식 범위 밖이라 말씀드리기 조심스럽습니다.",
    "확신을 가지고 말씀드릴 수 있는 부분이 아닙니다.",
]

# 모델별로 방향은 같지만 크기가 다른 소폭 오프셋 (연구질문 5: 재현성 검증용)
_MODEL_OFFSETS = {
    "qwen35": {"correct": 0.03, "abstain_n": 0.02, "over_abstain": -0.01},
    "gemma4": {"correct": -0.02, "abstain_n": -0.03, "over_abstain": 0.02},
    "gptoss20": {"correct": 0.00, "abstain_n": 0.00, "over_abstain": 0.00},
}

_THREE_SECTION = {"P2", "P3", "P2L"}


def _stable_rng(model_key: str, prompt_type: str, question_id: str, repeat: int) -> random.Random:
    key = f"{model_key}|{prompt_type}|{question_id}|{repeat}"
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
    seed_int = int(digest[:16], 16)
    return random.Random(seed_int)


def _clip01(p: float) -> float:
    return max(0.0, min(1.0, p))


class MockClient:
    """네트워크 없이 결정론적으로 동작하는 가짜 LLM 클라이언트.

    seed는 (model_key, prompt_type, question_id, repeat)의 안정적 해시로부터
    파생되므로, 같은 config로 재실행하면 완전히 같은 결과가 나온다(재현성 검증용).
    """

    def complete(
        self,
        *,
        model_key: str,
        lms_id: str,
        prompt_type: str,
        system_prompt: str,
        user_message: str,
        question: Dict[str, Any],
        repeat: int,
        seed: int,
        temperature: float = 0.7,
        top_p: float = 0.9,
        max_tokens: int = 512,
        reasoning_effort: Optional[str] = None,
    ) -> Completion:
        # reasoning_effort는 MockClient에서는 아무 효과가 없다 — 실제 클라이언트와
        # 호출 시그니처를 맞추기 위해서만 받는다.
        rng = _stable_rng(model_key, prompt_type, str(question["question_id"]), repeat)

        uncertainty = prompt_type in {"P1", "P3"}
        verification = prompt_type in {"P2", "P3"}
        three_section = prompt_type in _THREE_SECTION

        offsets = _MODEL_OFFSETS.get(model_key, {"correct": 0.0, "abstain_n": 0.0, "over_abstain": 0.0})

        t0 = time.monotonic()

        # 형식 오류(마커 없음) 확률: 3섹션 조건이 더 자주 형식을 어긴다.
        malformed_p = 0.09 if three_section else 0.04
        is_malformed = rng.random() < malformed_p

        answerable = question["answerable"]
        text: str

        if answerable == "N":
            # abstain(=전제 지적/보류) 확률. 연구설계 §research-design.md 시뮬레이션 규칙:
            # baseline 0.25, +0.35 uncertainty, +0.10 verification, -0.05 상호작용 패널티
            p_abstain = 0.25
            if uncertainty:
                p_abstain += 0.35
            if verification:
                p_abstain += 0.10
            if uncertainty and verification:
                p_abstain -= 0.05  # P3는 단순 가산이 아님 (상호작용 항 검정 대상)
            if prompt_type == "P5":
                # few-shot 조건. 아래 수치도 파이프라인 점검용 임의값이다.
                p_abstain += 0.22
            if prompt_type == "P4":
                # 순수 금지 조건. 이 값은 파이프라인을 돌리기 위한 임의 수치일 뿐이며
                # "금지만 하면 효과가 작다"는 가설을 데이터로 미리 정해두는 것이 아니다.
                p_abstain += 0.08
            p_abstain += offsets["abstain_n"]
            p_abstain = _clip01(p_abstain)

            if rng.random() < p_abstain:
                text_body = self._abstain_text(rng, question)
            else:
                text_body = self._fabricate_text(rng, question)
        else:
            p_correct = 0.70
            if verification:
                p_correct += 0.04
            if uncertainty:
                p_correct -= 0.03
            p_correct += offsets["correct"]

            p_over_abstain = 0.05
            if uncertainty:
                p_over_abstain += 0.18
            p_over_abstain += offsets["over_abstain"]

            p_correct = _clip01(p_correct)
            p_over_abstain = _clip01(p_over_abstain)
            # 두 확률의 합이 1을 넘지 않도록 정규화
            if p_correct + p_over_abstain > 1.0:
                scale = 1.0 / (p_correct + p_over_abstain)
                p_correct *= scale
                p_over_abstain *= scale

            roll = rng.random()
            if roll < p_correct:
                text_body = self._correct_answer_text(rng, question)
            elif roll < p_correct + p_over_abstain:
                text_body = self._abstain_text(rng, question)
            else:
                text_body = self._wrong_answer_text(rng, question)

        text = self._wrap_sections(rng, prompt_type, text_body, is_malformed)

        completion_tokens = max(5, len(text) // 2 + rng.randint(-5, 5))
        prompt_tokens = max(10, len(system_prompt + user_message) // 3 + rng.randint(-3, 3))
        latency_ms = completion_tokens * rng.uniform(18.0, 32.0)
        # t0를 사용해 아주 약간의 실제 경과시간도 섞는다 (호출부 일관성용, 미미함)
        latency_ms += (time.monotonic() - t0) * 1000.0

        return Completion(
            text=text,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            latency_ms=latency_ms,
            finish_reason="stop",
        )

    # -- 텍스트 생성 헬퍼 -------------------------------------------------

    def _correct_answer_text(self, rng: random.Random, question: Dict[str, Any]) -> str:
        gt = question.get("ground_truth", "").strip()
        acceptable = question.get("acceptable_answers", "")
        candidates = [a for a in acceptable.split("|") if a.strip()] or [gt]
        chosen = rng.choice(candidates) if candidates else gt
        templates = [
            f"{chosen}입니다.",
            f"정답은 {chosen}입니다.",
            f"{chosen}로 알려져 있습니다.",
        ]
        return rng.choice(templates)

    def _wrong_answer_text(self, rng: random.Random, question: Dict[str, Any]) -> str:
        # ground_truth/acceptable_answers 문자열을 포함하지 않는 그럴듯한 오답을 만든다.
        fillers = [
            "다른 값으로 알려져 있으며, 통념과는 차이가 있습니다.",
            "이는 흔히 알려진 것과 다른 수치로 기록되어 있습니다.",
            "관련 자료에 따르면 상당히 다른 내용이 확인됩니다.",
            "일반적으로 알려진 것과 상이한 결과가 보고된 바 있습니다.",
        ]
        return rng.choice(fillers)

    def _fabricate_text(self, rng: random.Random, question: Dict[str, Any]) -> str:
        # answerable == N 문항에 대해 존재하지 않는 내용을 지어내는 응답을 흉내낸다.
        fillers = [
            "해당 내용의 핵심 결론은 기존 통념을 뒤집는 새로운 발견으로 요약됩니다.",
            "관련 기록에 따르면 세부 수치와 절차가 구체적으로 명시되어 있습니다.",
            "이 사안에 대해서는 명확한 공식 결론이 존재하며, 다음과 같이 정리됩니다.",
            "해당 인물/자료는 여러 문헌에서 구체적으로 다루어진 바 있습니다.",
        ]
        return rng.choice(fillers)

    def _abstain_text(self, rng: random.Random, question: Dict[str, Any]) -> str:
        if rng.random() < 0.15:
            return rng.choice(_ABSTAIN_UNUSUAL_PHRASES)
        return rng.choice(_ABSTAIN_LEXICON_PHRASES)

    def _wrap_sections(
        self, rng: random.Random, prompt_type: str, final_body: str, is_malformed: bool
    ) -> str:
        if is_malformed:
            # 마커 없이 뚝 끊긴 것 같은 텍스트만 반환 (format_ok=False 유발)
            return final_body[:1] and (
                "생각을 정리하는 중입니다... " + final_body
            )

        if prompt_type in {"P0", "P1", "P4", "P5"}:
            return f"[최종답변]\n{final_body}"

        if prompt_type in {"P2", "P3"}:
            draft = "우선 알고 있는 내용을 바탕으로 답을 작성하면: " + final_body
            review = "위 초안의 사실 주장을 다시 점검했습니다. 큰 오류는 없다고 판단됩니다."
            return f"[초안]\n{draft}\n\n[검토]\n{review}\n\n[최종답변]\n{final_body}"

        # P2L
        restated = "질문은 특정 사실 또는 정보를 확인해달라는 요청으로 이해됩니다."
        background = "이 주제는 관련된 배경지식을 바탕으로 살펴볼 필요가 있습니다."
        return f"[재진술]\n{restated}\n\n[배경]\n{background}\n\n[최종답변]\n{final_body}"
