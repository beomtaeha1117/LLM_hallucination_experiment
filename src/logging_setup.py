"""로깅 소음 억제를 한 곳에 둔다.

openai의 HTTP 계층이 호출마다 INFO로 "HTTP Request: POST ... 200 OK"를 찍어서
진행 막대를 매번 밀어버린다. run_experiment에만 넣었더니 judge_on_sheet에서
그대로 재현됐다 — 그래서 공용 모듈로 뺀다.

setLevel로 로거 이름을 못 박지 않는 이유가 둘이다. openai 3.x는 httpx가 아니라
httpx2/httpcore2를 쓰므로 이름을 찍으면 빗나가고("httpx"만 껐다가 전부 그대로
찍힌 적이 있다), 그 로거들은 첫 요청 때 만들어지므로 임포트 시점에는 아직 없다.
핸들러 필터는 로거가 언제 생기든, 이름이 무엇으로 바뀌든 걸린다.
"""

from __future__ import annotations

import logging

NOISY_PREFIXES = ("httpx", "httpcore", "openai", "urllib3")


class DropNoisyHTTP(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return not (
            record.name.startswith(NOISY_PREFIXES) and record.levelno < logging.WARNING
        )


def quiet_http_logs() -> None:
    """루트 핸들러에 필터를 건다. 경고·오류는 그대로 통과한다."""
    f = DropNoisyHTTP()
    for h in logging.getLogger().handlers:
        if not any(isinstance(x, DropNoisyHTTP) for x in h.filters):
            h.addFilter(f)
