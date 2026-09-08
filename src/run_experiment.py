"""응답 생성 스테이지: 모델 x 조건 x 문항 x 반복을 순회하며 results/raw_responses.csv를 만든다."""

from __future__ import annotations

import argparse
import csv
import glob
import io
import json
import logging
import os
import shutil
import time
from collections import Counter
from datetime import datetime, timezone
from typing import Dict, Optional, Set, Tuple

import pandas as pd
import yaml
from tqdm import tqdm

from src.client import LMStudioClient, MockClient
from src.schema import RAW_COLUMNS, load_questions, parse_final_answer

logger = logging.getLogger("run_experiment")
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
# openai의 HTTP 계층이 호출마다 INFO로 "HTTP Request: POST ... 200 OK"를 찍는다. 응답
# 하나당 한 줄이라 진행 막대가 매번 밀려서 실행을 지켜볼 수가 없고, 로그 파일도 그 줄로
# 뒤덮인다. 실패는 예외로 올라오고 재시도는 client.py가 따로 경고하므로 없어도 된다.
#
# setLevel로 끄지 않는 이유가 둘이다. openai 3.x는 httpx가 아니라 httpx2/httpcore2를
# 쓰므로 이름을 못 박으면 빗나가고("httpx"만 껐다가 전부 그대로 찍힌 적이 있다),
# 그 로거들은 첫 요청 때 만들어지므로 임포트 시점에 이름을 훑어봐야 아직 없다.
# 핸들러에 필터를 걸면 로거가 언제 생기든, 이름이 무엇으로 바뀌든 걸린다.
_NOISY_PREFIXES = ("httpx", "httpcore", "openai", "urllib3")


class _DropNoisyHTTP(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return not (
            record.name.startswith(_NOISY_PREFIXES) and record.levelno < logging.WARNING
        )


for _h in logging.getLogger().handlers:
    _h.addFilter(_DropNoisyHTTP())


def _load_prompt(prompt_type: str) -> str:
    """prompts/<조건이름>_*.txt 를 조건 이름으로 찾아 읽는다.

    조건 목록은 config.yaml이 정하고, 프롬프트 파일은 파일명 규약으로 찾는다.
    조건을 추가할 때 코드를 고치지 않아도 되게 하기 위한 것이다.
    """
    matches = sorted(glob.glob(f"prompts/{prompt_type}_*.txt"))
    if not matches:
        raise FileNotFoundError(
            f"조건 '{prompt_type}'의 프롬프트 파일이 없습니다. "
            f"prompts/{prompt_type}_<설명>.txt 를 만드십시오."
        )
    if len(matches) > 1:
        raise ValueError(
            f"조건 '{prompt_type}'에 해당하는 프롬프트 파일이 여러 개입니다: {matches}"
        )
    with open(matches[0], "r", encoding="utf-8") as f:
        return f.read()


def _build_user_message(question: pd.Series) -> str:
    context = str(question.get("context", "") or "").strip()
    q_text = str(question["question"]).strip()
    if context:
        return f"[문서]\n{context}\n\n[질문]\n{q_text}"
    return q_text


def _load_done_keys(raw_path: str) -> Set[Tuple]:
    """이미 완료된 (run_id, model_key, prompt_type, question_id, repeat, is_mock) 키 집합을 반환한다.

    is_mock을 키에 포함하는 것은 defect 8의 방어적 조치다. raw_path가 이제
    run_id별로 분리되어 있어 구조적으로는 섞일 수 없지만(defect 7의
    .run_meta.json 검사가 이미 막는다), resume 로직 자체도 이중으로 안전하게
    만들어 둔다.
    """
    if not os.path.exists(raw_path):
        return set()
    done: Set[Tuple] = set()
    try:
        existing = pd.read_csv(raw_path, dtype=str, encoding="utf-8-sig")
    except pd.errors.EmptyDataError:
        return set()
    except pd.errors.ParserError:
        # 전원이 행을 쓰는 도중에 끊기면 마지막 줄이 잘린 채 남는다. 응답 텍스트에
        # 줄바꿈과 쉼표가 있어 따옴표 안에서 잘리면 파일 전체 파싱이 실패하고,
        # 그러면 resume이 아니라 재시작 자체가 막힌다. 잘린 꼬리만 잘라내고 잇는다.
        existing = _recover_truncated_csv(raw_path)
        if existing is None:
            raise
    for _, row in existing.iterrows():
        # 전원이 끊겨 잘린 마지막 행은 pandas가 오류 없이 통과시키기도 한다(모자란
        # 칸을 NaN으로 채운다). 그걸 완료로 세면 그 응답은 resume이 영영 건너뛰고
        # 데이터가 조용히 한 칸 빈다 — 로그에도 안 남는다. timestamp는 행의 마지막
        # 칸이므로, 그것까지 있어야 그 행이 끝까지 쓰인 것이다.
        if any(
            pd.isna(row.get(col)) or str(row.get(col)).strip() == ""
            for col in ("run_id", "model_key", "prompt_type", "question_id", "repeat", "is_mock", "timestamp")
        ):
            logger.warning(
                "resume: 끝까지 쓰이지 않은 행을 발견해 미완료로 취급합니다 "
                "(question_id=%s). 이번 실행에서 다시 생성됩니다.",
                row.get("question_id"),
            )
            continue
        key = (
            row["run_id"],
            row["model_key"],
            row["prompt_type"],
            row["question_id"],
            str(row["repeat"]),
            str(row["is_mock"]),
        )
        done.add(key)
    return done


def _recover_truncated_csv(raw_path: str) -> Optional[pd.DataFrame]:
    """끝이 잘린 raw_responses.csv에서 마지막 온전한 행까지만 살려서 돌려준다.

    잘린 꼬리는 원본을 `.truncated-<타임스탬프>` 로 백업한 뒤 파일에서 잘라낸다.
    잘라낸 행은 resume이 미완료로 보고 다시 생성하므로 데이터 손실은 없다.
    """
    with open(raw_path, "r", encoding="utf-8-sig", newline="") as f:
        text = f.read()
    lines = text.splitlines(keepends=True)
    for drop in range(1, min(len(lines), 50) + 1):
        candidate = "".join(lines[:-drop])
        if not candidate.strip():
            return None
        try:
            df = pd.read_csv(io.StringIO(candidate), dtype=str)
        except (pd.errors.ParserError, pd.errors.EmptyDataError):
            continue
        backup = f"{raw_path}.truncated-{time.strftime('%Y%m%d-%H%M%S')}"
        shutil.copy2(raw_path, backup)
        with open(raw_path, "w", encoding="utf-8", newline="") as f:
            f.write(candidate)
        logger.warning(
            "%s의 마지막 %d줄이 잘려 있었습니다(전원 차단으로 보입니다). "
            "원본을 %s에 백업하고 잘린 꼬리를 잘라냈습니다 — 해당 응답은 이번 "
            "실행에서 다시 생성됩니다.",
            raw_path, drop, backup,
        )
        return df
    return None


def _check_and_record_run_meta(meta_path: str, run_id: str, is_mock: bool) -> None:
    """run_id 디렉터리의 .run_meta.json으로 mock/real 혼입을 막는다 (defect 7).

    처음 만들어지는 run_id면 현재 is_mock 값을 기록한다. 이미 기록이 있는데
    현재 config의 is_mock과 다르면, 아무것도 쓰기 전에(=클라이언트를 만들기도
    전에) RuntimeError로 즉시 중단한다.
    """
    if os.path.exists(meta_path):
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        existing_is_mock = meta.get("is_mock")
        if existing_is_mock != is_mock:
            raise RuntimeError(
                f"run_id '{run_id}'는 이미 is_mock={existing_is_mock}로 기록되어 있는데, "
                f"현재 config는 mock={is_mock}입니다. mock 데이터와 real 데이터가 같은 "
                f"디렉터리(results/{run_id}/)에 섞이면 안 되므로 실행을 중단합니다. "
                f"새로운 run_id를 사용하십시오."
            )
    else:
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump({"run_id": run_id, "is_mock": is_mock}, f, ensure_ascii=False, indent=2)


def run(config_path: str) -> None:
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    run_id = config["run_id"]
    is_mock = bool(config.get("mock", False))

    # defect 7/8: mock/real 혼입 방지 검사를 그 무엇보다(질문 로드, 클라이언트
    # 생성보다) 먼저 한다 — 어긋나면 네트워크 호출은커녕 아무 파일도 건드리기 전에
    # 죽어야 한다.
    raw_dir = f"results/{run_id}"
    raw_path = f"{raw_dir}/raw_responses.csv"
    meta_path = f"{raw_dir}/.run_meta.json"
    os.makedirs(raw_dir, exist_ok=True)
    _check_and_record_run_meta(meta_path, run_id, is_mock)

    questions_file = config["questions_file"]
    conditions = config["conditions"]
    models = config["models"]
    repeats = config["repeats"]
    seeds = config["seeds"]
    gen_cfg = config["generation"]

    questions_df = load_questions(questions_file)
    logger.info("문항 %d개 로드 완료 (%s)", len(questions_df), questions_file)

    done_keys = _load_done_keys(raw_path)
    logger.info("이미 완료된 응답 %d건 발견 (resume)", len(done_keys))

    if is_mock:
        client = MockClient()
        logger.info("MOCK 모드: MockClient 사용, 네트워크 호출 없음")
    else:
        server_cfg = config["server"]
        client = LMStudioClient(
            base_url=server_cfg["base_url"],
            api_key=server_cfg.get("api_key", "lm-studio"),
            timeout_s=server_cfg.get("timeout_s", 180),
            max_retries=server_cfg.get("max_retries", 3),
        )

    prompt_texts = {c: _load_prompt(c) for c in conditions}

    total_planned = len(models) * len(conditions) * len(questions_df) * repeats
    written = 0
    skipped = 0
    format_ok_counter: Dict[str, Counter] = {c: Counter() for c in conditions}
    # 잘림은 format_ok 실패로 나타나지만 원인이 전혀 다르다(형식 불이행이 아니라
    # max_tokens 부족). 파일럿에서 둘을 갈라내는 데 별도 도구가 필요했으므로
    # 요약에 같이 띄운다.
    truncated_counter: Dict[str, int] = {c: 0 for c in conditions}

    file_exists = os.path.exists(raw_path) and os.path.getsize(raw_path) > 0
    csv_file = open(raw_path, "a", newline="", encoding="utf-8")
    writer = csv.DictWriter(csv_file, fieldnames=RAW_COLUMNS)
    if not file_exists:
        writer.writeheader()
        csv_file.flush()

    pbar = tqdm(total=total_planned, desc="responses")
    try:
        for model in models:  # 모델이 가장 바깥 루프 — 모델 로드/스와핑 비용 최소화
            model_key = model["key"]
            for prompt_type in conditions:
                system_prompt = prompt_texts[prompt_type]
                for _, question in questions_df.iterrows():
                    question_id = question["question_id"]
                    user_message = _build_user_message(question)
                    for repeat in range(repeats):
                        key = (run_id, model_key, prompt_type, question_id, str(repeat), str(is_mock))
                        if key in done_keys:
                            skipped += 1
                            pbar.update(1)
                            continue

                        seed = seeds[repeat % len(seeds)]

                        completion = client.complete(
                            model_key=model_key,
                            lms_id=model["lms_id"],
                            prompt_type=prompt_type,
                            system_prompt=system_prompt,
                            user_message=user_message,
                            question=question.to_dict(),
                            repeat=repeat,
                            seed=seed,
                            temperature=gen_cfg["temperature"],
                            top_p=gen_cfg["top_p"],
                            max_tokens=gen_cfg["max_tokens"],
                            reasoning_effort=gen_cfg.get("reasoning_effort"),
                        )

                        response_final, format_ok, parse_mode = parse_final_answer(completion.text)
                        format_ok_counter[prompt_type]["ok" if format_ok else "bad"] += 1
                        if completion.finish_reason == "length":
                            truncated_counter[prompt_type] += 1

                        row = {
                            "run_id": run_id,
                            "is_mock": is_mock,
                            "question_id": question_id,
                            "question_type": question["question_type"],
                            "question_subtype": question.get("question_subtype", ""),
                            "answerable": question["answerable"],
                            "question": question["question"],
                            "context": question.get("context", ""),
                            "ground_truth": question.get("ground_truth", ""),
                            "model_key": model_key,
                            "lms_id": model["lms_id"],
                            "quant": model["quant"],
                            "engine": model["engine"],
                            "prompt_type": prompt_type,
                            "repeat": repeat,
                            "seed": seed,
                            "temperature": gen_cfg["temperature"],
                            "top_p": gen_cfg["top_p"],
                            "max_tokens": gen_cfg["max_tokens"],
                            "reasoning_effort": gen_cfg.get("reasoning_effort", ""),
                            "response_raw": completion.text,
                            "response_final": response_final,
                            "prompt_tokens": completion.prompt_tokens,
                            "completion_tokens": completion.completion_tokens,
                            "latency_ms": completion.latency_ms,
                            "finish_reason": completion.finish_reason,
                            "format_ok": format_ok,
                            "parse_mode": parse_mode,
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                        }
                        writer.writerow(row)
                        csv_file.flush()
                        # flush()는 OS 버퍼까지만이다. 이 실험은 전원이 예고 없이
                        # 끊기는 기계에서 돌기 때문에(학교 건물 야간 차단) 버퍼에만
                        # 있던 행은 그대로 사라진다. 응답 1건이 8초씩 걸리므로
                        # fsync 비용은 무시할 수 있다 — 잃는 쪽이 훨씬 비싸다.
                        os.fsync(csv_file.fileno())
                        written += 1
                        pbar.update(1)
    finally:
        pbar.close()
        csv_file.close()

    logger.info("=== 실행 요약 ===")
    logger.info("작성된 행: %d, 건너뛴 행(resume): %d", written, skipped)
    for prompt_type, counter in format_ok_counter.items():
        total = counter["ok"] + counter["bad"]
        if total == 0:
            continue
        rate = counter["ok"] / total
        n_trunc = truncated_counter.get(prompt_type, 0)
        logger.info(
            "  %s: format_ok_rate=%.3f (n=%d), 잘림(max_tokens 도달)=%d",
            prompt_type, rate, total, n_trunc,
        )
        if n_trunc and n_trunc >= (total - counter["ok"]) * 0.5:
            logger.warning(
                "  ↑ %s의 형식 실패는 대부분 잘림이다 — 형식 불이행이 아니라 "
                "max_tokens가 모자란 것이므로 프롬프트를 고치지 말 것.",
                prompt_type,
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
