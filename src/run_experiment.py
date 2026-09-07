"""응답 생성 스테이지: 모델 x 조건 x 문항 x 반복을 순회하며 results/raw_responses.csv를 만든다."""

from __future__ import annotations

import argparse
import csv
import glob
import logging
import os
import time
from collections import Counter
from datetime import datetime, timezone
from typing import Dict, Set, Tuple

import pandas as pd
import yaml
from tqdm import tqdm

from src.client import LMStudioClient, MockClient
from src.schema import RAW_COLUMNS, load_questions, parse_final_answer

logger = logging.getLogger("run_experiment")
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

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
    """이미 완료된 (run_id, model_key, prompt_type, question_id, repeat) 키 집합을 반환한다."""
    if not os.path.exists(raw_path):
        return set()
    done: Set[Tuple] = set()
    try:
        existing = pd.read_csv(raw_path, dtype=str, encoding="utf-8-sig")
    except pd.errors.EmptyDataError:
        return set()
    for _, row in existing.iterrows():
        key = (
            row["run_id"],
            row["model_key"],
            row["prompt_type"],
            row["question_id"],
            str(row["repeat"]),
        )
        done.add(key)
    return done


def run(config_path: str) -> None:
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    run_id = config["run_id"]
    is_mock = bool(config.get("mock", False))
    questions_file = config["questions_file"]
    conditions = config["conditions"]
    models = config["models"]
    repeats = config["repeats"]
    seeds = config["seeds"]
    gen_cfg = config["generation"]

    os.makedirs("results", exist_ok=True)
    raw_path = "results/raw_responses.csv"

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
                        key = (run_id, model_key, prompt_type, question_id, str(repeat))
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
                        )

                        response_final, format_ok = parse_final_answer(completion.text)
                        format_ok_counter[prompt_type]["ok" if format_ok else "bad"] += 1

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
                            "response_raw": completion.text,
                            "response_final": response_final,
                            "prompt_tokens": completion.prompt_tokens,
                            "completion_tokens": completion.completion_tokens,
                            "latency_ms": completion.latency_ms,
                            "finish_reason": completion.finish_reason,
                            "format_ok": format_ok,
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                        }
                        writer.writerow(row)
                        csv_file.flush()
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
        logger.info("  %s: format_ok_rate=%.3f (n=%d)", prompt_type, rate, total)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
