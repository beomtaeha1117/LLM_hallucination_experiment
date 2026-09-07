"""전체 파이프라인 순차 실행: run_experiment -> evaluate -> analyze. 로직 없음."""

from __future__ import annotations

import argparse
import time

from src import analyze, evaluate, run_experiment


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    stages = [
        ("run_experiment", run_experiment.run),
        ("evaluate", evaluate.run),
        ("analyze", analyze.run),
    ]

    for name, func in stages:
        t0 = time.monotonic()
        print(f">>> [{name}] 시작")
        func(args.config)
        elapsed = time.monotonic() - t0
        print(f">>> [{name}] 완료 ({elapsed:.1f}초)")


if __name__ == "__main__":
    main()
