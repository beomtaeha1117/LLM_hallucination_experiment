# 한국어 질의에서 프롬프트 구조가 로컬 LLM의 환각과 답변 보류에 미치는 영향

고등학교 소논문 연구 — 팀명 **할루시네이션** / 연구자 **이경빈**

> **상태: 설계 및 자동화 완료, 실험 미실시.**
> 이 저장소에는 아직 **어떤 실험 결과도 들어 있지 않습니다.** 파이프라인은 mock(가짜) 데이터로만 검증했고,
> mock 산출물은 실제 결과로 오인되지 않도록 저장소에서 제외했습니다.

---

## 연구 질문

동일한 모델·동일한 질문에서 **프롬프트 구조**만 바꿨을 때,

1. 환각(hallucination) 발생률이 유의미하게 달라지는가?
2. 불확실성 명시 프롬프트는 답할 수 없는 질문에서 **적절한 보류**를 늘리는가?
3. 자기검증 프롬프트는 환각을 줄이는가? (둘을 합치면 더 좋은가 = **상호작용**)
4. 환각 감소가 **정상 질문의 정확도 저하**를 동반하는가?
5. 다른 로컬 모델에서도 같은 효과가 재현되는가?

환각률이 가장 낮은 프롬프트를 찾는 연구가 **아니다.**
정확도를 지키면서 환각을 줄이고, 답할 수 없는 질문에서는 적절히 보류하는 지점을 찾는 **트레이드오프 분석**이다.

## 프롬프트 조건 7종

2×2 요인설계 + 비교군 3종.

| 조건 | 불확실성 명시 | 자기검증 | 설계상 위치 |
|------|:---:|:---:|------|
| P0 baseline | X | X | 요인 |
| P1 uncertainty | O | X | 요인 |
| P2 verification | X | O | 요인 |
| P3 combined | O | O | 요인 (상호작용 검정) |
| P2L length-control | X | X | 비교군 — 출력 길이 교란 통제 |
| P4 prohibition | X | X | 비교군 — "지어내지 마"라는 순수 금지 |
| P5 few-shot | X | X | 비교군 — 지시 대신 예시 3개 |

`P2L`이 있는 이유: 자기검증 프롬프트는 출력이 길어진다. 환각이 준 것이 **검증 때문인지 길이 때문인지** 가르지 못하면 결론을 쓸 수 없다.

## 지표

```
Hallucination Rate       환각 / 전체
Accuracy@Answerable      정답 / 답변가능 문항        (보류는 오답 처리)
Correct Abstention Rate  보류 / 답변불가 문항
Over-Abstention Rate     보류 / 답변가능 문항        ← 트레이드오프의 반대쪽 축
Answer Rate
Abstention F1            "보류해야 할 것을 보류했는가"의 이진분류 F1
```

`Abstention F1`이 종합 지표다. "전부 모르겠다"고 답하면 Recall은 1이지만 Precision이 무너지므로,
환각 억제와 유용성 사이의 트레이드오프가 하나의 수치에 잡힌다.

## 문항 200개

`data/questions_v1_DRAFT.csv` — 답변가능 100 / 답변불가 100.

| 유형 | 수 | |
|------|---:|---|
| easy_factual | 20 | 과잉보류 탐지기 |
| hard_factual | 50 | 조건 간 정확도 차이가 나오는 구간 |
| numeric | 15 | |
| context_qa | 15 | 지문 안에 답 있음 |
| fake_paper | 15 | 존재하지 않는 논문 |
| fake_concept | 25 | 존재하지 않는 학술 조어 |
| false_premise | 35 | 잘못된 전제 (사건/시대착오/수치/인물) |
| unknowable | 15 | 미래·비공개·주관 |
| fake_statute | 5 | 존재하지 않는 법 조항 |
| context_qa_nogold | 5 | 지문에 답 없음 |

전량 **자체 제작 한국어 문항**이다. 기존 벤치마크(TruthfulQA, HaluEval 등)는 학습 데이터 오염 가능성이 있어 주 분석에 쓰지 않는다.

> ⚠️ `verified` 컬럼이 `NO`인 문항은 아직 출처 검증이 끝나지 않았다.
> **존재하지 않는다고 표시한 논문·용어·조항이 실제로 존재하면 채점 기준 자체가 틀린 것이 된다.**
> 검증 진행 상황은 `docs/verification-log.md` 참고.

## 실행

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python run_all.py --config config.yaml
```

`config.yaml`의 `mock: true`면 LM Studio 없이 가짜 응답으로 전 구간이 돈다(파이프라인 점검용).
실제 실험은 `mock: false`로 바꾸고 LM Studio를 띄운 뒤 실행한다.

```
run_all.py → src/run_experiment.py  응답 생성 (중단 재개 지원)
           → src/evaluate.py        규칙 → 정답매칭 → LLM judge(3회 다수결) → 라벨
           → src/analyze.py         지표 + Wilson CI + GEE + Cochran's Q + McNemar
```

## 통계

- **요인 분석**: GEE 로지스틱 `hallucination ~ uncertainty * verification`, 질문 ID 클러스터. 상호작용항이 연구질문 3.
- **길이 교란 통제**: 위 모형에 `completion_tokens` 공변량 추가.
- **사전 계획 대조**: P0-P2L, P2-P2L, P0-P4, P1-P4, P0-P5, P1-P5 (별도 검정 가족, Bonferroni).
- **판정 신뢰도**: 자동 판정 400건을 사람이 블라인드로 재판정 → Cohen's κ.

## 환각 판정

LLM 하나에게 전량 판정시키면 판정자 자신의 환각이 문제가 된다. 4단계로 나눴다.

1. **규칙** — 한국어 보류 표현 정규식 (`data/abstention_lexicon.txt`)
2. **정답 매칭** — 표기 정규화 + 허용 답안 + 수치 허용오차
3. **LLM judge** — 미결분만. **정답을 함께 제공**하여 "사실 판단"이 아니라 "일치 여부 3분류"만 시킨다. 3회 다수결.
4. **인간 검증** — 조건별 균등 층화 400건, 블라인드 판정 후 κ

## 문서

| 파일 | 내용 |
|------|------|
| `docs/research-design.md` | 변인 설계, 지표 정의, 통계 |
| `docs/experiment-plan.md` | 프롬프트 설계 원칙, 코드 구조, AI 역할 분리 |
| `docs/question-design.md` | 문항 유형 패턴과 작성 규칙 |
| `docs/verification-log.md` | 정답 검증 기록 (발견한 오류 포함) |
| `docs/STATUS.md` | 진행 상황과 함정 |

## 원칙

- 결과를 먼저 정해두지 않는다. 결론은 실제 데이터가 나온 뒤에 쓴다.
- 존재하지 않는 논문·DOI·링크를 만들어내지 않는다.
- 채점 기준은 데이터 수집 **전에** 확정한다. 결과를 보고 기준을 바꾸지 않는다.
- 응답 생성과 채점은 전량 로컬 모델로 수행한다. 클라우드 AI는 코드 작성 보조에만 사용했다.
