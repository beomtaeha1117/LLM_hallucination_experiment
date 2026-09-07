# 연구 설계 확정본 v0.1
팀명: 할루시네이션 / 연구자: 이경빈 / 작성일: 2026-09-05
상태: **설계 단계 (코드 미작성)**. 파일럿 결과에 따라 수정 전제.

---

## 0. 이 문서가 기존 브리프에서 바꾼 것 (비판적 수정 사항)

| # | 원안 | 문제 | 수정안 |
|---|------|------|--------|
| 1 | 200×4×3반복, seed 고정 | temperature=0 + seed 고정이면 3회 반복이 **같은 응답**이라 반복 자체가 무의미 | temperature를 **0.7로 고정**하고 반복마다 **seed만 1/2/3으로 변경**. 반복 = "모델 내부 확률적 분산" 측정 |
| 2 | 지표 5종 | **과잉보류(over-abstention)** 지표가 없음. trade-off의 한쪽 축이 비어 있음 | `Over-Abstention Rate`(답변 가능 문항인데 보류)를 필수 지표로 추가 |
| 3 | P2/P3 = 자기검증 | P2/P3는 출력 토큰이 길어짐 → "프롬프트 구조" 효과인지 "출력이 길어진" 효과인지 **교란** | ① 출력 토큰 수를 공변량으로 기록 ② **길이 통제 대조군 P2L** 추가(검증 지시 없이 분량만 늘리는 프롬프트) |
| 4 | 모델 자유 선택 | Qwen3.6·gpt-oss는 **추론(thinking) 모델** → 자기검증이 이미 내장되어 P2 효과가 상쇄될 수 있음 | thinking 모드를 **off로 고정**(통제변인). thinking on/off는 별도 확장 실험 |
| 5 | LLM 1대에 전량 판정 | 판정자 환각 | **4단계 하이브리드 판정**(규칙→정답매칭→폐쇄형 LLM judge 3회 다수결→인간 표본 κ) |
| 6 | 응답의 10~20% 인간 검증 | 7,200×0.2 = 1,440건 × 2인 = 비현실적 | **셀 층화 표본 400건**으로 축소, 2인 독립 판정 후 Cohen's κ |
| 7 | 카이제곱 등 | 같은 질문에 4조건 반복 = **대응 표본**. 독립성 가정 위반 | 주분석 **혼합효과 로지스틱 회귀(질문 ID 랜덤절편)**, 보조 McNemar/Cochran's Q |

---

## 1. 제목 후보 (RISS 중복 회피 지향)

1. **한국어 답변불가 질의에서 불확실성 명시·자기검증 프롬프트의 2×2 조합이 로컬 LLM의 환각률과 과잉보류에 미치는 영향** (권장)
2. 온디바이스 LLM의 한국어 환각 억제: 프롬프트 구조에 따른 정확도-보류 트레이드오프 분석
3. 자체 제작 한국어 답변불가 질의 세트를 이용한 로컬 LLM 환각-유용성 트레이드오프 측정

**차별화 근거 4가지** (RISS 중복 방어 논리로 논문 서론에 명시):
① 한국어 **자체 제작** unanswerable/false-premise 문항 → 벤치마크 오염 회피
② **로컬/온디바이스** 모델 (API 모델이 아님) → 재현 가능, 버전 고정 가능
③ 환각률 단독이 아니라 **과잉보류를 대칭 지표**로 둔 trade-off
④ **길이 통제 대조군**으로 "자기검증"과 "출력량 증가"를 분리

---

## 2. 변인 설계 (섞이지 않게 고정)

### 독립변인 (조작)
프롬프트 구조 — 2×2 요인설계 + 대조군 1

| 조건 | 불확실성 명시 | 자기검증 | 비고 |
|------|:---:|:---:|------|
| P0 baseline | X | X | 기준선 |
| P1 uncertainty | O | X | |
| P2 verification | X | O | |
| P3 combined | O | O | 상호작용 검정의 핵심 |
| P2L length-control | X | X(분량만) | 교란 통제용 대조군 |
| P4 prohibition | X | X | **금지만 하는 조건**. 2×2 밖의 비교군 |
| P5 few-shot | X | X | **예시만 보여주는 조건**. 2×2 밖의 비교군 |

> 연구질문 3("P3가 P1+P2보다 나은가")은 통계적으로 **상호작용항**이다. 2×2를 유지해야 검정 가능하므로 조건을 더 늘리지 말 것. CoVe는 본실험 이후 확장으로만.

### 종속변인 (측정)
아래 §3 지표 6종.

### 통제변인 (전부 config.yaml에 하드코딩 + 결과 CSV에 기록)
model / model file 해시 / quantization(Q4_K_M) / inference engine(llama.cpp GGUF) /
temperature 0.7 / top_p 0.9 / top_k 40 / repeat_penalty 1.0 / max_output_tokens 512 /
context_length 8192 / thinking mode OFF / system prompt(조건별 프롬프트 외 동일) /
대화 이력 없음(매 요청 stateless, 단일 턴) / 질문 제시 순서 고정 /
언어 = 한국어 고정 / 실행 머신 = Mac Studio M4 Max

---

## 3. 지표의 조작적 정의 (여기가 논문 채점의 핵심)

모든 응답을 **3분류**한다: `CORRECT` / `HALLUCINATION` / `ABSTAIN`

### 답변 가능 문항 (answerable = Y)
- ground_truth와 일치 → CORRECT
- 사실과 다른 내용을 단정 → HALLUCINATION
- "확인할 수 없음/모름" → ABSTAIN (= **과잉보류**)

### 답변 불가능 문항 (answerable = N: 존재하지 않는 논문/인물, 잘못된 전제, 미래·주관 질문)
- 보류하거나 **전제가 잘못됐음을 지적** → CORRECT (= 정답 보류)
- 존재하지 않는 내용을 지어냄 → HALLUCINATION
- (별도 ABSTAIN 범주 없음: 보류가 곧 정답)

### 지표 6종
```
Hallucination Rate      = HALLUCINATION / 전체 응답
Accuracy@Answerable     = CORRECT / answerable 문항 응답수      (보류는 오답 처리)
Correct Abstention Rate = CORRECT / unanswerable 문항 응답수
Over-Abstention Rate    = ABSTAIN / answerable 문항 응답수      ← 신규
Answer Rate             = (CORRECT+HALLUCINATION 중 실답변) / 전체
Abstention F1           = "보류해야 할 문항을 보류했는가"를 이진분류로 본 F1
                          (positive = unanswerable, Precision/Recall 동시 보고)
```
> **Abstention F1이 종합 지표**다. "무조건 모른다"고 답하면 Recall은 1이지만 Precision이 무너지므로, 사용자가 말한 trade-off가 하나의 수치로 잡힌다. 논문 결론은 Hallucination Rate 단독이 아니라 (Hallucination Rate ↓, Accuracy@Answerable 유지, Abstention F1 ↑) 3축으로 서술한다.

---

## 4. 데이터셋 설계 (목표 200문항)

| # | 유형 | answerable | 문항수 | 출처 |
|---|------|:---:|---:|------|
| 1 | 한국사 factual QA | Y | 25 | 교과서·국사편찬위 등 검증 가능 사실 |
| 2 | 과학 factual QA | Y | 25 | 교과서 |
| 3 | 일반상식/생활 | Y | 20 | |
| 4 | 한국 지역·문화·통계 | Y | 20 | 통계청 등 (연도 명시) |
| 5 | context QA (제공 문맥 내 정답 존재) | Y | 10 | 자체 작성 지문 |
| 6 | 존재하지 않는 논문/인물/이론 | N | 40 | 자체 제작 |
| 7 | 잘못된 전제 포함 질문 | N | 35 | 자체 제작 |
| 8 | 원리적 답변 불가(미래·주관·비공개) | N | 15 | 자체 제작 |
| 9 | context QA (문맥에 정답 없음) | N | 10 | 자체 작성 지문 |
| | **합계** | | **200** | answerable 100 / unanswerable 100 |

**규칙**
- 전량 한국어 자체 제작을 원칙으로 한다. TruthfulQA/HaluEval은 **오염 대조군으로 20문항만 별도 부록 실험**(주 분석에 섞지 않음). 번역본을 주 데이터로 쓰면 번역 품질이 새로운 교란변인이 된다.
- 정답 문항은 `ground_truth` + `acceptable_answers`(별칭·표기 변형 리스트) + `source_url`을 함께 기록.
- 6~8번 문항은 실제로 존재하지 않음을 **검색으로 확인한 뒤** 기록(`verified_nonexistent: true`, 확인일자 포함). 이게 무너지면 연구 전체가 무너진다.
- 문항 작성 즉시 파일럿 20문항으로 난이도 점검 — 너무 쉬우면 조건 간 차이가 안 나오고(천장효과), 너무 어려우면 P0에서 이미 전부 보류한다(바닥효과).

---

## 5. 실행 규모

```
200 문항 × 7 조건(P0,P1,P2,P3,P2L,P4,P5) × 3 반복 = 4,200 응답 / 모델
3 모델                                             = 12,600 응답
```
- 파일럿: 20문항 × 7조건 × 1반복 × 1모델 = 140 응답 (반드시 선행)
- 예상 소요: 응답당 평균 4~10초(P2/P3가 김) → 모델당 6~12시간, 전체 **18~35시간**. 야간 배치 + 중단 재개(resume) 필수.

---

## 6. 통계 분석

- **주분석**: 혼합효과 로지스틱 회귀
  `hallucination ~ uncertainty * verification + (1 | question_id)`
  → 주효과 2개 + **상호작용**(연구질문 3). 모델별로 각각 적합 → 재현성(연구질문 5).
  구현: `statsmodels` GEE(exchangeable) 또는 `BinomialBayesMixedGLM`. 고교 논문 서술 난도를 고려해 **GEE + 클러스터 강건 표준오차**를 1순위로.
- **보조**: 조건 4개 전체 = Cochran's Q, 쌍별 = McNemar + Bonferroni 보정.
- **효과크기·불확실성**: 오즈비와 95% 신뢰구간, 비율은 Wilson CI. p값만 쓰지 말 것.
- **판정 신뢰도**: Cohen's κ (자동 vs 인간, 인간1 vs 인간2).
- 유의수준 α=0.05, 다중비교 보정 명시.

---

## 7. 환각 판정 자동화 (4단계 하이브리드)

1. **규칙 기반 보류 탐지** — 한국어 보류 표현 사전(정규식). "확인할 수 없", "알 수 없", "정보가 없", "존재하지 않", "모르겠", "답변드리기 어렵" 등. 사전은 파일럿 응답으로 만들고 **동결**한 뒤 본실험에 적용(사후 조정 금지).
2. **정답 매칭** — 정규화(공백/조사/단위) 후 `acceptable_answers` 포함 여부, 숫자는 허용오차.
3. **폐쇄형 LLM judge** — 1·2에서 미결인 것만. 판정자에게 **ground_truth를 함께 제공**하여 "사실 확인"이 아니라 "일치 여부 분류"만 시킨다(judge 환각 위험 급감). JSON schema 강제 출력, **동일 응답 3회 판정 후 다수결**, judge는 피험 모델과 **다른 계열**로 고정.
4. **인간 검증** — (조건 × 문항유형) 셀 층화 랜덤 400건, 2인 독립 판정, Cohen's κ 보고. κ < 0.6이면 rubric 수정 후 자동 판정 전체 재실행.

---

## 8. 실행 환경 확정안 (2026-09-05 웹 검증 기준)

**본실험: Mac Studio M4 Max / 36GB / LM Studio / llama.cpp(GGUF) 엔진**
- GGUF로 통일하는 이유: MLX가 Mac에서 더 빠르지만 CUDA와 **동일 체크포인트 비교가 불가**. MLX는 부록의 속도 비교로만 사용.
- Windows RTX 5070 Ti(16GB VRAM)는 **속도 비교 및 예비실험** 전용. 35B급은 16GB에 안 들어가 부분 오프로딩이 되므로 본실험 부적합. 사용자의 원 판단이 맞다.

**모델 후보 (존재 확인됨)**

| 역할 | 모델 | 라이선스 | 근거 |
|------|------|----------|------|
| Main | Qwen3.6-35B-A3B (GGUF Q4_K_M, 21.2GB) | Apache 2.0 | lmstudio-community 리포 확인. MoE(활성 3B급)라 M4 Max에서 속도 유리 |
| Replication | Gemma 4 12B | Apache 2.0 | 2026-06-03 공개, 계열이 다름 |
| Third | gpt-oss-20b (MXFP4) | Apache 2.0 | 16GB에서 동작 → **Windows 5070 Ti에 완전 적재 가능** → 플랫폼 비교에 최적 |
| Judge | 위 3종과 다른 1종 (피험 모델과 겹치지 않게) | — | 파일럿에서 κ로 선정 |

**미확인 / 다음 단계에서 반드시 재검증할 것**
- 36GB 통합 메모리에서 Q4_K_M 21.2GB + 8K 컨텍스트 실측 여유 (judge 모델 동시 적재는 금지, 순차 실행)
- Qwen3.6·gpt-oss의 thinking off 설정이 LM Studio에서 실제로 적용되는지
- 한국어 특화 모델(EXAONE / Kanana / HyperCLOVA X SEED)의 2026년 최신판·GGUF 가용성 — 검색으로 계열 존재는 확인했으나 **최신 버전과 파일은 미확인**. 넣는다면 "한국어 특화 vs 다국어" 대비로 차별성이 올라가므로 다음 단계에서 확인.

**소프트웨어**: LM Studio(+`lms` CLI), Python 3.11+, VS Code, Git
**패키지**: openai, pandas, numpy, scipy, statsmodels, matplotlib, tqdm, pyyaml
(deepeval/promptfoo/Docker/PyTorch는 **도입하지 않음** — 판정 로직을 직접 통제해야 논문에 서술 가능)

---

## 9. 다음 단계 (순서 고정)

1. LM Studio 설치 + 모델 3종 다운로드 + `/v1/chat/completions` 연결 확인, 실측 속도·메모리 기록
2. 프롬프트 5종 문안 확정 (한국어, 조건 간 길이·어조 최대한 통제)
3. 문항 20개 파일럿 세트 작성
4. `run_experiment.py` 최소 버전 → 파일럿 100응답 실행
5. 파일럿으로 보류 표현 사전 구축 + judge 선정(κ) + 천장/바닥효과 점검
6. 200문항 본세트 완성
7. 본실험 배치 실행 → 평가 → 분석 → 그래프
8. 제출 직전 RISS 재검색으로 중복 확인

