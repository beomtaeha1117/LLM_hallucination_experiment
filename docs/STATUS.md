# 진행 상황 — 2026-09-05

## 한 줄 요약
**코드는 전부 완성되어 실제 200문항 규모로 통과했다. 남은 일은 문항 180개의 사실 검증뿐이다.**

---

## 1. 확정된 것

### 프롬프트 7종 (`prompts/`)
| 조건 | 불확실성 | 자기검증 | 위치 |
|------|:---:|:---:|------|
| P0 baseline | X | X | 2×2 요인 |
| P1 uncertainty | O | X | 2×2 요인 |
| P2 verification | X | O | 2×2 요인 |
| P3 combined | O | O | 2×2 요인 |
| P2L length-control | X | X | 비교군 (출력 길이 교란 통제) |
| P4 prohibition | X | X | 비교군 ("지어내지 마"라는 금지만) |
| P5 few-shot | X | X | 비교군 (예시 3개만) |

### 문항 200개 (`data/questions_v1_DRAFT.csv`)
스키마 검증 통과, Y=100 / N=100, few-shot 오염 없음.

| 유형 | 수 | 검증 상태 |
|------|---:|-----------|
| easy_factual | 20 | 미검증 |
| hard_factual | 50 | 미검증 |
| numeric | 15 | 미검증 |
| context_qa | 15 | **완료** |
| fake_paper | 15 | 미검증 (위험도 높음) |
| fake_concept | 25 | 미검증 |
| false_premise | 35 | 미검증 (event 8 / anachronism 12 / number 7 / person 8) |
| unknowable | 15 | 미검증 (future 5 / private 5 / subjective 5) |
| fake_statute | 5 | 미검증 (조문 범위 밖 번호라 구조적으로 안전) |
| context_qa_nogold | 5 | **완료** |
| **합계** | **200** | **20 완료 / 180 대기** |

### 파이프라인 (`src/`, `run_all.py`)
`python run_all.py` 한 번으로 응답생성 → 파싱 → 4단계 판정 → GEE 통계 → 그래프 4종.
**mock 모드로 200문항 전체 규모(12,600 응답)까지 통과했다.** LM Studio 없이 돌아간다.

- `src/client.py` — LM Studio 클라이언트 + MockClient
- `src/run_experiment.py` — 응답 생성, 중단 재개(resume) 지원
- `src/evaluate.py` — 규칙 → 정답매칭 → LLM judge(3회 다수결) → 라벨 확정
- `src/analyze.py` — 지표 6종 + Wilson CI + GEE + Cochran's Q + McNemar 2계열
- `src/sample_for_human.py` — 인간 검증용 층화 표본(조건별 정확히 균등)
- `src/agreement.py` — Cohen's κ
- `tools/make_template.py`, `tools/validate_questions.py`

### 통계 설계
- **요인 분석**: GEE 로지스틱 `hallucination ~ uncertainty * verification`, P0~P3만. 상호작용항이 연구질문 3.
- **길이 교란 통제**: 위 모형에 `completion_tokens_z` 공변량 추가, P2L 포함.
- **요인 쌍별**: Cochran's Q + McNemar 6쌍, Bonferroni.
- **사전 계획 대조** (별도 검정 가족, Bonferroni 6): P0-P2L, P2-P2L, P0-P4, P1-P4, P0-P5, P1-P5.

---

## 2. 남은 일 (우선순위 순)

1. **문항 180개 사실 검증** ← 유일한 병목
   - `verify/worklist.html`을 브라우저로 열어 진행 (검색 링크가 문항마다 붙어 있음)
   - 끝나면 `tools/apply_verification.py`로 `data/questions_v1.csv` 생성
2. 맥 스튜디오에서 LM Studio 설치 + 모델 3종 + `LMStudioClient`/`LMStudioJudge` 실동작 확인 (**아직 미검증**)
3. 파일럿 실행 (20문항 × 7조건 × 1반복 × 1모델 = 140응답)
4. 파일럿 결과로:
   - 보류 표현 사전(`data/abstention_lexicon.txt`) 구축 후 **동결**
   - judge 모델 선정 (κ 기준)
   - 천장·바닥 효과 점검 → 난이도 재배정 (P0 정답률 100%인 hard 문항은 easy로 이동)
   - P4의 "환각" 용어 vs 일상어 표현 차이 확인 → 필요하면 P4a/P4b 분리
5. 본실험 (200 × 7 × 3 × 3모델 = 12,600 응답, 예상 18~35시간)
6. 인간 검증 400건 → κ
7. 제출 직전 RISS 재검색

---

## 3. 반드시 기억할 함정

- **문항 검증을 건너뛰면 안 된다.** 존재하지 않는다고 표시한 논문·용어·조항이 실제로 존재하면 채점 기준 자체가 틀린 것이 되고, 감점 -2 조건에 직결된다.
- **보류 표현 사전은 P0·P2 응답으로 먼저 만든다.** P1·P3에만 "확인할 수 없습니다"라는 표현이 지정되어 있어서, 사전을 그 표현에 맞추면 조건 간 차이가 인위적으로 부풀려진다.
- **judge와 피험 모델을 동시에 메모리에 올리지 않는다.** 36GB에 21.2GB 모델 두 개는 불가. 응답 전량 생성 → 언로드 → judge 로드.
- **결과를 먼저 정해두지 않는다.** `results/`에 지금 들어 있는 것은 전부 mock 데이터이며, `is_mock` 컬럼과 그래프 제목에 표시되어 있다.
- **temperature 0.7 고정, 반복마다 seed만 변경.** 둘 다 고정하면 3회 반복이 같은 응답이 된다.

---

## 4. 현재 config
`config.yaml` — run_id `mock_002_full200`, `mock: true`, 200문항 세트.
`config_pilot.yaml` — 27문항 파일럿 초안용 백업.
실제 실험 시 `mock: false`로 바꾸고 `lms_id`를 실제 모델 ID로 채울 것.
