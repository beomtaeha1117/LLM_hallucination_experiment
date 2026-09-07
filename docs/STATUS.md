# 진행 상황 — 2026-09-08

## 한 줄 요약
**"코드 완성, 문항 검증만 남음"은 틀렸다.** 코드는 mock 규모로 돌아갈 뿐이고,
**채점·저장·통계에 실제로 재현되는 결함이 12건 있다**(§5). 실제 모델로는 단 한 번도 안 돌렸다.
문항 검증은 65/200까지 왔다 — 위험도가 가장 높은 U1·U2·U5 45건이 원본 DB 조회로 끝났다.

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
| fake_paper | 15 | **완료** (KCI+RISS 0건) |
| fake_concept | 25 | **완료** (KCI 0건) |
| false_premise | 35 | 미검증 (event 8 / anachronism 12 / number 7 / person 8) |
| unknowable | 15 | 미검증 (future 5 / private 5 / subjective 5) |
| fake_statute | 5 | **완료** (법령 원문 확인) |
| context_qa_nogold | 5 | **완료** |
| **합계** | **200** | **65 완료 / 135 대기** |

### 파이프라인 (`src/`, `run_all.py`)
`python run_all.py` 한 번으로 응답생성 → 파싱 → 4단계 판정 → GEE 통계 → 그래프 4종.
**mock 모드로 200문항 전체 규모(12,600 응답)까지 "돌아간다".**
돌아가는 것과 맞게 채점하는 것은 다르다 — §5의 결함을 먼저 읽을 것.

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

---

## 5. 코드 결함 12건 — **2026-09-08 전부 수정 완료** ✅

아래는 발견 당시의 기록이다. 전부 고쳤고 `pytest tests/ -q` 12건 통과 + mock 전 구간
end-to-end 실행으로 확인했다. **다시 열지 말 것** — 어떤 것이 왜 위험했는지 남겨두는 표다.

### 채점 (`src/evaluate.py`)
| # | 위치 | 증상 |
|---|---|---|
| 1 | `_match_answer` L60 | 정규화 후 **단순 부분문자열 포함**. "수도는 서울이 아니라 부산입니다" → CORRECT |
| 2 | 같은 곳 | `A1-05`의 `acceptable_answers`에 `12`가 있어 "**1**12개월" → CORRECT |
| 3 | 같은 곳 | `A2-35`가 `아데닌\|구아닌\|사이토신\|티민`으로 쪼개져 **"아데닌입니다" 하나로 CORRECT** |
| 4 | `_evaluate_row` L216 | N문항은 보류 표현만 걸리면 즉시 CORRECT. **보류 뒤에 날조를 붙여도 정답 보류** |
| 5 | `LMStudioJudge` L178 | judge 접속 실패를 `HALLUCINATION`으로 **기록**. 실패가 데이터가 된다 |
| 6 | `LMStudioJudge` L152 | judge에 `context`·`acceptable_answers`·`why_unanswerable`이 안 감. L231에서 merge까지 해놓고 안 쓴다 |

### 실행·저장
| # | 위치 | 증상 |
|---|---|---|
| 7 | `run_experiment` | `results/raw_responses.csv` **한 파일에 append**. `run_id`·`is_mock` 컬럼은 있는데 evaluate L231·analyze L549가 **필터를 안 한다.** mock과 실제가 한 통계에 섞인다 |
| 8 | `_load_done_keys` | `mock: false`만 바꾸고 `run_id`를 그대로 두면 resume 키가 맞아 **실제 생성이 통째로 건너뛰어진다.** 가장 위험 |
| 9 | 전역 | `thinking: false`가 config에만 있고 **어디서도 읽히지 않는다** |
| 10 | `analyze` | `graphs/`에 `makedirs` 없음. 새 클론에서 그래프 단계 실패 |

### 통계 (`src/analyze.py`)
| # | 위치 | 증상 |
|---|---|---|
| 11 | L353 `with_p2l = g.copy()` | 주석은 "P0-P3-P2L"인데 **P4·P5까지 들어가고**, 이들이 `uncertainty=0, verification=0`으로 **P0과 같은 칸에 합쳐진다** |
| 12 | `_wilson_ci` L57 | **문항당 3반복의 의존성을 무시**하고 풀링. 그래프 CI가 실제보다 좁다 |

### 회귀 테스트로 고정할 6개 사례
부산 / 112개월 / 아데닌 단독 / 보류+날조 / judge 접속실패 / judge 컨텍스트 누락.

---

## 6. 2026-09-08 확정된 설계 변경 3건

1. **Abstention F1을 종합지표에서 보조지표로 강등.**
   Y·N이 반반이면 전부 보류해도 F1 ≈ 0.67이고, 답한 것의 정확성이 반영되지 않는다.
   종합 판단은 **환각률–정확도 트레이드오프 평면**으로 한다.
2. **실험 B(지시 위치 효과)를 확장 실험으로 확정.** → `docs/experiment-b-position.md`
   실험 A 본실험이 끝나기 전에는 착수하지 않는다. 게이트는 B 문서 §8.
3. **`config.yaml`의 `gemma4: quant: Q4_K_M`은 틀렸다.**
   공식 GGUF는 `google/gemma-4-12b-it-qat-q4_0-gguf` = **QAT Q4_0, 약 6.98GB**.
   실제로 받은 파일에 맞춰 고칠 것. (Qwen3.6-35B-A3B Q4_K_M 21.2GB는 config와 일치, 확인됨.)

## 7. 모델 정보 확인 상태 (2026-09-08)
| 항목 | 상태 |
|---|---|
| Qwen3.6-35B-A3B GGUF Q4_K_M 21.2GB | 공식 저장소 확인 ✅ |
| Gemma 4 12B IT QAT Q4_0 약 6.98GB | 공식 저장소 확인 ✅ (config 표기 수정 필요) |
| Gemma 라이선스 | **미확정.** HF 페이지에서 Apache 2.0으로 읽혔으나 Gemma 계열은 통상 별도 약관. 논문에 쓰기 전 원문 재확인 |
| 36GB에서의 실제 메모리·속도 | **미실측** |
| LM Studio가 seed를 반영하는지 | **미확인** |
| judge 모델 | **미선정** (`judge-placeholder`) |


---

## 8. 결함 수정 후 확인된 것 (2026-09-08, 전부 직접 실행)

```
pytest tests/ -q                      -> 12 passed
python run_all.py --config <mock>     -> 2,800응답 생성→판정→통계→그래프 전 구간 통과
재실행                                 -> "작성된 행: 0, 건너뛴 행(resume): 2800"
같은 run_id에 mock:false               -> RuntimeError로 중단 (클라이언트 생성 전)
python tools/validate_questions.py    -> 200행, Y=100/N=100 유지
```

### 바뀐 구조
- 결과 경로가 **`results/<run_id>/`, `graphs/<run_id>/`로 분리**됐다. `.run_meta.json`이
  그 run의 mock 여부를 기록하고, 어긋나면 **실행 자체를 막는다.**
  옛 mock 산출물은 `results/legacy_mock/`, `graphs/legacy_mock/`에 보존.
- 채점은 "부분문자열이 걸리면 정답"에서 **"확실할 때만 자동 정답, 애매하면 judge로 escalate"**로 바뀌었다.
  부정어 가드, 숫자 경계 검사, `match_mode`/`tolerance_pct`/`reject_answers` 컬럼이 추가됐다.
- judge 실패는 `JUDGE_ERROR` 라벨 + `decided_by="judge_error"`로 기록되고,
  `analyze.py`가 **분모에서 제외**한다. **5%를 넘으면 분석을 거부하고 중단한다** —
  judge가 절반쯤 죽은 실행을 정상 실행으로 착각하지 않기 위해서다.
- CI가 문항 클러스터 부트스트랩(n_boot=2000, question_id 재표집)으로 바뀌었다. 그래프에도 표기된다.
- 길이 교란 통제 GEE에서 **P4·P5가 빠졌다**(P0·P1·P2·P3·P2L 5조건만).

### 추가로 확정한 설계 변경 2건
1. **judge 투표 3 → 1.**
   temperature 0.0에서 같은 요청을 세 번 반복하는 것은 **독립 평가자 세 명이 아니다.**
   거의 같은 답 세 개이고, 그걸 "3회 다수결"이라 부르면 있지도 않은 신뢰도를 주장하게 된다.
   판정 신뢰도는 **인간 400건 재판정 κ**로 잡는다. 부수 효과로 judge 호출이 1/3이 된다
   (결함 #4 수정으로 N행이 전부 judge를 타게 되어 호출량이 늘었는데, 이걸로 상쇄된다).
   투표 기계는 남겨뒀다 — temperature>0으로 부분표본 자기일관성을 볼 때 쓴다.
2. **`tolerance_pct`는 상대 백분율이지 백분율 포인트가 아니다.**
   78%에 1.5를 주면 ±1.17%p다. config와 문항 컬럼 모두 같은 의미다. 헷갈리면 채점 기준이 흔들린다.

### 아직 확인 못 한 것 (실서버 필요)
- **`thinking` 파라미터명 미검증.** 설정이 요청까지 전달되는 경로는 만들었고
  `extra_body={"thinking": ...}`로 실었지만, LM Studio가 실제로 받는 이름인지는 **모른다.**
  코드 주석과 첫 호출 로그에 경고를 남겨뒀다. **실서버에서 반드시 확인할 것.**
- `seed`는 원래부터 요청에 정상 전달되고 있었다(코드로 확인, 새로 고친 것 아님).
  다만 **LM Studio가 그 seed를 실제로 반영하는지는 미확인.**
- 클러스터 부트스트랩 n_boot=2000을 12,600행 실규모에서 안 돌려봤다. 느리면 하향 조정.
