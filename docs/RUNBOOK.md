# 실험 실행 안내서 (맥 스튜디오)

작성 2026-09-08. **위에서 아래로 순서대로.** 각 단계에 "넘어가도 되는 조건"이 붙어 있다.
조건을 못 채우면 다음으로 가지 말 것 — 그 상태로 진행하면 결과를 못 쓴다.

---

## 0. 설치 (한 번만, 약 1시간 — 대부분 다운로드 시간)

### 0-1. 저장소와 파이썬
```bash
git clone https://github.com/beomtaeha1117/LLM_hallucination_experiment.git
cd LLM_hallucination_experiment
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install pytest
```
파이썬은 **3.11 이상**을 쓴다(개발은 3.11.16에서 했다). macOS 기본 파이썬이 낮으면
`brew install python@3.11` 후 그걸로 venv를 만든다.

**넘어가도 되는 조건**
```bash
pytest tests/ -q                                    # 12 passed
python tools/validate_questions.py data/questions_v1_DRAFT.csv   # 스키마 오류 없음
python run_all.py --config config_pilot.yaml        # mock으로 전 구간 통과
```
세 번째는 mock이라 LM Studio 없이 돌아간다. **여기서 실패하면 모델을 받기 전에 먼저 고친다.**
mock 산출물은 `results/mock_001/`에 생기므로 실제 결과와 섞이지 않는다.

### 0-2. LM Studio
https://lmstudio.ai 에서 macOS(Apple Silicon)용을 받아 설치한다.
설치 후 **Developer → Local Server → Start** 로 서버를 켠다. 기본 포트 1234.

### 0-3. 모델 내려받기
LM Studio 검색창에 아래 저장소 이름을 그대로 넣는다.
**2026-09-08에 공식 저장소 존재를 확인한 것들이다.**

| 용도 | 저장소 | 양자화 | 크기 | 라이선스 |
|---|---|---|---|---|
| 주모델 | `unsloth/Qwen3.6-35B-A3B-GGUF` | **Q4_K_S** | 20.9GB | Apache 2.0 |
| 재현모델 | `google/gemma-4-12b-it-qat-q4_0-gguf` | **QAT Q4_0** | ~6.98GB | ⚠️ 아래 주의 |
| 플랫폼 | `openai/gpt-oss-20b` | MXFP4 | ~16GB 메모리 | Apache 2.0 |

- 주모델은 `lmstudio-community` Q4_K_M(22.1GB) 대신 **`unsloth` Q4_K_S(20.9GB)**를 쓴다.
  양자화는 조건 **간**이 아니라 조건 **내** 통제변인이므로 프롬프트 7종이 같은 파일을 쓰기만 하면 비교는 성립한다.
  🚨 **파일럿과 본실험이 반드시 같은 파일이어야 한다.** 중간에 바꾸면 앞뒤 데이터를 합칠 수 없다.
- ⚠️ **Gemma 양자화는 Q4_K_M이 아니라 QAT Q4_0이다.** config에 `Q4_0_QAT`로 적어뒀다.
  다른 파일을 받았으면 **받은 파일 이름에 맞춰 config를 고칠 것.**
- ⚠️ **Gemma 라이선스는 확정하지 못했다.** HF 페이지가 Apache 2.0으로 읽혔으나 Gemma 계열은
  통상 별도 이용약관을 쓴다. **논문에 라이선스를 쓰기 전에 원문을 직접 확인할 것.**
- **judge 모델은 아직 안 정했다.** 위 3종과 **다른 계열** 1종을 받아둔다.
  파일럿에서 κ로 고른다(3단계).
- 디스크 여유 **60GB 이상**을 확보한다.

### 0-4. 메모리 규칙 (어기면 스왑으로 실험이 몇 배 느려진다)
**36GB에 21.2GB 모델 두 개는 안 올라간다.**
**응답을 전량 생성 → 모델 언로드 → judge 로드 → 판정**, 이 순서를 지킨다.
`run_all.py`는 생성과 판정을 연달아 하므로, 실제 실행에서는 **단계를 나눠 돌린다**(4단계 참조).

---

## 1. ⛔ 관문 — 프로브 (10분)

**이걸 통과하기 전에는 아무것도 시작하지 않는다.**

LM Studio에 주모델 하나를 로드하고 서버를 켠 뒤:
```bash
python tools/probe_lmstudio.py                     # 모델 id 목록만 보고 끝
python tools/probe_lmstudio.py --model <위에서 본 id>
```

세 가지를 서버에 직접 물어본다. **코드가 알 수 없는 것들이라 추측하지 않고 실제로 물어본다.**

| 확인 항목 | 통과 기준 | 실패하면 |
|---|---|---|
| **seed 반영** | 같은 seed 두 번 → 동일 / 다른 seed → 상이 | **설계 재검토.** 반복 3회가 seed로만 갈리는 구조다. 반영 안 되면 재현이 불가능하고, 그 사실을 논문에 그대로 밝혀야 한다 |
| **thinking 끄기** | 추론 흔적(`reasoning_content`)이 사라지는 후보가 있는지. **수락됐다는 것만으로는 통과가 아니다** — 무시하면서 수락하는 서버가 있다 | 어느 것도 확실하지 않으면 **config의 설정을 지우고 "추론 모드를 끄지 못했다"를 한계에 쓴다.** 되는 척하지 말 것 |
| **처리 속도** | tok/s와 조건별 예상 시간이 출력됨 | 실험 A 원안이 감당 안 되는 시간이면 **조건이나 모델을 줄인다. 문항은 줄이지 않는다** |

출력을 그대로 저장해두고 나에게 주면 config와 문서에 반영한다.

### 1-1. 프로브 실측 결과 (2026-09-08, qwen3.6-35b-a3b / Mac Studio 36GB / ctx 8192)

**[1] seed — 통과.** 같은 seed(1) 두 번 동일 = True, 다른 seed(1 vs 2) 동일 = False.
반복마다 seed만 바꾸는 설계가 그대로 성립한다.

**[2] thinking — `reasoning_effort="none"` 하나만 효과가 있었다.**

| 후보 | 서버 | 추론 흔적 |
|---|---|---|
| (대조군) 아무 것도 안 보냄 | 수락 | 있음 (`reasoning_content`) |
| `extra_body={'thinking': False}` | 수락 | **있음** |
| `extra_body={'enable_thinking': False}` | 수락 | **있음** |
| `extra_body={'chat_template_kwargs': {'enable_thinking': False}}` | 수락 | **있음** |
| `reasoning_effort='low'` | 수락 | 있음 |
| 프롬프트 `/no_think` 접미사 | 수락 | **있음** |
| **`reasoning_effort='none'`** | 수락 | **없음** ✅ |

즉 **수락과 효과는 별개다.** 서버는 넷 다 받아주면서 무시했다 — 거부되지 않았다는 것만 보고
`thinking: false`를 믿었다면 추론이 켜진 채로 실험이 끝났을 것이다. `config.generation.thinking`은
`reasoning_effort`로 교체했고, `src/client.py`의 `extra_body={"thinking": ...}` 경로는 제거했다.

⚠️ **이 판정의 한계 두 가지 — 논문에 그대로 적을 것.**
1. 프로브의 `max_tokens`가 256이라 **7개 후보 전부 `completion_tokens=256`으로 상한에 걸렸다.**
   "토큰이 뚜렷이 줄었는가"라는 원래 판정 기준은 쓸 수 없었고, 판정은 오직
   `reasoning_content` 필드의 유무로 했다. 추론이 실제로 꺼진 것인지 필드만 감춰진 것인지는
   구분하지 못한다. (`reasoning_effort='low'`에서는 흔적이 남는다는 점이 "effort 눈금이 실제로
   먹힌다"는 방증이지만 증명은 아니다.)
2. **검증한 모델은 qwen3.6-35b-a3b 하나뿐이다.** gemma-4-12b-it-qat과 gpt-oss-20b에서도
   같은지는 확인하지 않았다. `src/client.py`에 런타임 누출 검사를 넣어뒀다 —
   `reasoning_effort='none'`인데 응답에 `reasoning_content`가 오면 경고를 한 번 찍는다.
   **실험 로그에 그 경고가 있으면 그 모델은 조건이 다른 것이다.**

**[3] 속도 — 8.18초/응답, 62.6 tok/s.** 5회 실측 8.12~8.24초로 편차가 거의 없다.

| 계획 | 응답 수 | 생성 시간 |
|---|---|---|
| 파일럿 | 140 | 0.3시간 |
| 실험 B | 960 | 2.2시간 |
| 실험 A 축소안 | 6,000 | 13.6시간 |
| 실험 A 원안 | 12,600 | **28.6시간** |

⚠️ **이 숫자는 상한이지 예상치가 아니다.** 5회 모두 `completion_tokens=512`로 상한에 걸렸다 —
즉 "가능한 가장 긴 응답"만 측정했다. 실제 응답은 구조화된 짧은 답이라 훨씬 빠를 것이다.
그리고 이 측정은 추론이 **켜진** 상태였다. `reasoning_effort='none'`을 넣으면 더 줄어든다.
**조건을 줄이는 결정은 파일럿의 실제 평균 `completion_tokens`를 보고 하라 — 지금 하지 말 것.**

⚠️ **judge 시간이 여기 안 들어 있다.** 응답 1건당 1회 호출이므로 실험 A 원안이면 judge도 12,600회다.
게다가 메모리가 36GB라 주모델(22.7GB)과 judge(15.3GB)를 동시에 올릴 수 없어 **생성과 판정이 겹치지
않는다** — 총 시간은 두 단계의 합이다.

---

## 2. 파일럿 (189응답, 프로브 속도로 시간 계산됨)

`config_pilot_real.yaml`을 쓴다. **주모델 1종 × 7조건 × 27문항 × 1반복 = 189응답.**

```bash
# 생성만
python -m src.run_experiment --config config_pilot_real.yaml
```

`run_id`가 `pilot_001`이므로 결과는 `results/pilot_001/`에 쌓인다.
mock과 섞이지 않고, 중단되면 다시 돌려도 이어서 한다(resume).

**파일럿의 목적은 결과를 보는 게 아니다.** 아래 넷을 정하는 것이다.

---

## 3. 파일럿으로 정할 것 넷 — 본실험 전에 전부 동결

### 3-1. 보류 표현 사전 (`data/abstention_lexicon.txt`, `data/false_premise_lexicon.txt`)
🚨 **P0·P2 응답을 먼저 보고 만든 뒤, 그 다음에 P1·P3·P5에만 나오는 표현을 추가한다.**
P1·P3 프롬프트는 "확인할 수 없습니다"를 직접 지정하므로, 사전을 그 표현에 맞춰 만들면
P0·P2의 보류가 과소 탐지되어 **조건 간 차이가 인위적으로 부풀려진다.** **P5(few-shot)도
같은 이유로 배제 대상이다** — P5의 예시 3개가 바로 그 "확인할 수 없습니다"라는 P1·P3
전용 문구를 그대로 쓰고 있어서 동일한 오염원이기 때문이다.

두 사전은 서로 다른 현상을 담는다: `abstention_lexicon.txt`는 순수 보류("모르겠다")
표현, `false_premise_lexicon.txt`는 잘못된 전제 지적("존재하지 않는다", "잘못된
전제다") 표현이다. 둘 다 위 동결 규칙(P0·P2 먼저, P1·P3·P5 제외)을 그대로 따른다.

```bash
# P0/P2 응답만 먼저 훑는다
python - <<'PY'
import pandas as pd
df = pd.read_csv("results/pilot_001/raw_responses.csv", dtype=str)
for pt in ["P0", "P2"]:
    print("="*60, pt)
    for t in df[df.prompt_type==pt].response_final.head(40):
        print("-", str(t)[:120])
PY
```
확정한 뒤 **본실험 결과를 보고 사후에 고치지 않는다.**

### 3-2. judge 모델 선정
파일럿 응답 중 **80~100건을 사람이 직접 라벨링**한 뒤, judge 후보들의 자동 라벨과 비교해
**Cohen's κ**가 가장 높은 것을 고른다. `src/agreement.py`가 κ를 계산한다.
**"다른 계열이니까 믿을 만하다"는 근거가 아니다. 숫자로 고른다.**

### 3-3. 천장·바닥 효과 점검
P0에서 정답률 100%인 `hard_factual` 문항은 난이도 재배정(easy로 이동)을 검토한다.
반대로 전 조건에서 0%인 문항도 본다. **본실험 데이터를 보고 옮기면 안 되므로 지금 한다.**

### 3-4. 소요 시간 재계산
파일럿 실측으로 본실험 시간을 다시 계산한다. 안 맞으면 **조건 또는 모델을 줄인다.**

---

## 4. 본실험

`config.yaml`을 편집한다. 반드시 바꿀 것:
- `run_id`: `main_001` 같은 **새 이름**. mock이나 파일럿 id를 재사용하지 말 것
- `mock: false`
- `models[].lms_id`: LM Studio에 실제로 뜨는 id로 교체
- `judge.lms_id`: 3-2에서 고른 모델로 교체 (`judge-placeholder` 그대로 두면 안 됨)

**같은 `run_id`에 `mock`만 바꿔 돌리면 실행이 중단된다.** 일부러 그렇게 막아뒀다 —
예전에는 조용히 건너뛰어서 실제 생성이 통째로 빠졌다.

### 4-1. 생성 (모델만 로드)
```bash
python -m src.run_experiment --config config.yaml
```
모델별로 순회하므로, **모델을 바꿀 때 LM Studio에서 이전 모델을 언로드**한다.
중단돼도 다시 돌리면 이어서 한다.

### 4-2. 판정 (피험 모델 언로드 → judge 로드)
```bash
# LM Studio에서 피험 모델 전부 언로드하고 judge만 로드한 뒤
python -m src.evaluate --config config.yaml
```

### 4-3. 분석
```bash
python -m src.analyze --config config.yaml
```
`results/<run_id>/`에 `summary.csv`·`stats.txt`, `graphs/<run_id>/`에 PNG 4종이 나온다.

⚠️ **`JUDGE_ERROR`가 5%를 넘으면 분석이 거부하고 중단한다.** judge를 고치고 4-2부터 다시 한다.
로그에 몇 건인지 찍히니 무시하지 말 것.

---

## 5. 인간 검증 (400건)
```bash
python -m src.sample_for_human --config config.yaml --n 400
```
`config`의 `run_id`를 읽어 `results/<run_id>/evaluated.csv`에서 뽑고,
`results/<run_id>/human_sample.csv`와 `..._blind.csv` 두 개를 만든다.
**블라인드 파일에는 자동 판정·프롬프트 조건·모델 이름이 가려져 있다.** 그 파일로 라벨링한다.

`human_label` 열을 채운 뒤 (경로는 실제 파일에 맞춰 쓸 것):
```bash
python -m src.agreement   --full results/<run_id>/human_sample.csv   results/<run_id>/human_sample_blind.csv
```
`agreement`는 `--config`를 받지 않는다. **위치 인자로 평가자 파일을, `--full`로 원본을 준다.**
2명이 평가했으면 두 번째 파일을 뒤에 하나 더 붙이면 평가자 간 κ도 같이 나온다:
```bash
python -m src.agreement --full results/<run_id>/human_sample.csv   rater1_blind.csv rater2_blind.csv
```
가능하면 **2명이 같은 표본을 독립 평가**한다.

---

## 6. 제출 직전
- **RISS 재검색** (동일 주제가 잡히면 -2)
- 참고문헌 APA, **존재하지 않는 문헌·DOI·링크가 하나라도 있으면 -2**
- `results/`·`graphs/`에 남은 mock 산출물이 논문에 섞이지 않았는지 확인
  (`results/legacy_mock/`은 옛 mock이다. 논문에 쓰지 말 것)

---

## 자주 걸리는 곳

| 증상 | 원인 |
|---|---|
| `RuntimeError: run_id ... 이미 is_mock=True로 기록` | 의도된 차단. **새 run_id를 쓸 것** |
| 생성이 0건이고 전부 resume로 건너뜀 | 같은 run_id로 이미 돌렸다. 새 id를 쓰거나 그 디렉터리를 지운다 |
| `JUDGE_ERROR` 대량 발생 | judge 모델이 언로드됐거나 서버가 죽었다. 되살리고 4-2 재실행 |
| 매우 느림 / 스왑 | 피험 모델과 judge가 동시에 올라가 있다. 언로드할 것 |
| 3회 반복 응답이 전부 같음 | temperature나 seed가 반영 안 되고 있다. 1단계 프로브로 확인 |
