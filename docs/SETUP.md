# 처음부터 따라 하기 — 맥에서 실험 환경 만들기

컴퓨터를 새로 켠 상태에서 시작해 실험을 돌릴 수 있는 데까지 간다.
**위에서부터 순서대로, 건너뛰지 말고.** 각 단계 끝에 "이렇게 나오면 성공"이 있다.
그것과 다르게 나오면 다음으로 가지 말고 맨 아래 §부록을 본다.

---

## 0. 터미널 쓰는 법 (이미 알면 §1로)

**터미널 여는 법**: `⌘(커맨드) + 스페이스` → `터미널` 입력 → 엔터.
검은(또는 흰) 창이 뜨고 커서가 깜빡인다. 여기에 명령을 붙여넣고 엔터를 친다.

**규칙 세 가지**
1. 이 문서의 회색 상자 안 내용을 **한 줄씩** 복사해서 붙여넣고 엔터.
2. 비밀번호를 물으면 입력한다. **화면에 아무것도 안 보이는 게 정상이다.** 그냥 치고 엔터.
3. 빨간 글씨나 `error`가 보이면 멈추고 §부록을 본다. 무시하고 계속 가면 나중에 원인을 못 찾는다.

**명령이 끝났는지 아는 법**: 다시 `이름@맥 ~ %` 같은 줄이 뜨고 커서가 깜빡이면 끝난 것이다.

---

## 1. 준비물 확인

- **맥 스튜디오 M4 Max, 메모리 36GB** (본실험 환경)
- **디스크 여유 60GB 이상** — 모델 파일이 크다
  확인: 화면 왼쪽 위 사과 → `이 Mac에 관하여` → `저장 공간`
- 인터넷 (모델 다운로드에 수십 GB 받는다)

---

## 2. 파이썬 설치

맥에 파이썬이 이미 있지만 **버전이 낮아서 못 쓴다.** 확인해 보자.

```bash
python3 --version
```

`Python 3.9.6`처럼 **3.10보다 낮게** 나오면 새로 깔아야 한다.
`Python 3.11.x` 이상이면 §3으로 건너뛴다.

### 설치 방법 (둘 중 하나만)

**방법 A — 설치 프로그램 (쉬움, 추천)**
1. https://www.python.org/downloads/macos/ 접속
2. `Latest Python 3 Release` 중 **3.11 이상**의 `macOS 64-bit universal2 installer` 다운로드
3. 받은 `.pkg` 파일을 두 번 눌러 실행 → 계속 → 동의 → 설치
4. 터미널을 **완전히 끄고 다시 연다** (중요)

**방법 B — Homebrew (터미널에 익숙하면)**
```bash
brew install python@3.11
```

### 이렇게 나오면 성공
```bash
python3 --version
```
→ `Python 3.11.x` 또는 그 이상

---

## 3. 프로젝트 내려받기

두 가지 방법이 있다. **B를 권한다** — 나중에 수정본을 받기 편하다.

### 방법 A — 압축 파일 (git을 안 써도 됨)
1. 브라우저에서 https://github.com/beomtaeha1117/LLM_hallucination_experiment 접속
2. 초록색 **`< > Code`** 버튼 클릭
3. 아래쪽 **`Download ZIP`** 클릭
4. `다운로드` 폴더에 `LLM_hallucination_experiment-main.zip`이 생긴다
5. 파일을 두 번 눌러 압축을 푼다 → `LLM_hallucination_experiment-main` 폴더가 생긴다
6. 그 폴더를 **`서류` 폴더로 옮긴다** (경로가 짧아야 나중에 편하다)

⚠️ 이 방법은 **나중에 수정본을 받으려면 ZIP을 다시 받아야 한다.** 그때 `results/` 폴더를
새 폴더로 옮기는 걸 잊지 말 것 — 실험 결과가 거기 있다.

### 방법 B — git (권장)
```bash
cd ~/Documents
```
```bash
git clone https://github.com/beomtaeha1117/LLM_hallucination_experiment.git
```
`git`이 없다고 나오면 설치 창이 뜬다. `설치`를 누르고 끝날 때까지 기다린 뒤 다시 실행한다.

나중에 수정본을 받을 때는 프로젝트 폴더에서 이것만 치면 된다:
```bash
git pull
```

---

## 4. 터미널을 프로젝트 폴더로 옮기기

**모든 명령은 프로젝트 폴더 안에서 실행해야 한다.** 아니면 "파일이 없다"는 오류가 난다.

```bash
cd ~/Documents/LLM_hallucination_experiment
```

ZIP으로 받았으면 폴더 이름이 `-main`으로 끝난다:
```bash
cd ~/Documents/LLM_hallucination_experiment-main
```

**경로를 모르겠으면**: 터미널에 `cd ` 를 치고 **(cd 뒤에 띄어쓰기 하나)**,
Finder에서 프로젝트 폴더를 터미널 창으로 **끌어다 놓은 뒤** 엔터. 경로가 자동으로 입력된다.

### 이렇게 나오면 성공
```bash
ls
```
→ `README.md  config.yaml  data  docs  prompts  src  tools ...` 가 보인다.
안 보이면 폴더를 잘못 찾은 것이다.

---

## 5. 가상환경 만들고 패키지 설치

"가상환경"은 이 프로젝트 전용 파이썬 상자다. 맥 전체를 건드리지 않아 안전하다.

```bash
python3 -m venv .venv
```
```bash
source .venv/bin/activate
```

성공하면 프롬프트 맨 앞에 **`(.venv)`** 가 붙는다. 이게 안 보이면 다음 명령이 엉뚱한 데 설치된다.

⚠️ **터미널을 새로 열 때마다 `source .venv/bin/activate`를 다시 쳐야 한다.**
`(.venv)`가 안 보이면 이것부터 친다.

```bash
pip install -r requirements.txt
```
```bash
pip install pytest
```

몇 분 걸린다. 노란 경고는 대개 무시해도 되지만 **빨간 `ERROR`는 무시하면 안 된다.**

### 이렇게 나오면 성공
```bash
python -c "import pandas, numpy, statsmodels, matplotlib, openai; print('전부 설치됨')"
```
→ `전부 설치됨`

---

## 6. ⭐ 모델 없이 먼저 확인 (여기서 실패하면 21GB 받기 전에 고친다)

```bash
pytest tests/ -q
```
→ 이렇게 나오면 성공: `12 passed`

```bash
python tools/validate_questions.py data/questions_v1_DRAFT.csv
```
→ `검증 통과: 스키마 오류 없음.` / `총 행수: 200` / `answerable 균형: Y=100, N=100`

```bash
python run_all.py --config config_pilot.yaml
```
→ 마지막에 `>>> [analyze] 완료` 가 나오면 성공. 1분쯤 걸린다.

이건 **가짜(mock) 데이터로 파이프라인 전체를 돌려보는 것**이다. LM Studio가 없어도 된다.
결과는 `results/mock_001/`에 생기고 실제 실험 결과와 섞이지 않는다.

**세 개 다 통과했으면 소프트웨어는 끝났다.** 이제 모델을 받는다.

---

## 7. LM Studio 설치

1. https://lmstudio.ai 접속
2. **Download for Mac (Apple Silicon)** 클릭
3. 받은 파일을 열고 **LM Studio 아이콘을 Applications 폴더로 끌어다 놓는다**
4. `런치패드` 또는 `응용 프로그램`에서 LM Studio 실행
5. "확인되지 않은 개발자" 경고가 뜨면 → 사과 → `시스템 설정` → `개인정보 보호 및 보안` →
   아래로 내려 `확인 없이 열기`

처음 켜면 온보딩 화면이 나온다. 건너뛰어도 된다.

---

## 8. 모델 내려받기

LM Studio 왼쪽의 **돋보기(검색)** 아이콘을 누르고, 아래 이름을 **그대로 복사해서** 검색창에 넣는다.

**2026-09-08에 공식 저장소에 실재함을 확인한 것들이다.**

| 순서 | 검색어 | 고를 파일 | 크기 |
|---|---|---|---|
| 1 (필수) | `lmstudio-community/Qwen3.6-35B-A3B-GGUF` | **Q4_K_M** | 21.2GB |
| 2 (필수) | `google/gemma-4-12b-it-qat-q4_0-gguf` | **QAT Q4_0** | 약 7GB |
| 3 (선택) | `openai/gpt-oss-20b` | MXFP4 | 약 16GB |
| 4 (judge) | 위 셋과 **다른 계열** 아무거나 1종 | — | — |

- ⚠️ **2번은 Q4_K_M이 아니라 Q4_0이다.** 헷갈리기 쉽다.
- **3번은 시간이 빠듯하면 나중에.** 1·2번만으로 실험이 성립한다.
- **4번(judge)은 파일럿에서 정한다.** 지금은 후보 하나만 받아두면 된다.
- 다운로드는 오래 걸린다. 인터넷이 끊기면 LM Studio가 이어받는다.

### 서버 켜기
1. LM Studio 왼쪽에서 **`Developer`**(또는 `Local Server` / `<>` 아이콘)를 찾는다
2. 위쪽에서 모델 하나를 **Load(로드)** 한다 — 처음엔 Qwen
3. **`Start Server`** 를 누른다
4. `http://localhost:1234` 같은 주소가 보이면 켜진 것이다

> 버전에 따라 메뉴 이름이 조금 다를 수 있다. **"모델을 로드하고, 로컬 서버를 켠다"** 이 두 가지만 하면 된다.

### 이렇게 나오면 성공
터미널에서 (`(.venv)` 붙어 있는 상태로):
```bash
curl -s http://localhost:1234/v1/models
```
→ 모델 이름이 들어간 긴 글자들이 나오면 성공. 아무것도 안 나오면 서버가 안 켜진 것이다.

---

## 9. ⛔ 관문 — 프로브 (10분)

**이걸 통과하기 전에는 실험을 시작하지 않는다.**

```bash
python tools/probe_lmstudio.py
```
먼저 이렇게 치면 **로드된 모델의 정확한 이름(id)** 을 보여주고 끝난다. 그 이름을 복사한다.

```bash
python tools/probe_lmstudio.py --model 여기에_복사한_이름
```

세 가지를 서버에 직접 물어본다.

| 확인 | 뭘 보는가 | 왜 중요한가 |
|---|---|---|
| **seed** | 같은 seed 두 번 → 같아야 / 다른 seed → 달라야 | 안 되면 3회 반복이 **같은 응답 3개**가 된다. 실험이 무너진다 |
| **thinking** | 어느 방법이 먹히는지 | 못 끄면 설정 파일이 실제와 다른 말을 하게 된다 |
| **속도** | tok/s와 예상 소요 시간 | 실험을 며칠에 끝낼지 여기서 결정된다 |

**출력을 통째로 복사해서 나에게 주면** config와 문서에 반영한다.
셋 중 하나라도 이상하면 그 상태로 진행하지 말 것.

---

## 10. 파일럿 (189응답)

```bash
open -e config_pilot_real.yaml
```
텍스트 편집기가 열린다. `lms_id: "qwen3.6-35b-a3b"` 부분을 **9단계에서 본 실제 모델 이름**으로
바꾸고 저장(`⌘S`)한 뒤 닫는다.

```bash
python -m src.run_experiment --config config_pilot_real.yaml
```

진행 막대가 나온다. 중간에 꺼져도 다시 같은 명령을 치면 **이어서** 한다.
결과는 `results/pilot_001/`에 쌓인다.

여기까지 오면 그다음은 `docs/RUNBOOK.md` §3부터 이어서 보면 된다.
(보류 표현 사전 만들기 → judge 고르기 → 본실험)

---

## 부록 — 자주 나는 오류

| 화면에 나온 말 | 뜻 / 해결 |
|---|---|
| `command not found: python3` | 파이썬이 없다. §2로 |
| `No module named pandas` | `(.venv)`가 안 붙어 있다. `source .venv/bin/activate` 후 다시 |
| `No such file or directory` | 프로젝트 폴더 밖이다. §4의 `cd`를 다시 |
| `zsh: permission denied` | 명령 앞에 `sudo `를 붙이지 말고, 폴더 위치부터 확인 |
| `Connection error` / `서버에 붙지 못했습니다` | LM Studio 서버가 꺼져 있다. §8 서버 켜기 |
| `RuntimeError: run_id ... 이미 is_mock=True로 기록` | **일부러 막은 것이다.** config의 `run_id`를 새 이름으로 바꾼다 |
| 생성이 0건이고 전부 건너뜀 | 같은 `run_id`로 이미 돌렸다. 새 이름을 쓴다 |
| 맥이 아주 느려지고 선풍기 소리 | 모델 두 개가 동시에 올라가 있다. LM Studio에서 하나를 **Unload** |
| 3회 반복 응답이 전부 똑같음 | seed나 temperature가 안 먹고 있다. §9 프로브로 확인 |

**막히면 그 화면을 그대로 복사해서 물어보면 된다.** 추측해서 넘기지 말 것.
