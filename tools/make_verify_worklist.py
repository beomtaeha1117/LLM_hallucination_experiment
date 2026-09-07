"""검증 워크리스트 생성 도구.

verified != YES 인 모든 문항에 대해, 사람이 바로 클릭해서 검색할 수 있는
링크와 확인해야 할 주장(claim)을 담은 작업 목록을 HTML(자기완결형)과
CSV 두 형태로 만든다.

사용법:
    .venv/bin/python tools/make_verify_worklist.py \
        --questions data/questions_v1_DRAFT.csv \
        --out verify/worklist.html \
        --csv verify/worklist.csv
"""

from __future__ import annotations

import argparse
import csv
import html
import os
import re
import sys
from urllib.parse import quote_plus

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from schema import load_questions  # noqa: E402

GOOGLE = "https://www.google.com/search?q="
RISS = "http://www.riss.kr/search/Search.do?query="
SCHOLAR = "https://scholar.google.com/scholar?q="
LAW_GO_KR = "https://www.law.go.kr/lsSc.do?query="

# 사람이 읽을 question_type 이름
TYPE_LABELS = {
    "easy_factual": "A1 쉬운 factual",
    "hard_factual": "A2 헷갈리는 factual",
    "numeric": "A3 수치·연도",
    "context_qa": "A4 context QA (답 있음)",
    "context_qa_nogold": "context QA (골드 없음)",
    "fake_paper": "U1 없는 논문",
    "fake_concept": "U2 없는 개념",
    "fake_statute": "U5 없는 조항",
    "false_premise": "U3 잘못된 전제",
    "unknowable": "U4 원리적 불가",
}

# 표시 순서 (question-design.md 순서를 따름)
TYPE_ORDER = [
    "easy_factual",
    "hard_factual",
    "numeric",
    "context_qa",
    "fake_paper",
    "fake_concept",
    "false_premise",
    "unknowable",
    "fake_statute",
    "context_qa_nogold",
]

QUOTE_RE = re.compile(r"'([^']+)'")
CORNER_RE = re.compile(r"「([^」]+)」")


class WorkItem:
    """한 문항에 대한 작업 항목: 확인할 주장 + 검색 링크들."""

    def __init__(self, row: pd.Series):
        self.question_id = row["question_id"]
        self.question_type = row["question_type"]
        self.question_subtype = row["question_subtype"]
        self.question = row["question"]
        self.answerable = row["answerable"]
        self.ground_truth = row["ground_truth"]
        self.acceptable_answers = row["acceptable_answers"]
        self.why_unanswerable = row["why_unanswerable"]
        self.verify_action = row["verify_action"]
        self.extract_failed = False
        self.claim = self._build_claim()
        self.note = ""
        self.search_urls: list[tuple[str, str]] = self._build_search_urls()

    def _build_claim(self) -> str:
        if self.answerable == "Y":
            parts = [f"ground_truth: {self.ground_truth}"]
            if self.acceptable_answers:
                parts.append(f"acceptable_answers: {self.acceptable_answers}")
            return " / ".join(parts)
        return f"why_unanswerable (NOT 존재해야 함): {self.why_unanswerable}"

    def _build_search_urls(self) -> list[tuple[str, str]]:
        qtype = self.question_type
        urls: list[tuple[str, str]] = []

        if qtype in ("fake_concept",):
            term = self._extract(QUOTE_RE, self.question)
            urls = [
                ("Google", GOOGLE + quote_plus(term)),
                ("RISS", RISS + quote_plus(term)),
                ("Google Scholar", SCHOLAR + quote_plus(term)),
            ]
        elif qtype in ("fake_paper",):
            term = self._extract(CORNER_RE, self.question)
            urls = [
                ("Google", GOOGLE + quote_plus(term)),
                ("RISS", RISS + quote_plus(term)),
                ("Google Scholar", SCHOLAR + quote_plus(term)),
            ]
        elif qtype == "fake_statute":
            term = self._extract(CORNER_RE, self.question)
            urls = [("국가법령정보센터", LAW_GO_KR + quote_plus(term))]
        elif qtype == "false_premise":
            entity = self._extract_entity(self.question)
            urls = [("Google", GOOGLE + quote_plus(entity))]
            self.note = "why_unanswerable에 적힌 날짜/수치만 확인하면 됨"
        elif qtype in ("easy_factual", "hard_factual", "numeric"):
            urls = [("Google", GOOGLE + quote_plus(self.question))]
            self.note = "출처 URL을 반드시 기록할 것"
        elif qtype in ("context_qa", "context_qa_nogold"):
            urls = []
            self.note = "검색 불필요 — 이미 verified=YES 여야 함"
        else:
            urls = [("Google", GOOGLE + quote_plus(self.question))]

        return urls

    def _extract(self, pattern: re.Pattern, text: str) -> str:
        m = pattern.search(text)
        if m:
            return m.group(1)
        self.extract_failed = True
        self.note = (self.note + " " if self.note else "") + "추출실패"
        return text

    def _extract_entity(self, text: str) -> str:
        # false_premise: 핵심 고유명사를 뽑아내기 어려우므로 질문 전체를 검색어로 쓴다.
        # (구두점만 제거)
        cleaned = re.sub(r"[?？]", "", text).strip()
        return cleaned


def build_items(df: pd.DataFrame) -> list[WorkItem]:
    pending = df[df["verified"] != "YES"]
    return [WorkItem(row) for _, row in pending.iterrows()]


def write_csv(items: list[WorkItem], path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "question_id",
                "question_type",
                "question",
                "claim",
                "verify_action",
                "search_url_1",
                "search_url_2",
                "search_url_3",
                "verified_by_human",
                "source_url",
                "note",
            ]
        )
        for it in items:
            urls = [u for _, u in it.search_urls]
            urls += [""] * (3 - len(urls))
            writer.writerow(
                [
                    it.question_id,
                    it.question_type,
                    it.question,
                    it.claim,
                    it.verify_action,
                    urls[0],
                    urls[1],
                    urls[2],
                    "",
                    "",
                    it.note,
                ]
            )


def _esc(s: str) -> str:
    return html.escape(str(s), quote=True)


def render_html(items: list[WorkItem]) -> str:
    by_type: dict[str, list[WorkItem]] = {}
    for it in items:
        by_type.setdefault(it.question_type, []).append(it)

    ordered_types = [t for t in TYPE_ORDER if t in by_type] + [
        t for t in by_type if t not in TYPE_ORDER
    ]

    nav_links = "\n".join(
        f'<a href="#sec-{_esc(t)}">{_esc(TYPE_LABELS.get(t, t))} '
        f'({len(by_type[t])})</a>'
        for t in ordered_types
    )

    sections = []
    for t in ordered_types:
        cards = []
        for it in by_type[t]:
            search_html = ""
            if it.search_urls:
                links = "\n".join(
                    f'<a href="{_esc(u)}" target="_blank" rel="noopener">{_esc(label)}</a>'
                    for label, u in it.search_urls
                )
                search_html = f'<div class="links">{links}</div>'
            else:
                search_html = '<div class="links muted">검색 불필요</div>'

            note_html = f'<div class="note">{_esc(it.note)}</div>' if it.note else ""

            cards.append(f"""
<div class="card" data-qid="{_esc(it.question_id)}">
  <div class="card-head">
    <span class="qid">{_esc(it.question_id)}</span>
    <span class="qtype">{_esc(TYPE_LABELS.get(it.question_type, it.question_type))}</span>
    {'<span class="subtype">' + _esc(it.question_subtype) + '</span>' if it.question_subtype else ''}
  </div>
  <div class="question">{_esc(it.question)}</div>
  <div class="claim"><b>확인할 주장:</b> {_esc(it.claim)}</div>
  <div class="verify-action"><b>검증 방법 힌트:</b> {_esc(it.verify_action)}</div>
  {search_html}
  {note_html}
  <div class="controls">
    <label class="chk-label">
      <input type="checkbox" class="chk" />
      확인 완료
    </label>
    <input type="text" class="src-url" placeholder="출처 URL 붙여넣기" />
  </div>
</div>""")

        sections.append(f"""
<section id="sec-{_esc(t)}">
  <h2>{_esc(TYPE_LABELS.get(t, t))} <span class="count">({len(by_type[t])}건)</span></h2>
  {''.join(cards)}
</section>""")

    total = len(items)
    all_ids = [it.question_id for it in items]

    return f"""<!doctype html>
<html lang="ko">
<head>
<meta charset="utf-8" />
<title>검증 워크리스트</title>
<style>
  body {{
    font-family: -apple-system, "Malgun Gothic", "Apple SD Gothic Neo", sans-serif;
    background: #f7f7f5;
    color: #222;
    margin: 0;
    padding: 0;
  }}
  header {{
    position: sticky;
    top: 0;
    background: #fff;
    border-bottom: 1px solid #ddd;
    padding: 12px 20px;
    z-index: 10;
  }}
  header h1 {{
    font-size: 18px;
    margin: 0 0 8px 0;
  }}
  #progress {{
    font-weight: bold;
    font-size: 15px;
    margin-bottom: 8px;
  }}
  #progress-bar-outer {{
    background: #e5e5e5;
    border-radius: 4px;
    height: 8px;
    width: 100%;
    max-width: 400px;
    overflow: hidden;
  }}
  #progress-bar-inner {{
    background: #4a7; height: 100%; width: 0%;
  }}
  nav {{
    margin-top: 10px;
    display: flex;
    flex-wrap: wrap;
    gap: 6px;
  }}
  nav a {{
    font-size: 12px;
    background: #eee;
    padding: 4px 8px;
    border-radius: 4px;
    text-decoration: none;
    color: #333;
  }}
  nav a:hover {{ background: #ddd; }}
  #download-btn {{
    margin-top: 10px;
    padding: 6px 14px;
    font-size: 13px;
    cursor: pointer;
  }}
  main {{ padding: 16px 20px 60px; max-width: 900px; margin: 0 auto; }}
  section h2 {{ font-size: 16px; border-bottom: 2px solid #333; padding-bottom: 4px; margin-top: 32px; }}
  .count {{ color: #888; font-weight: normal; font-size: 13px; }}
  .card {{
    background: #fff;
    border: 1px solid #ddd;
    border-radius: 6px;
    padding: 12px 14px;
    margin: 10px 0;
  }}
  .card.done {{ background: #eefaf0; border-color: #bde5c8; }}
  .card-head {{ font-size: 12px; color: #666; margin-bottom: 4px; }}
  .qid {{ font-weight: bold; color: #000; margin-right: 8px; }}
  .qtype {{ margin-right: 8px; }}
  .subtype {{ color: #999; }}
  .question {{ font-size: 15px; margin: 6px 0; }}
  .claim {{ font-size: 13px; margin: 4px 0; color: #333; }}
  .verify-action {{ font-size: 12px; margin: 4px 0; color: #666; }}
  .links {{ margin: 8px 0; display: flex; gap: 8px; flex-wrap: wrap; }}
  .links a {{
    font-size: 12px;
    background: #f0f4ff;
    border: 1px solid #cdd8f5;
    padding: 3px 8px;
    border-radius: 4px;
    text-decoration: none;
    color: #2a4;
    color: #345;
  }}
  .links.muted {{ color: #999; font-size: 12px; }}
  .note {{ font-size: 12px; color: #b45; margin: 4px 0; }}
  .controls {{
    margin-top: 8px;
    display: flex;
    align-items: center;
    gap: 12px;
    flex-wrap: wrap;
  }}
  .chk-label {{ font-size: 13px; display: flex; align-items: center; gap: 4px; }}
  .src-url {{
    flex: 1;
    min-width: 200px;
    font-size: 13px;
    padding: 4px 6px;
    border: 1px solid #ccc;
    border-radius: 4px;
  }}
</style>
</head>
<body>
<header>
  <h1>검증 워크리스트 — verified != YES ({total}건)</h1>
  <div id="progress">확인 완료 0 / {total}</div>
  <div id="progress-bar-outer"><div id="progress-bar-inner"></div></div>
  <button id="download-btn">결과를 CSV로 내려받기</button>
  <nav>{nav_links}</nav>
</header>
<main>
{''.join(sections)}
</main>
<script>
(function() {{
  var ALL_IDS = {all_ids!r};
  var STORAGE_KEY = "verify_worklist_v1";

  function loadState() {{
    try {{
      var raw = localStorage.getItem(STORAGE_KEY);
      return raw ? JSON.parse(raw) : {{}};
    }} catch (e) {{
      return {{}};
    }}
  }}

  function saveState(state) {{
    try {{
      localStorage.setItem(STORAGE_KEY, JSON.stringify(state));
    }} catch (e) {{
      // localStorage 사용 불가 (private mode 등) — 조용히 무시하고 계속 동작
    }}
  }}

  var state = loadState();

  function updateProgress() {{
    var done = 0;
    ALL_IDS.forEach(function(qid) {{
      if (state[qid] && state[qid].checked) done++;
    }});
    document.getElementById("progress").textContent =
      "확인 완료 " + done + " / " + ALL_IDS.length;
    var pct = ALL_IDS.length ? (done / ALL_IDS.length * 100) : 0;
    document.getElementById("progress-bar-inner").style.width = pct + "%";
  }}

  function applyStateToCard(card) {{
    var qid = card.getAttribute("data-qid");
    var s = state[qid] || {{}};
    var chk = card.querySelector(".chk");
    var src = card.querySelector(".src-url");
    chk.checked = !!s.checked;
    src.value = s.sourceUrl || "";
    card.classList.toggle("done", !!s.checked);
  }}

  document.querySelectorAll(".card").forEach(function(card) {{
    applyStateToCard(card);
    var qid = card.getAttribute("data-qid");
    var chk = card.querySelector(".chk");
    var src = card.querySelector(".src-url");

    function persist() {{
      state[qid] = {{
        checked: chk.checked,
        sourceUrl: src.value
      }};
      saveState(state);
      card.classList.toggle("done", chk.checked);
      updateProgress();
    }}

    chk.addEventListener("change", persist);
    src.addEventListener("input", persist);
  }});

  updateProgress();

  document.getElementById("download-btn").addEventListener("click", function() {{
    var rows = [["question_id", "checked", "source_url"]];
    ALL_IDS.forEach(function(qid) {{
      var s = state[qid] || {{}};
      rows.push([qid, s.checked ? "YES" : "NO", s.sourceUrl || ""]);
    }});
    var csv = rows.map(function(r) {{
      return r.map(function(cell) {{
        var v = String(cell).replace(/"/g, '""');
        return '"' + v + '"';
      }}).join(",");
    }}).join("\\n");
    var blob = new Blob(["\\ufeff" + csv], {{ type: "text/csv;charset=utf-8;" }});
    var url = URL.createObjectURL(blob);
    var a = document.createElement("a");
    a.href = url;
    a.download = "verify_results.csv";
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  }});
}})();
</script>
</body>
</html>
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", required=True, help="문항 CSV 경로")
    parser.add_argument("--out", required=True, help="출력 HTML 경로")
    parser.add_argument("--csv", required=True, help="출력 CSV 경로")
    args = parser.parse_args()

    try:
        df = load_questions(args.questions)
    except ValueError as e:
        print("경고: 문항 파일이 스키마 검증을 통과하지 못했습니다. 원본으로 진행합니다.")
        print(str(e))
        df = pd.read_csv(args.questions, dtype=str, keep_default_na=False, encoding="utf-8-sig")

    items = build_items(df)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(render_html(items))

    write_csv(items, args.csv)

    print(f"작업 항목 수: {len(items)}")
    by_type_counts = {}
    for it in items:
        by_type_counts[it.question_type] = by_type_counts.get(it.question_type, 0) + 1
    for t in TYPE_ORDER:
        if t in by_type_counts:
            print(f"  {t:<20} {by_type_counts[t]}")
    extract_failed = [it.question_id for it in items if it.extract_failed]
    if extract_failed:
        print(f"\n추출실패: {len(extract_failed)}건 -> {', '.join(extract_failed)}")
    print(f"\nHTML: {args.out}")
    print(f"CSV:  {args.csv}")


if __name__ == "__main__":
    main()
