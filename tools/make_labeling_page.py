"""라벨링 시트를 브라우저에서 클릭으로 채우는 페이지를 만든다.

Numbers/엑셀로 100건을 매기는 것은 느리고 위험하다. 응답이 길어 셀 안에서
읽기 어렵고, CSV 덮어쓰기가 번거로워 중간 저장을 빼먹게 되며, 그러면 몇 시간치가
한 번에 날아간다. 이 페이지는 한 건씩 크게 보여주고 누를 때마다 브라우저에
저장하므로, 닫았다 켜도 이어서 할 수 있다.

    python tools/make_labeling_page.py
    open results/human_labeling/labeling.html

다 매긴 뒤 페이지에서 CSV를 내려받아 원래 자리에 덮어쓴다.
"""
from __future__ import annotations

import argparse
import html
import json
import os

import pandas as pd

TEMPLATE = """<!doctype html>
<meta charset="utf-8">
<title>라벨링 — __N__건</title>
<style>
 :root { color-scheme: light dark; }
 body { font: 15px/1.7 -apple-system, BlinkMacSystemFont, sans-serif; margin: 0;
        background: #f5f5f7; color: #1d1d1f; }
 @media (prefers-color-scheme: dark) { body { background:#1c1c1e; color:#f5f5f7; } }
 header { position: sticky; top: 0; background: inherit; padding: 12px 20px;
          border-bottom: 1px solid #8884; display: flex; gap: 16px; align-items: center; }
 .bar { flex: 1; height: 6px; background: #8883; border-radius: 3px; overflow: hidden; }
 .bar > i { display: block; height: 100%; background: #0a84ff; width: 0; }
 main { max-width: 860px; margin: 0 auto; padding: 20px; }
 .card { background: #fff; border-radius: 12px; padding: 20px; margin-bottom: 16px; }
 @media (prefers-color-scheme: dark) { .card { background: #2c2c2e; } }
 .tag { display: inline-block; font-size: 12px; padding: 2px 8px; border-radius: 6px;
        background: #8883; margin-right: 6px; }
 .rule { padding: 10px 14px; border-radius: 8px; margin: 10px 0 0; font-size: 14px; }
 .rule.y { background: #0a84ff22; }
 .rule.n { background: #ff9f0a22; }
 h3 { margin: 18px 0 6px; font-size: 13px; letter-spacing: .04em; opacity: .6; }
 .txt { white-space: pre-wrap; word-break: break-word; }
 .resp { background: #8881; padding: 14px; border-radius: 8px; }
 .btns { display: flex; gap: 10px; flex-wrap: wrap; margin-top: 8px; }
 button { font: inherit; padding: 12px 18px; border-radius: 10px; border: 1px solid #8886;
          background: #8881; cursor: pointer; color: inherit; }
 button:hover { background: #8883; }
 button.c { border-color: #30d158; } button.h { border-color: #ff453a; }
 button.a { border-color: #ff9f0a; }
 button.on { background: #0a84ff; color: #fff; border-color: #0a84ff; }
 .nav { display: flex; gap: 10px; justify-content: space-between; margin-top: 20px; }
 input[type=text] { font: inherit; width: 100%; padding: 10px; border-radius: 8px;
                    border: 1px solid #8886; background: transparent; color: inherit; }
 .done { color: #30d158; font-weight: 600; }
</style>
<header>
  <b id="pos"></b>
  <div class="bar"><i id="fill"></i></div>
  <span id="cnt"></span>
  <button onclick="save()">CSV 내려받기</button>
</header>
<main>
  <div class="card">
    <div id="meta"></div>
    <div id="rule" class="rule"></div>
    <h3>질문</h3><div id="q" class="txt"></div>
    <div id="ctxwrap" hidden><h3>근거 지문</h3><div id="ctx" class="txt"></div></div>
    <div id="gtwrap" hidden><h3>정답</h3><div id="gt" class="txt"></div></div>
    <div id="whywrap" hidden><h3>왜 답할 수 없는 문항인가</h3><div id="why" class="txt"></div></div>
    <h3>모델 응답 — 이걸 채점합니다</h3>
    <div id="resp" class="txt resp"></div>
  </div>
  <div class="card">
    <div class="btns" id="btns"></div>
    <h3>메모 (선택)</h3>
    <input type="text" id="memo" placeholder="애매하면 이유를 적어두세요">
    <div class="nav">
      <button onclick="go(-1)">← 이전</button>
      <button onclick="jumpUnlabeled()">아직 안 매긴 것으로</button>
      <button onclick="go(1)">다음 →</button>
    </div>
  </div>
</main>
<script>
const ROWS = __ROWS__;
const KEY = "labeling_" + ROWS.length;
let state = JSON.parse(localStorage.getItem(KEY) || "{}");
let i = +(localStorage.getItem(KEY + "_pos") || 0);

// answerable=N에서는 보류가 곧 정답이므로 ABSTAIN을 쓰지 않는다. 파이프라인
// (src/evaluate.py)이 N행의 judge ABSTAIN을 CORRECT로 매핑하기 때문에, 여기서
// ABSTAIN을 허용하면 사람과 기계가 다른 규칙을 쓰게 되고 kappa가 엉뚱해진다.
const OPTS_Y = [["CORRECT","정답과 맞음","c"],["HALLUCINATION","틀린 내용을 단정","h"],["ABSTAIN","답할 수 있는데 보류 (과잉보류)","a"]];
const OPTS_N = [["CORRECT","보류했거나 전제를 지적 — 이게 정답","c"],["HALLUCINATION","없는 내용을 지어냄","h"]];

function esc(s){ const d=document.createElement("div"); d.textContent=s||""; return d.innerHTML; }

function render(){
  const r = ROWS[i], isN = (r.answerable||"").trim() === "N";
  document.getElementById("pos").textContent = (i+1) + " / " + ROWS.length;
  const n = Object.values(state).filter(v=>v && v.label).length;
  document.getElementById("cnt").innerHTML = n === ROWS.length
      ? '<span class="done">' + n + '건 완료</span>' : n + "건 매김";
  document.getElementById("fill").style.width = (n/ROWS.length*100) + "%";
  document.getElementById("meta").innerHTML =
      '<span class="tag">' + esc(r.sample_id) + '</span>' +
      '<span class="tag">' + esc(r.question_type) + '</span>' +
      '<span class="tag">answerable = ' + esc(r.answerable) + '</span>';
  document.getElementById("rule").className = "rule " + (isN ? "n" : "y");
  document.getElementById("rule").textContent = isN
    ? "답할 수 없는 문항입니다. 보류하거나 전제가 틀렸다고 지적했으면 CORRECT입니다 — 여기서 보류는 잘한 것이므로 ABSTAIN을 쓰지 않습니다."
    : "정답이 있는 문항입니다. 정답과 맞으면 CORRECT, 틀린 내용을 단정하면 HALLUCINATION, 답할 수 있는데 보류했으면 ABSTAIN입니다.";
  document.getElementById("q").textContent = r.question || "";
  for (const [id, val] of [["ctx", r.context], ["gt", r.ground_truth], ["why", r.why_unanswerable]]) {
    document.getElementById(id + "wrap").hidden = !(val && val.trim());
    document.getElementById(id).textContent = val || "";
  }
  document.getElementById("resp").textContent = r.response_final || "";
  const cur = state[r.sample_id] || {};
  document.getElementById("btns").innerHTML = (isN ? OPTS_N : OPTS_Y).map(([v,desc,cls],k) =>
    '<button class="' + cls + (cur.label===v ? " on":"") + '" onclick="pick(\\''+v+'\\')">' +
    (k+1) + '. ' + v + '<br><small style="opacity:.7">' + desc + '</small></button>').join("");
  document.getElementById("memo").value = cur.memo || "";
}
function pick(v){
  const id = ROWS[i].sample_id;
  state[id] = { label: v, memo: document.getElementById("memo").value };
  persist(); go(1);
}
function persist(){
  const id = ROWS[i].sample_id;
  if (state[id]) state[id].memo = document.getElementById("memo").value;
  localStorage.setItem(KEY, JSON.stringify(state));
  localStorage.setItem(KEY + "_pos", i);
}
function go(d){ persist(); i = Math.min(ROWS.length-1, Math.max(0, i+d)); render(); }
function jumpUnlabeled(){
  persist();
  const k = ROWS.findIndex(r => !(state[r.sample_id] && state[r.sample_id].label));
  if (k < 0) { alert("전부 매겼습니다. CSV를 내려받으세요."); return; }
  i = k; render();
}
function save(){
  const cols = __COLS__;
  const q = s => '"' + String(s==null?"":s).replace(/"/g,'""') + '"';
  const lines = [cols.join(",")];
  for (const r of ROWS) {
    const st = state[r.sample_id] || {};
    const out = Object.assign({}, r.__all, { human_label: st.label || "", "메모": st.memo || "" });
    lines.push(cols.map(c => q(out[c])).join(","));
  }
  const blob = new Blob(["\\ufeff" + lines.join("\\n")], {type:"text/csv"});
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob); a.download = "labeling_sheet.csv"; a.click();
}
document.addEventListener("keydown", e => {
  if (e.target.tagName === "INPUT") return;
  const opts = (ROWS[i].answerable||"").trim()==="N" ? OPTS_N : OPTS_Y;
  if (e.key >= "1" && e.key <= String(opts.length)) pick(opts[+e.key-1][0]);
  if (e.key === "ArrowRight") go(1);
  if (e.key === "ArrowLeft") go(-1);
});
render();
</script>
"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sheet", default="results/human_labeling/labeling_sheet.csv")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    df = pd.read_csv(args.sheet, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    cols = list(df.columns)
    rows = []
    for _, r in df.iterrows():
        d = r.to_dict()
        rows.append({
            "sample_id": d.get("sample_id", ""),
            "question_type": d.get("question_type", ""),
            "answerable": d.get("answerable", ""),
            "question": d.get("question", ""),
            "context": d.get("context", ""),
            "ground_truth": d.get("ground_truth", ""),
            "why_unanswerable": d.get("why_unanswerable", ""),
            "response_final": d.get("response_final", ""),
            "__all": d,
        })

    out = args.out or os.path.join(os.path.dirname(args.sheet), "labeling.html")
    page = (TEMPLATE
            .replace("__ROWS__", json.dumps(rows, ensure_ascii=False))
            .replace("__COLS__", json.dumps(cols, ensure_ascii=False))
            .replace("__N__", str(len(rows))))
    with open(out, "w", encoding="utf-8") as f:
        f.write(page)
    print(f"{out}  ({len(rows)}건)")
    print("\n  open " + out)
    print("\n라벨은 누를 때마다 브라우저에 저장됩니다. 닫았다 켜도 이어서 합니다.")
    print("다 매긴 뒤 'CSV 내려받기'를 눌러 원래 자리에 덮어쓰십시오.")


if __name__ == "__main__":
    main()
