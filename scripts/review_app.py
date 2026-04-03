"""수동 라벨링 웹 UI

전체 크롤링 데이터에서 랜덤 샘플링하여 수동 라벨링.
제목 + 본문 미리보기를 보고 6가지 프레이밍 유형 버튼 클릭.
키보드 단축키 지원 (1~6), 진행 상황 자동 저장.

사용법: python scripts/review_app.py
→ http://localhost:8080
"""

import json
import glob
import random
from pathlib import Path
from flask import Flask, render_template_string, request, jsonify
import pandas as pd

app = Flask(__name__)

# ── 설정 ──
RAW_DIR = Path("data/raw")
LABEL_PATH = Path("data/labeled/manual_labels.json")
TARGET_COUNT = 500  # 목표 라벨링 수

# ── 데이터 로드 ──
def load_articles():
    files = sorted(glob.glob(str(RAW_DIR / "*.csv")))
    dfs = []
    for f in files:
        try:
            df = pd.read_csv(f)
            dfs.append(df)
        except:
            pass
    combined = pd.concat(dfs, ignore_index=True)
    combined = combined.drop_duplicates(subset=["url"], keep="first").reset_index(drop=True)
    # 셔플 (고정 시드로 재현 가능)
    combined = combined.sample(frac=1, random_state=42).reset_index(drop=True)
    return combined

df = load_articles()
print(f"총 {len(df):,}건 로드")

# 기존 라벨 로드
LABEL_PATH.parent.mkdir(parents=True, exist_ok=True)
if LABEL_PATH.exists():
    with open(LABEL_PATH, "r") as f:
        labels = json.load(f)
else:
    labels = {}

HTML = """
<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>프레이밍 수동 라벨링</title>
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body { font-family: -apple-system, 'Malgun Gothic', sans-serif; background: #0f172a; color: #e2e8f0; min-height: 100vh; }

  .header { background: #1e293b; padding: 16px 24px; border-bottom: 1px solid #334155; display: flex; justify-content: space-between; align-items: center; position: sticky; top: 0; z-index: 100; }
  .header h1 { font-size: 18px; color: #f8fafc; }
  .progress-info { display: flex; gap: 16px; align-items: center; font-size: 14px; color: #94a3b8; }
  .progress-bar { width: 300px; height: 10px; background: #334155; border-radius: 5px; overflow: hidden; }
  .progress-fill { height: 100%; background: linear-gradient(90deg, #3b82f6, #8b5cf6); transition: width 0.3s; border-radius: 5px; }

  .container { max-width: 800px; margin: 0 auto; padding: 32px 24px; }

  .stats { display: grid; grid-template-columns: repeat(6, 1fr); gap: 8px; margin-bottom: 24px; }
  .stat-card { background: #1e293b; border: 1px solid #334155; border-radius: 8px; padding: 10px; text-align: center; }
  .stat-card .label { font-size: 11px; color: #94a3b8; margin-bottom: 2px; }
  .stat-card .count { font-size: 22px; font-weight: 700; }

  .card { background: #1e293b; border: 2px solid #334155; border-radius: 16px; padding: 32px; margin-bottom: 24px; }
  .card-header { display: flex; justify-content: space-between; margin-bottom: 16px; font-size: 13px; color: #64748b; }
  .title { font-size: 20px; font-weight: 700; color: #f1f5f9; line-height: 1.6; margin-bottom: 16px; }
  .content { font-size: 14px; color: #94a3b8; line-height: 1.8; max-height: 200px; overflow-y: auto; padding: 16px; background: #0f172a; border-radius: 8px; margin-bottom: 24px; white-space: pre-wrap; }

  .actions { display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px; }
  .btn { padding: 16px; border: 2px solid transparent; border-radius: 12px; font-size: 16px; font-weight: 700; cursor: pointer; transition: all 0.15s; text-align: center; }
  .btn:hover { transform: translateY(-2px); box-shadow: 0 4px 12px rgba(0,0,0,0.3); }
  .btn:active { transform: translateY(0); }
  .btn .shortcut { display: block; font-size: 11px; font-weight: 400; margin-top: 4px; opacity: 0.6; }

  .btn-optimistic { background: #064e3b; color: #34d399; border-color: #047857; }
  .btn-optimistic:hover { background: #047857; }
  .btn-pessimistic { background: #450a0a; color: #f87171; border-color: #991b1b; }
  .btn-pessimistic:hover { background: #991b1b; }
  .btn-alarmist { background: #431407; color: #fb923c; border-color: #9a3412; }
  .btn-alarmist:hover { background: #9a3412; }
  .btn-defensive { background: #1e1b4b; color: #a78bfa; border-color: #4338ca; }
  .btn-defensive:hover { background: #4338ca; }
  .btn-comparative { background: #172554; color: #60a5fa; border-color: #1d4ed8; }
  .btn-comparative:hover { background: #1d4ed8; }
  .btn-neutral { background: #1c1917; color: #a8a29e; border-color: #57534e; }
  .btn-neutral:hover { background: #57534e; }

  .nav { display: flex; gap: 12px; justify-content: center; margin-top: 16px; }
  .nav-btn { padding: 8px 20px; border-radius: 8px; border: 1px solid #475569; background: transparent; color: #94a3b8; font-size: 14px; cursor: pointer; }
  .nav-btn:hover { border-color: #60a5fa; color: #60a5fa; }

  .skip-btn { padding: 10px 24px; border-radius: 8px; border: 1px solid #475569; background: transparent; color: #64748b; font-size: 14px; cursor: pointer; margin-top: 12px; width: 100%; }
  .skip-btn:hover { border-color: #94a3b8; color: #94a3b8; }

  .toast { position: fixed; bottom: 24px; right: 24px; padding: 12px 24px; border-radius: 8px; font-size: 14px; font-weight: 600; opacity: 0; transition: opacity 0.3s; pointer-events: none; }
  .toast.show { opacity: 1; }
  .toast-success { background: #065f46; color: #34d399; }

  .done-msg { text-align: center; padding: 60px; }
  .done-msg h2 { font-size: 28px; margin-bottom: 16px; color: #34d399; }
  .done-msg p { font-size: 16px; color: #94a3b8; }
</style>
</head>
<body>

<div class="header">
  <h1>프레이밍 수동 라벨링</h1>
  <div class="progress-info">
    <span id="progressText">0 / {{ target }}건</span>
    <div class="progress-bar"><div class="progress-fill" id="progressFill" style="width:0%"></div></div>
  </div>
</div>

<div class="container">
  <div class="stats" id="statsGrid"></div>
  <div id="cardArea"></div>
  <div class="nav">
    <button class="nav-btn" onclick="goTo(currentIdx-1)">← 이전 (Q)</button>
    <button class="nav-btn" onclick="goTo(currentIdx+1)">다음 (E) →</button>
  </div>
</div>

<div class="toast toast-success" id="toast"></div>

<script>
const LABELS = ["optimistic","pessimistic","alarmist","defensive","comparative","neutral"];
const LABEL_KR = {optimistic:"낙관적",pessimistic:"비관적",alarmist:"경보적",defensive:"방어적",comparative:"비교적",neutral:"중립적"};
const COLORS = {optimistic:"#34d399",pessimistic:"#f87171",alarmist:"#fb923c",defensive:"#a78bfa",comparative:"#60a5fa",neutral:"#a8a29e"};
const totalArticles = {{ total }};
const target = {{ target }};
let labels = {{ labels_json|safe }};
let currentIdx = findNextUnlabeled(0);

function findNextUnlabeled(from) {
  // 라벨링 안 된 다음 기사 찾기
  for (let i = from; i < totalArticles; i++) {
    if (!labels[String(i)]) return i;
  }
  return from;  // 다 했으면 현재 위치
}

function loadArticle(idx) {
  fetch('/api/article/' + idx)
    .then(r => r.json())
    .then(a => {
      const area = document.getElementById('cardArea');
      const labeled = labels[String(idx)];
      const contentPreview = (a.content || '').substring(0, 500) + ((a.content||'').length > 500 ? '...' : '');

      area.innerHTML = `
        <div class="card" id="mainCard">
          <div class="card-header">
            <span>#${idx+1} / ${totalArticles.toLocaleString()}건</span>
            <span>${a.media_name || ''} | ${a.published_at || ''} | 키워드: ${a.keyword || ''}</span>
          </div>
          <div class="title">${a.title}</div>
          <div class="content">${contentPreview || '(본문 없음)'}</div>
          <div class="actions">
            ${LABELS.map((l,i) => `
              <button class="btn btn-${l} ${labeled===l?'selected':''}" onclick="setLabel(${idx},'${l}')"
                style="${labeled===l?'box-shadow: 0 0 0 4px '+COLORS[l]+'66;transform:scale(1.02)':''}">
                ${LABEL_KR[l]}
                <span class="shortcut">단축키: ${i+1}</span>
              </button>
            `).join('')}
          </div>
          <button class="skip-btn" onclick="skipAndNext()">건너뛰기 (S)</button>
        </div>
      `;
    });
}

function setLabel(idx, label) {
  labels[String(idx)] = label;
  fetch('/api/label', {
    method: 'POST',
    headers: {'Content-Type':'application/json'},
    body: JSON.stringify({ index: idx, label: label })
  });
  showToast(LABEL_KR[label] + ' ✓');
  updateStats();
  // 0.4초 후 다음으로
  setTimeout(() => {
    currentIdx = findNextUnlabeled(idx + 1);
    loadArticle(currentIdx);
  }, 400);
}

function skipAndNext() {
  currentIdx = findNextUnlabeled(currentIdx + 1);
  loadArticle(currentIdx);
}

function goTo(idx) {
  if (idx < 0) idx = 0;
  if (idx >= totalArticles) idx = totalArticles - 1;
  currentIdx = idx;
  loadArticle(currentIdx);
}

function updateStats() {
  const counts = {};
  LABELS.forEach(l => counts[l] = 0);
  Object.values(labels).forEach(l => { if(counts[l] !== undefined) counts[l]++; });
  const total = Object.keys(labels).length;

  document.getElementById('progressText').textContent = total + ' / ' + target + '건';
  document.getElementById('progressFill').style.width = (total/target*100) + '%';

  document.getElementById('statsGrid').innerHTML = LABELS.map(l =>
    `<div class="stat-card">
      <div class="label">${LABEL_KR[l]}</div>
      <div class="count" style="color:${COLORS[l]}">${counts[l]}</div>
    </div>`
  ).join('');

  if (total >= target) {
    showToast('🎉 목표 ' + target + '건 달성!');
  }
}

function showToast(msg) {
  const t = document.getElementById('toast');
  t.textContent = msg;
  t.classList.add('show');
  setTimeout(() => t.classList.remove('show'), 1500);
}

// 키보드 단축키
document.addEventListener('keydown', e => {
  if (e.key >= '1' && e.key <= '6') {
    setLabel(currentIdx, LABELS[parseInt(e.key)-1]);
  } else if (e.key === 'q' || e.key === 'Q') {
    goTo(currentIdx - 1);
  } else if (e.key === 'e' || e.key === 'E') {
    goTo(currentIdx + 1);
  } else if (e.key === 's' || e.key === 'S') {
    skipAndNext();
  }
});

// 초기 로드
updateStats();
loadArticle(currentIdx);
</script>
</body>
</html>
"""


@app.route("/")
def index():
    return render_template_string(
        HTML,
        total=len(df),
        target=TARGET_COUNT,
        labels_json=json.dumps(labels, ensure_ascii=False),
    )


@app.route("/api/article/<int:idx>")
def get_article(idx):
    if idx < 0 or idx >= len(df):
        return jsonify({"error": "out of range"}), 404
    row = df.iloc[idx]
    return jsonify({
        "title": str(row.get("title", "")),
        "content": str(row.get("content", ""))[:1000],
        "media_name": str(row.get("media_name", "")),
        "published_at": str(row.get("published_at", "")),
        "keyword": str(row.get("keyword", "")),
        "url": str(row.get("url", "")),
    })


@app.route("/api/label", methods=["POST"])
def save_label():
    data = request.json
    idx = str(data["index"])
    label = data["label"]
    labels[idx] = label

    with open(LABEL_PATH, "w") as f:
        json.dump(labels, f, ensure_ascii=False, indent=2)

    return jsonify({"ok": True, "total": len(labels)})


@app.route("/api/export", methods=["POST"])
def export():
    """라벨링 결과를 학습용 CSV로 내보내기"""
    rows = []
    for idx_str, label in labels.items():
        idx = int(idx_str)
        if idx < len(df):
            row = df.iloc[idx]
            rows.append({
                "title": row.get("title", ""),
                "content": row.get("content", ""),
                "published_at": row.get("published_at", ""),
                "media_name": row.get("media_name", ""),
                "url": row.get("url", ""),
                "keyword": row.get("keyword", ""),
                "framing_label": label,
            })

    out_df = pd.DataFrame(rows)
    out_path = "data/labeled/manual_labeled_dataset.csv"
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False, encoding="utf-8-sig")

    return jsonify({
        "count": len(rows),
        "path": out_path,
        "distribution": out_df["framing_label"].value_counts().to_dict(),
    })


if __name__ == "__main__":
    labeled_count = len(labels)
    print(f"\n{'='*50}")
    print(f"  프레이밍 수동 라벨링 UI")
    print(f"  전체 기사: {len(df):,}건")
    print(f"  기존 라벨링: {labeled_count}건")
    print(f"  목표: {TARGET_COUNT}건")
    print(f"{'='*50}")
    print(f"\n  → http://localhost:8080")
    print(f"\n  단축키: 1~6 (라벨 선택) | Q/E (이전/다음) | S (건너뛰기)")
    print()
    app.run(debug=False, port=8080)
