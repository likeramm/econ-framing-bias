"""프레이밍 분류 데모 (Gradio 웹 UI)

사용법:
  python3.14 scripts/demo_framing.py

브라우저에서 http://localhost:7860 접속 후 기사 제목/본문 입력.
"""

import re
import torch
import gradio as gr
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ── 설정 ──────────────────────────────────────────────
MODEL_ID = "likeramm/klue-roberta-framing-classifier"

LABEL_KR = {
    "positive": "긍정",
    "neutral": "중립",
    "negative": "부정",
}

LABEL_KR_FULL = {
    "positive": "긍정 (Positive)",
    "neutral": "중립 (Neutral)",
    "negative": "부정 (Negative)",
}

LABEL_DESC = {
    "positive": "이 기사는 **긍정적 프레이밍**입니다.\n경제 지표를 성장·회복·기대의 관점에서 해석하며, 우려가 있더라도 이를 완화하는 방향으로 서술하여 독자에게 긍정적 심리를 형성합니다.",
    "neutral": "이 기사는 **중립적 프레이밍**입니다.\n기자의 해석을 최소화하고 수치와 팩트를 있는 그대로 전달하거나, 타 국가·시기와의 비교를 통해 상대적으로만 기술합니다.",
    "negative": "이 기사는 **부정적 프레이밍**입니다.\n경제 지표를 둔화·우려·리스크의 관점에서 해석하며, 위기 표현을 부각하여 독자에게 불안 심리를 형성합니다.",
}

LABEL_EMOJI = {
    "positive": "🟢", "neutral": "⚪", "negative": "🔴",
}

FRAMING_SCORE = {
    "positive": +1, "neutral": 0, "negative": -1,
}

# ── 예시 데이터 (제목, 본문) ──────────────────────────
EXAMPLES = [
    [
        "한은, 기준금리 동결…하반기 성장률 상향 기대",
        "한국은행이 기준금리를 3.25%로 동결했다. 수출 호조와 내수 회복세가 뚜렷해지면서 하반기 성장률 전망도 상향 조정될 것으로 기대된다. 이창용 총재는 \"경기 회복 흐름이 견조하다\"고 평가했다.",
    ],
    [
        "GDP 2%대 추락…경기 둔화 우려 확산",
        "GDP 성장률이 2%대로 떨어지면서 경기 둔화 우려가 확산되고 있다. 전문가들은 하반기 전망도 불투명하다고 입을 모은다. 민간소비와 설비투자 모두 부진한 흐름을 이어가고 있어 회복 시점을 장담하기 어렵다는 분석이다.",
    ],
    [
        "글로벌 금융위기 재현 우려…경제 위기 신호 곳곳에서",
        "경제 위기 신호가 곳곳에서 감지되고 있다. 글로벌 금융위기 재현 우려까지 나오면서 시장 불안이 극에 달하고 있다. 가계부채는 사상 최고치를 경신했고, 부동산 시장은 급랭하면서 연쇄 부실 가능성까지 제기된다.",
    ],
    [
        "우려에도 불구 펀더멘털 견고…수출이 리스크 상쇄",
        "우려에도 불구하고 한국 경제의 펀더멘털은 여전히 견고하다는 평가가 나온다. 수출 회복세가 하방 리스크를 상쇄하고 있으며, 반도체 업황 개선이 경상수지 흑자를 뒷받침하고 있다.",
    ],
    [
        "한국 GDP, OECD 평균 상회…미국 대비 양호",
        "한국의 GDP 성장률은 OECD 평균을 상회하며, 미국과 비교해도 양호한 수준을 유지하고 있다. 일본, 독일 등 주요국이 마이너스 성장을 기록한 것과 대조적이다.",
    ],
    [
        "2분기 GDP 2.3% 기록…전분기比 0.1%p 상승",
        "2분기 GDP 성장률이 2.3%를 기록했다. 전분기 대비 0.1%포인트 상승한 수치다. 한국은행은 잠정치 기준으로 민간소비가 0.5%, 설비투자가 1.2% 각각 증가했다고 발표했다.",
    ],
]

# ── 모델 로드 ─────────────────────────────────────────
print(f"모델 로드 중: {MODEL_ID}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_ID, output_attentions=True
)
model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
id2label = model.config.id2label
print(f"모델 로드 완료! (device: {device})")


# ── 유틸 함수 ─────────────────────────────────────────
def split_sentences(text: str) -> list[str]:
    """한국어 문장 분리 (마침표/물음표/느낌표 기준)"""
    sents = re.split(r"(?<=[.!?다])\s+", text)
    return [s.strip() for s in sents if s.strip() and len(s.strip()) > 5]


def classify_text(text: str) -> tuple[torch.Tensor, torch.Tensor]:
    """텍스트를 모델에 입력하고 (확률, attention) 반환"""
    inputs = tokenizer(
        text, max_length=512, truncation=True, padding=True, return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    probs = torch.softmax(outputs.logits, dim=-1).squeeze()

    # Attention: 마지막 레이어, 모든 헤드 평균 → [seq_len, seq_len]
    # CLS 토큰(0번)이 각 토큰에 주는 attention을 가져옴
    last_attn = outputs.attentions[-1].squeeze()        # [heads, seq, seq]
    cls_attn = last_attn.mean(dim=0)[0]                  # [seq_len] — CLS → 각 토큰

    # 토큰 목록
    input_ids = inputs["input_ids"].squeeze()
    tokens = tokenizer.convert_ids_to_tokens(input_ids)

    return probs, cls_attn, tokens


def get_attention_highlights(text: str, top_k: int = 10) -> list[tuple[str, float]]:
    """텍스트의 토큰별 attention 점수를 반환 (서브워드 병합)"""
    inputs = tokenizer(
        text, max_length=512, truncation=True, padding=True, return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    last_attn = outputs.attentions[-1].squeeze()
    cls_attn = last_attn.mean(dim=0)[0]  # CLS → 각 토큰

    input_ids = inputs["input_ids"].squeeze()
    tokens = tokenizer.convert_ids_to_tokens(input_ids)
    scores = cls_attn.cpu().tolist()

    # 서브워드(##) 병합 + 특수 토큰 제거
    merged = []
    for tok, sc in zip(tokens, scores):
        if tok in ("[CLS]", "[SEP]", "[PAD]", "<s>", "</s>", "<pad>"):
            continue
        if tok.startswith("##") and merged:
            merged[-1] = (merged[-1][0] + tok[2:], merged[-1][1] + sc)
        else:
            merged.append((tok, sc))

    # 정규화 (0~1)
    if merged:
        max_sc = max(s for _, s in merged)
        min_sc = min(s for _, s in merged)
        rng = max_sc - min_sc if max_sc > min_sc else 1.0
        merged = [(w, (s - min_sc) / rng) for w, s in merged]

    return merged


def sentence_analysis(sentences: list[str]) -> list[dict]:
    """각 문장을 개별적으로 모델에 넣어서 프레이밍 분류"""
    results = []
    for sent in sentences:
        if len(sent.strip()) < 5:
            continue
        inputs = tokenizer(
            sent, max_length=512, truncation=True, padding=True, return_tensors="pt"
        ).to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1).squeeze()
        pred_idx = probs.argmax().item()
        pred_label = id2label[pred_idx]
        results.append({
            "sentence": sent,
            "label": pred_label,
            "confidence": probs[pred_idx].item(),
            "probs": {id2label[i]: float(probs[i]) for i in range(len(probs))},
        })
    return results


# ── 메인 예측 함수 ────────────────────────────────────
def predict(title: str, body: str) -> tuple[dict, str]:
    """제목+본문 → 전체 분류 + 문장별 분석 + Attention 시각화"""
    title = (title or "").strip()
    body = (body or "").strip()

    if not title and not body:
        return {}, "제목 또는 본문을 입력해주세요."

    # 제목 + 본문 결합
    if title and body:
        text = f"{title} [SEP] {body}"
        full_text = f"{title} {body}"
    else:
        text = title or body
        full_text = text

    # ── 1. 전체 분류 ──
    inputs = tokenizer(
        text, max_length=512, truncation=True, padding=True, return_tensors="pt"
    ).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    probs = torch.softmax(outputs.logits, dim=-1).squeeze()

    label_probs = {}
    for i, p in enumerate(probs):
        eng = id2label[i]
        label_probs[LABEL_KR_FULL.get(eng, eng)] = float(p)

    pred_idx = probs.argmax().item()
    pred_label = id2label[pred_idx]
    confidence = probs[pred_idx].item()
    emoji = LABEL_EMOJI.get(pred_label, "")
    score = FRAMING_SCORE.get(pred_label, 0)

    # 확률 순위
    sorted_probs = sorted(
        [(id2label[i], float(probs[i])) for i in range(len(probs))],
        key=lambda x: x[1], reverse=True,
    )
    ranking_rows = ""
    for rank, (lbl, prob) in enumerate(sorted_probs, 1):
        e = LABEL_EMOJI.get(lbl, "")
        bar = "█" * int(prob * 20) + "░" * (20 - int(prob * 20))
        ranking_rows += f"| {rank} | {e} {LABEL_KR_FULL[lbl]} | `{bar}` | {prob:.1%} |\n"

    # ── 2. 문장별 분석 ──
    sentences = split_sentences(full_text)
    sent_results = sentence_analysis(sentences)

    sent_md = ""
    if sent_results:
        sent_md += "### 🔬 문장별 프레이밍 분석\n\n"
        sent_md += "모델이 **각 문장을 개별적으로** 분석한 결과입니다.\n\n"

        # 기여도 요약 (전체 분류와 같은 유형인 문장 수)
        same_count = sum(1 for r in sent_results if r["label"] == pred_label)
        total_count = len(sent_results)
        sent_md += f"> 전체 {total_count}개 문장 중 **{same_count}개**가 "
        sent_md += f"{emoji} {LABEL_KR[pred_label]} 프레이밍으로 분류되어 "
        sent_md += f"최종 결과에 기여했습니다.\n\n"

        sent_md += "| # | 프레이밍 | 신뢰도 | 문장 |\n"
        sent_md += "|---|---------|--------|------|\n"
        for i, r in enumerate(sent_results, 1):
            e = LABEL_EMOJI.get(r["label"], "")
            kr = LABEL_KR.get(r["label"], r["label"])
            conf = r["confidence"]
            # 문장이 너무 길면 자르기
            sent_text = r["sentence"]
            if len(sent_text) > 60:
                sent_text = sent_text[:57] + "..."
            marker = " **◀**" if r["label"] == pred_label else ""
            sent_md += f"| {i} | {e} {kr} | {conf:.0%} | {sent_text}{marker} |\n"

        sent_md += f"\n> **◀** 표시 = 전체 분류({LABEL_KR[pred_label]})와 동일한 유형\n"

    # ── 3. Attention 시각화 ──
    attn_tokens = get_attention_highlights(text)

    attn_md = ""
    if attn_tokens:
        attn_md += "\n### 🧠 Attention 시각화\n\n"
        attn_md += "모델이 분류 시 **실제로 집중한 단어**입니다. "
        attn_md += "진할수록 모델이 더 많이 참고한 토큰입니다.\n\n"

        # 상위 집중 토큰
        sorted_tokens = sorted(attn_tokens, key=lambda x: x[1], reverse=True)
        top_tokens = [(w, s) for w, s in sorted_tokens if len(w) > 1][:12]

        if top_tokens:
            attn_md += "**모델이 가장 집중한 단어 (Top 12):**\n\n"
            attn_md += "| 순위 | 토큰 | 집중도 |\n"
            attn_md += "|------|------|--------|\n"
            for rank, (word, sc) in enumerate(top_tokens, 1):
                bar = "🟥" * int(sc * 5) + "⬜" * (5 - int(sc * 5))
                attn_md += f"| {rank} | **{word}** | {bar} {sc:.0%} |\n"

        # 전체 텍스트 히트맵 (마크다운으로)
        attn_md += "\n**전체 텍스트 Attention 히트맵:**\n\n> "
        for word, sc in attn_tokens:
            if sc >= 0.8:
                attn_md += f"**`{word}`**"
            elif sc >= 0.5:
                attn_md += f"**{word}**"
            elif sc >= 0.3:
                attn_md += f"*{word}*"
            else:
                attn_md += word
        attn_md += "\n\n"
        attn_md += "> `강조` = 매우 높은 집중 | **굵게** = 높은 집중 | *기울임* = 보통 | 일반 = 낮음\n"

    # ── 최종 마크다운 ──
    result_md = f"""## {emoji} 분류 결과: {LABEL_KR_FULL[pred_label]}

| 항목 | 값 |
|------|-----|
| **프레이밍 유형** | {emoji} {LABEL_KR_FULL[pred_label]} |
| **신뢰도** | **{confidence:.1%}** |
| **프레이밍 점수** | **{score:+d}** (범위: -2 ~ +2) |

---

### 유형별 확률 순위

| 순위 | 유형 | 확률 바 | 확률 |
|------|------|---------|------|
{ranking_rows}
---

### 해석

{LABEL_DESC.get(pred_label, "")}

---

{sent_md}
---

{attn_md}
"""

    if title:
        result_md += f"\n> **분석 대상 제목**: {title}"

    return label_probs, result_md


# ── Gradio UI ─────────────────────────────────────────
with gr.Blocks(title="경제뉴스 프레이밍 편향 분석") as demo:
    gr.Markdown(
        """
# 📰 경제뉴스 프레이밍 편향 분석 시스템
### NLP 기반 6가지 프레이밍 유형 자동 분류 | KLUE-RoBERTa-large Fine-tuned

동일한 경제 이벤트도 언론사마다 다른 **프레이밍(틀짓기)**으로 보도합니다.
기사의 제목과 본문을 입력하면 AI가 프레이밍 유형을 분류하고, **문장별 분석**과 **Attention 시각화**로 판단 근거를 설명합니다.

| 유형 | 설명 | 편향 점수 | 이론적 근거 |
|------|------|----------|-------------|
| 🟢 낙관적 | 긍정적 전망 강조 | +2 | 이익 프레임 (Kahneman & Tversky, 1979) |
| 🟡 방어적 | 우려 인정 + 완화 | +1 | 양면적 프레이밍 (De Vreese, 2005) |
| ⚪ 중립적 | 사실 중심 보도 | 0 | 주제적 프레임 (Iyengar, 1991) |
| 🔵 비교적 | 타국/타시기 비교 | 0 | 귀인 프레임 (Iyengar, 1991) |
| 🔴 비관적 | 부정적 전망 강조 | -1 | 손실 프레임 (Kahneman & Tversky, 1979) |
| 🚨 경고적 | 위기 상황 부각 | -2 | 갈등 프레임 (Semetko & Valkenburg, 2000) |
"""
    )

    with gr.Row():
        with gr.Column(scale=1):
            title_input = gr.Textbox(
                label="📌 기사 제목",
                placeholder='예: "한은, 기준금리 동결…하반기 성장률 상향 기대"',
                lines=1,
            )
            body_input = gr.Textbox(
                label="📝 기사 본문",
                placeholder="기사 본문을 붙여넣으세요... (선택사항 — 제목만으로도 분석 가능)",
                lines=8,
            )
            submit_btn = gr.Button("🔍 프레이밍 분석하기", variant="primary", size="lg")

        with gr.Column(scale=1):
            label_output = gr.Label(
                label="프레이밍 유형별 확률",
                num_top_classes=6,
            )
            result_output = gr.Markdown(label="분석 결과")

    gr.Markdown("### 💡 예시 기사 (클릭하면 자동 입력)")
    gr.Examples(
        examples=EXAMPLES,
        inputs=[title_input, body_input],
        label="",
    )

    submit_btn.click(
        fn=predict,
        inputs=[title_input, body_input],
        outputs=[label_output, result_output],
    )
    title_input.submit(
        fn=predict,
        inputs=[title_input, body_input],
        outputs=[label_output, result_output],
    )

if __name__ == "__main__":
    demo.launch(
        share=False,
        server_name="0.0.0.0",
        server_port=7860,
        theme=gr.themes.Soft(primary_hue="blue"),
    )
