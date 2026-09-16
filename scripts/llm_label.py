"""LLM 자동 라벨링 (3-class: 긍정/중립/부정)

PDF 2차발표 계획 [1단계] LLM 자동 라벨링에 대응.
docs/라벨링_가이드라인_v1.md 의 프롬프트 템플릿을 그대로 사용한다.

특징:
  - 중단/재개 지원 (체크포인트 저장 → 재실행 시 남은 것만 처리)
  - 동시 요청 (기본 8스레드)
  - 실패 항목 자동 재시도 + 실패 로그 분리
  - --dry-run 으로 비용/토큰 추정만

사전 준비:
  .env 에 다음 중 하나 추가
    OPENAI_API_KEY=sk-...
    ANTHROPIC_API_KEY=sk-ant-...

사용법:
  python3 scripts/llm_label.py --dry-run          # 비용 추정만
  python3 scripts/llm_label.py --limit 50         # 50건 시험
  python3 scripts/llm_label.py                    # 전체
  python3 scripts/llm_label.py --provider anthropic
"""

import argparse
import json
import os
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
from tqdm.auto import tqdm

# ══════════════════════════════════════════════════════════
# 설정
# ══════════════════════════════════════════════════════════
CONFIG = {
    "input_path": "data/processed/dataset.csv",
    "output_path": "data/labeled/llm_labeled.csv",
    "failed_path": "data/labeled/llm_failed.csv",
    "content_chars": 600,      # 본문 앞 N자만 전송 (비용 절감)
    "max_retries": 6,
}

MODELS = {
    "openai":    "gpt-4o",        # 상위 모델 기본값 (--model 로 변경 가능)
    "anthropic": "claude-sonnet-5",
}

# 1M 토큰당 USD (입력, 출력)
PRICING = {
    "gpt-4o":          (2.50, 10.00),
    "gpt-4o-mini":     (0.15,  0.60),
    "gpt-4.1":         (2.00,  8.00),
    "gpt-4.1-mini":    (0.40,  1.60),
    "claude-sonnet-5": (3.00, 15.00),
    "claude-opus-5":   (15.00, 75.00),
}

BAD_MEDIA = ["매일경제TV", "서울경제TV", "미주중앙일보"]
VALID = {"긍정", "중립", "부정"}
KO2EN = {"긍정": "positive", "중립": "neutral", "부정": "negative"}

SYSTEM_PROMPT = """당신은 한국어 경제 뉴스의 프레이밍을 분류하는 전문 어노테이터입니다.

[분류 기준]
- 긍정: 성장·회복·개선·안도의 틀로 서술. 우려를 언급해도 완화·방어로 마무리.
- 중립: 기자의 해석이 최소화된 사실·수치 전달. 또는 양측 병렬 제시로 평가 유보.
- 부정: 둔화·하락·우려·위기의 틀로 서술.

[판정 규칙 — 번호가 낮을수록 우선. 충돌하면 반드시 작은 번호가 이깁니다]

R0. 사건이 아니라 서술을 본다 (최상위)
    사건 자체의 호재/악재는 판단 대상이 아닙니다.
    나쁜 사건이라도 건조한 사실 전달이면 중립입니다.
    예: 범죄·사고·불리한 여론조사를 수치와 경과만으로 보도 -> 중립

R1. 제목과 본문이 충돌하면 제목의 프레임을 따릅니다.

R2. 병렬 제시면 중립
    긍정 요소와 부정 요소가 비슷한 비중으로 병기되고
    기자가 어느 쪽으로도 결론짓지 않으면 중립입니다.
    예: "A 역대 최대...B 부진 전망도", "A 주춤...B는 고공행진"
    시황 기사에서 지수 등락을 수치로 전달하는 것도 중립입니다.

R3. 결론 우선
    앞뒤 방향이 다르고 기자가 한쪽으로 분명히 결론을 냈다면
    최종적으로 향하는 방향을 따릅니다.
    결론을 내지 않았다면 R2(중립)가 우선합니다.

R4. 그래도 판단이 서지 않으면 중립.

반드시 JSON 객체 하나만 출력하십시오. 다른 텍스트를 덧붙이지 마십시오.
형식: {"label": "긍정|중립|부정", "confidence": 0.0~1.0, "reason": "적용한 규칙 번호와 근거 한 문장"}"""

USER_TMPL = "제목: {title}\n본문: {content}"

_print_lock = threading.Lock()
SHUFFLE = False


# ══════════════════════════════════════════════════════════
# 데이터 준비
# ══════════════════════════════════════════════════════════
def load_targets(limit=None):
    df = pd.read_csv(CONFIG["input_path"], low_memory=False)
    before = len(df)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df[df["date"].notna()]
    df = df[~df["media_name"].isin(BAD_MEDIA)]
    df = df[df["content_clean"].astype(str).str.len() >= 200]
    df = df.drop_duplicates(subset=["content_clean"])
    print(f"  정제: {before:,}건 → {len(df):,}건")

    # 체크포인트: 이미 라벨링된 것 제외
    out = Path(CONFIG["output_path"])
    if out.exists():
        done = pd.read_csv(out)
        done_ids = set(done["article_id"])
        df = df[~df["article_id"].isin(done_ids)]
        print(f"  체크포인트: {len(done_ids):,}건 완료 → 남은 {len(df):,}건")

    if limit:
        # --shuffle 시 무작위 표본 (테스트용). 전체 실행에는 영향 없음.
        df = df.sample(n=min(limit, len(df)), random_state=42) if SHUFFLE else df.head(limit)
    return df.reset_index(drop=True)


# ══════════════════════════════════════════════════════════
# LLM 호출
# ══════════════════════════════════════════════════════════
def make_client(provider):
    key_env = "OPENAI_API_KEY" if provider == "openai" else "ANTHROPIC_API_KEY"
    key = os.environ.get(key_env)
    if not key:
        sys.exit(
            f"\n❌ {key_env} 가 설정되지 않았습니다.\n"
            f"   .env 파일에 다음 줄을 추가하세요:\n   {key_env}=여기에_키_입력\n"
        )
    if provider == "openai":
        from openai import OpenAI
        return OpenAI(api_key=key)
    import anthropic
    return anthropic.Anthropic(api_key=key)


def call_llm(client, provider, model, title, content):
    user = USER_TMPL.format(title=title, content=content)
    if provider == "openai":
        r = client.chat.completions.create(
            model=model, temperature=0, max_tokens=200,
            response_format={"type": "json_object"},
            messages=[{"role": "system", "content": SYSTEM_PROMPT},
                      {"role": "user", "content": user}],
        )
        return r.choices[0].message.content
    r = client.messages.create(
        model=model, max_tokens=200, temperature=0,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user}],
    )
    return r.content[0].text


def parse(raw):
    """LLM 응답에서 label/confidence/reason 추출"""
    m = re.search(r"\{.*\}", raw, re.S)
    if not m:
        raise ValueError(f"JSON 없음: {raw[:80]}")
    d = json.loads(m.group())
    label = str(d.get("label", "")).strip()
    if label not in VALID:
        raise ValueError(f"허용되지 않은 label: {label!r}")
    conf = d.get("confidence", "")
    try:
        conf = round(float(conf), 3)
    except (TypeError, ValueError):
        conf = ""
    return label, conf, str(d.get("reason", ""))[:200]


def label_one(client, provider, model, row):
    title = str(row["title_clean"])[:200]
    content = str(row["content_clean"])[:CONFIG["content_chars"]]
    last = None
    for attempt in range(CONFIG["max_retries"]):
        try:
            raw = call_llm(client, provider, model, title, content)
            label, conf, reason = parse(raw)
            return {
                "article_id": row["article_id"],
                "framing_label": KO2EN[label],
                "framing_label_kr": label,
                "confidence": conf,
                "reason": reason,
                "media_name": row["media_name"],
                "media_group": row["media_group"],
                "event_type": row["event_type"],
                "date": row["date"].strftime("%Y-%m-%d"),
                "title": title,
            }
        except Exception as e:
            last = e
            msg = str(e)
            if "429" in msg or "rate_limit" in msg.lower():
                time.sleep(20 + 10 * attempt)   # TPM 초과: 길게 대기
            else:
                time.sleep(2 ** attempt)        # 그 외: 지수 백오프
    return {"article_id": row["article_id"], "_error": str(last)[:200]}


# ══════════════════════════════════════════════════════════
# 비용 추정
# ══════════════════════════════════════════════════════════
def estimate(df, model):
    # 한국어는 대략 1.5자/토큰
    sys_tok = len(SYSTEM_PROMPT) / 1.5
    body = (df["content_clean"].astype(str).str.len().clip(upper=CONFIG["content_chars"])
            + df["title_clean"].astype(str).str.len().clip(upper=200))
    in_tok = (sys_tok + body / 1.5).sum()
    out_tok = len(df) * 60
    pin, pout = PRICING.get(model, (0, 0))
    usd = in_tok / 1e6 * pin + out_tok / 1e6 * pout
    print(f"\n  모델: {model}")
    print(f"  대상: {len(df):,}건")
    print(f"  입력 토큰 ≈ {in_tok/1e6:.2f}M / 출력 토큰 ≈ {out_tok/1e6:.2f}M")
    print(f"  예상 비용 ≈ ${usd:.2f}  (약 {usd*1400:,.0f}원)")


# ══════════════════════════════════════════════════════════
# 메인
# ══════════════════════════════════════════════════════════
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", choices=["openai", "anthropic"], default="openai")
    ap.add_argument("--model", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--workers", type=int, default=4)  # gpt-4o TPM 300k 고려
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--shuffle", action="store_true", help="--limit 표본을 무작위로 (테스트용)")
    a = ap.parse_args()

    model = a.model or MODELS[a.provider]
    global SHUFFLE
    SHUFFLE = a.shuffle

    # .env 로드
    envf = Path(".env")
    if envf.exists():
        for line in envf.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())

    print("=" * 58)
    print(f"LLM 자동 라벨링 (3-class)  provider={a.provider}")
    print("=" * 58)

    df = load_targets(a.limit)
    if df.empty:
        print("\n✅ 모든 항목이 이미 라벨링되어 있습니다.")
        return

    estimate(df, model)
    if a.dry_run:
        print("\n(--dry-run 이므로 실제 호출 없이 종료)")
        return

    client = make_client(a.provider)
    out_path = Path(CONFIG["output_path"])
    out_path.parent.mkdir(parents=True, exist_ok=True)

    results, failures = [], []
    rows = [r for _, r in df.iterrows()]

    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        futs = {ex.submit(label_one, client, a.provider, model, r): r for r in rows}
        with tqdm(total=len(futs), desc="라벨링") as bar:
            for fut in as_completed(futs):
                res = fut.result()
                (failures if "_error" in res else results).append(res)
                bar.update(1)

                # 200건마다 중간 저장 (중단 대비)
                if len(results) % 200 == 0 and results:
                    pd.DataFrame(results).to_csv(
                        out_path, mode="a", header=not out_path.exists(),
                        index=False, encoding="utf-8-sig")
                    results.clear()

    if results:
        pd.DataFrame(results).to_csv(
            out_path, mode="a", header=not out_path.exists(),
            index=False, encoding="utf-8-sig")
    if failures:
        pd.DataFrame(failures).to_csv(CONFIG["failed_path"], index=False, encoding="utf-8-sig")

    # 결과 요약
    final = pd.read_csv(out_path)
    print(f"\n{'='*58}")
    print(f"완료: {len(final):,}건 저장  |  실패: {len(failures):,}건")
    print(f"  {out_path}")
    if failures:
        print(f"  실패 로그: {CONFIG['failed_path']} (재실행하면 자동 재시도)")
    print(f"\n[라벨 분포]")
    vc = final["framing_label_kr"].value_counts()
    for k in ["긍정", "중립", "부정"]:
        n = vc.get(k, 0)
        print(f"  {k}  {n:6,}건 ({n/len(final)*100:5.1f}%)  " + "#" * int(n / len(final) * 40))
    if "confidence" in final.columns:
        c = pd.to_numeric(final["confidence"], errors="coerce")
        print(f"\n  평균 confidence: {c.mean():.3f}  |  0.7 미만: {(c < 0.7).sum():,}건 (검수 권장)")


if __name__ == "__main__":
    main()
