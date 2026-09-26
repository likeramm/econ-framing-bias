"""LLM 자동 라벨링 (3-class: 긍정/중립/부정)

PDF 2차발표 계획 [1단계] LLM 자동 라벨링에 대응.
docs/라벨링_가이드라인_v1.md 의 프롬프트 템플릿을 그대로 사용한다.

특징:
  - 중단/재개 지원 (체크포인트 저장 → 재실행 시 남은 것만 처리)
  - 동시 요청 (기본 8스레드)
  - 실패 항목 자동 재시도 + 실패 로그 분리
  - --dry-run 으로 비용/토큰 추정만
  - --batch 로 OpenAI Batch API 사용 (비용 50% 절감, 최대 24시간 소요)

사전 준비:
  .env 에 다음 중 하나 추가
    OPENAI_API_KEY=sk-...
    ANTHROPIC_API_KEY=sk-ant-...

사용법:
  python3 scripts/llm_label.py --dry-run          # 비용 추정만
  python3 scripts/llm_label.py --limit 50         # 50건 시험
  python3 scripts/llm_label.py                    # 전체
  python3 scripts/llm_label.py --provider anthropic

  # Batch API (OpenAI 전용)
  python3 scripts/llm_label.py --batch auto       # 끝날 때까지 제출/수집 자동 반복 (권장)
  python3 scripts/llm_label.py --batch submit     # 배치 1개(800건) 제출
  python3 scripts/llm_label.py --batch status     # 진행 상황 확인
  python3 scripts/llm_label.py --batch collect    # 완료된 배치 결과를 CSV 에 반영
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
    "batch_dir": "data/labeled/batch",   # 배치 요청 파일 + 상태 (git 제외)
    # 배치 1개당 요청 수. gpt-5.5 대기열 한도 90만 토큰 / 건당 ~900 토큰 → 800건씩 하나만 대기
    "batch_chunk": 800,
    "batch_poll_sec": 120,     # --batch auto 상태 확인 주기
    "max_parse_fail": 3,       # 응답 파싱이 이만큼 실패하면 재제출 중단
}

MODELS = {
    "openai":    "gpt-5.5",       # 상위 모델 기본값 (--model 로 변경 가능)
    "anthropic": "claude-sonnet-5",
}

# 1M 토큰당 USD (입력, 출력)
PRICING = {
    "gpt-4o":          (2.50, 10.00),
    "gpt-4o-mini":     (0.15,  0.60),
    "gpt-4.1":         (2.00,  8.00),
    "gpt-4.1-mini":    (0.40,  1.60),
    "gpt-5.4":         (2.50, 15.00),
    "gpt-5.5":         (5.00, 30.00),
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
REASONING_EFFORT = "medium"


def is_reasoning_model(model):
    """gpt-5*, o1/o3/o4 계열은 temperature/max_tokens 미지원 → max_completion_tokens 사용"""
    return model.startswith(("gpt-5", "o1", "o3", "o4"))


# ══════════════════════════════════════════════════════════
# 데이터 준비
# ══════════════════════════════════════════════════════════
def load_targets(limit=None, checkpoint=True, quiet=False):
    log = (lambda *_: None) if quiet else print
    df = pd.read_csv(CONFIG["input_path"], low_memory=False)
    before = len(df)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df[df["date"].notna()]
    df = df[~df["media_name"].isin(BAD_MEDIA)]
    df = df[df["content_clean"].astype(str).str.len() >= 200]
    df = df.drop_duplicates(subset=["content_clean"])
    log(f"  정제: {before:,}건 → {len(df):,}건")

    # 체크포인트: 이미 라벨링된 것 제외
    out = Path(CONFIG["output_path"])
    if checkpoint and out.exists():
        done = pd.read_csv(out)
        done_ids = set(done["article_id"])
        df = df[~df["article_id"].isin(done_ids)]
        log(f"  체크포인트: {len(done_ids):,}건 완료 → 남은 {len(df):,}건")

    # 배치에서 응답은 왔지만 파싱 실패가 반복된 항목은 제외 (무한 재제출 방지)
    if checkpoint:
        bad = load_state().get("parse_fail", {})
        gave_up = {aid for aid, n in bad.items() if n >= CONFIG["max_parse_fail"]}
        if gave_up:
            df = df[~df["article_id"].isin(gave_up)]
            log(f"  반복 실패 제외: {len(gave_up):,}건 ({CONFIG['failed_path']} 참고)")

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


def row_text(row):
    return str(row["title_clean"])[:200], str(row["content_clean"])[:CONFIG["content_chars"]]


def openai_body(model, title, content):
    """chat.completions 요청 본문 (실시간 호출과 Batch API 공용)"""
    if is_reasoning_model(model):
        # 추론 토큰도 max_completion_tokens 에 포함되므로 넉넉히
        params = dict(max_completion_tokens=4000, reasoning_effort=REASONING_EFFORT)
    else:
        params = dict(temperature=0, max_tokens=200)
    return dict(
        model=model, **params,
        response_format={"type": "json_object"},
        messages=[{"role": "system", "content": SYSTEM_PROMPT},
                  {"role": "user", "content": USER_TMPL.format(title=title, content=content)}],
    )


def call_llm(client, provider, model, title, content):
    if provider == "openai":
        r = client.chat.completions.create(**openai_body(model, title, content))
        return r.choices[0].message.content
    user = USER_TMPL.format(title=title, content=content)
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


def make_record(row, raw):
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
        "title": row_text(row)[0],
    }


def label_one(client, provider, model, row):
    title, content = row_text(row)
    last = None
    for attempt in range(CONFIG["max_retries"]):
        try:
            return make_record(row, call_llm(client, provider, model, title, content))
        except Exception as e:
            last = e
            msg = str(e)
            if "429" in msg or "rate_limit" in msg.lower():
                time.sleep(20 + 10 * attempt)   # TPM 초과: 길게 대기
            else:
                time.sleep(2 ** attempt)        # 그 외: 지수 백오프
    return {"article_id": row["article_id"], "_error": str(last)[:200]}


# ══════════════════════════════════════════════════════════
# Batch API (OpenAI)
# ══════════════════════════════════════════════════════════
def _state_path():
    return Path(CONFIG["batch_dir"]) / "state.json"


def load_state():
    p = _state_path()
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {"batches": []}


def save_state(state):
    p = _state_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")


def batch_submit(client, model, df):
    """배치 하나(batch_chunk 건)만 제출. 대기열 토큰 한도 때문에 대기 중인 배치가 있으면 건너뜀"""
    state = load_state()
    if any(not b["collected"] for b in state["batches"]):
        print("\n대기 중인 배치가 있습니다. 완료 후 collect → submit 하세요 (또는 --batch auto).")
        return False
    if df.empty:
        print("\n제출할 항목이 없습니다.")
        return False

    bdir = Path(CONFIG["batch_dir"])
    bdir.mkdir(parents=True, exist_ok=True)
    chunk = df.iloc[:CONFIG["batch_chunk"]]
    path = bdir / f"requests_{time.strftime('%Y%m%d_%H%M%S')}.jsonl"
    with path.open("w", encoding="utf-8") as f:
        for _, row in chunk.iterrows():
            f.write(json.dumps({
                "custom_id": row["article_id"], "method": "POST",
                "url": "/v1/chat/completions",
                "body": openai_body(model, *row_text(row)),
            }, ensure_ascii=False) + "\n")
    with path.open("rb") as f:
        up = client.files.create(file=f, purpose="batch")
    b = client.batches.create(
        input_file_id=up.id, endpoint="/v1/chat/completions", completion_window="24h",
        metadata={"model": model, "reasoning_effort": REASONING_EFFORT},
    )
    state["batches"].append({
        "id": b.id, "model": model, "reasoning_effort": REASONING_EFFORT,
        "n": len(chunk), "request_file": str(path), "collected": False,
        "ids": chunk["article_id"].tolist(),
    })
    save_state(state)
    print(f"  제출: {b.id}  ({len(chunk):,}건, 남은 {len(df) - len(chunk):,}건)")
    return True


def batch_auto(client, model):
    """collect → submit 을 반복해 남은 항목이 없을 때까지 진행 (중단 후 재실행 가능)"""
    while True:
        batch_collect(client, quiet=True)
        state = load_state()
        waiting = any(not b["collected"] for b in state["batches"])
        recent = [b.get("status") for b in state["batches"][-3:]]
        if not waiting and len(recent) == 3 and all(s == "failed" for s in recent):
            sys.exit("\n❌ 배치가 3번 연속 실패해 중단합니다. --batch status 로 원인을 확인하세요.")
        if not waiting:
            df = load_targets(quiet=True)
            if df.empty:
                break
            batch_submit(client, model, df)
        time.sleep(CONFIG["batch_poll_sec"])
    print_summary(Path(CONFIG["output_path"]), 0)
    print("\n✅ 전체 배치 완료")


def batch_status(client):
    state = load_state()
    if not state["batches"]:
        print("\n제출된 배치가 없습니다.")
        return
    for b in state["batches"]:
        r = client.batches.retrieve(b["id"])
        c = r.request_counts
        done = f"완료 {c.completed:,}/{c.total:,}  실패 {c.failed:,}" if c else ""
        print(f"  {b['id']}  {r.status:11s}  {done}  {'(반영됨)' if b['collected'] else ''}")


def batch_collect(client, quiet=False):
    state = load_state()
    parse_fail = state.setdefault("parse_fail", {})
    rows = load_targets(checkpoint=False, quiet=True).set_index("article_id", drop=False)
    out_path = Path(CONFIG["output_path"])
    results, failures = [], []
    collected = 0

    for b in state["batches"]:
        if b["collected"]:
            continue
        r = client.batches.retrieve(b["id"])
        if r.status not in ("completed", "expired", "cancelled", "failed"):
            c = r.request_counts
            prog = f" {c.completed + c.failed:,}/{c.total:,}" if c and c.total else ""
            print(f"  [{time.strftime('%H:%M:%S')}] {b['id']}: {r.status}{prog}")
            continue
        collected += 1

        seen = set()
        for fid in (r.output_file_id, r.error_file_id):
            if not fid:
                continue
            for line in client.files.content(fid).text.splitlines():
                if not line.strip():
                    continue
                d = json.loads(line)
                aid = d["custom_id"]
                seen.add(aid)
                resp = d.get("response") or {}
                try:
                    if resp.get("status_code") != 200:
                        raise ValueError(d.get("error") or resp.get("body", {}).get("error"))
                    raw = resp["body"]["choices"][0]["message"]["content"]
                    results.append(make_record(rows.loc[aid], raw))
                except Exception as e:
                    parse_fail[aid] = parse_fail.get(aid, 0) + 1
                    failures.append({"article_id": aid, "_error": str(e)[:200]})
        # 결과가 없는 항목은 다음 submit 때 다시 제출됨
        if r.status == "failed":
            # 배치 전체 실패 (대기열 한도 등) → 항목별 로그 대신 원인만 출력
            errs = "; ".join(e.message for e in (r.errors.data if r.errors else []))
            print(f"  {b['id']}: 배치 실패 — {errs[:200]}")
        else:
            failures += [{"article_id": aid, "_error": f"batch {r.status}: 결과 없음"}
                         for aid in b["ids"] if aid not in seen]
        b["collected"] = True
        b["status"] = r.status
        print(f"  [{time.strftime('%H:%M:%S')}] {b['id']}: {r.status} → 수집 "
              f"(성공 누적 {len(results):,}건)")

    if results:
        pd.DataFrame(results).to_csv(out_path, mode="a", header=not out_path.exists(),
                                     index=False, encoding="utf-8-sig")
    save_state(state)
    if failures:
        fp = Path(CONFIG["failed_path"])
        pd.DataFrame(failures).to_csv(fp, mode="a", header=not fp.exists(),
                                      index=False, encoding="utf-8-sig")
    if not quiet and collected and out_path.exists():
        print_summary(out_path, len(failures))
        if failures:
            print("  실패 항목은 --batch submit 을 다시 실행하면 재제출됩니다.")


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
    print(f"\n  모델: {model}")
    print(f"  대상: {len(df):,}건")
    print(f"  입력 토큰 ≈ {in_tok/1e6:.2f}M / 출력 토큰 ≈ {out_tok/1e6:.2f}M"
          + ("  (+ 추론 토큰 별도)" if is_reasoning_model(model) else ""))
    if model not in PRICING:
        print("  예상 비용: 가격표에 없는 모델 (PRICING 에 추가하면 계산됨)")
        return
    pin, pout = PRICING[model]
    usd = in_tok / 1e6 * pin + out_tok / 1e6 * pout
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
    ap.add_argument("--reasoning-effort", default="medium",
                    choices=["minimal", "low", "medium", "high"],
                    help="gpt-5/o 계열 추론 강도")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--shuffle", action="store_true", help="--limit 표본을 무작위로 (테스트용)")
    ap.add_argument("--batch", choices=["submit", "status", "collect", "auto"],
                    help="OpenAI Batch API 사용 (비용 50%%, 최대 24시간)")
    a = ap.parse_args()
    if a.batch and a.provider != "openai":
        sys.exit("--batch 는 --provider openai 에서만 지원합니다.")

    model = a.model or MODELS[a.provider]
    global SHUFFLE, REASONING_EFFORT
    SHUFFLE = a.shuffle
    REASONING_EFFORT = a.reasoning_effort

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

    if a.batch in ("status", "collect", "auto"):
        client = make_client(a.provider)
        if a.batch == "auto":
            batch_auto(client, model)
        else:
            (batch_status if a.batch == "status" else batch_collect)(client)
        return

    df = load_targets(a.limit)
    if df.empty:
        print("\n✅ 모든 항목이 이미 라벨링되어 있습니다.")
        return

    estimate(df, model)
    if a.batch:
        print("  (Batch API: 위 비용의 50%)")
    if a.dry_run:
        print("\n(--dry-run 이므로 실제 호출 없이 종료)")
        return

    client = make_client(a.provider)
    if a.batch == "submit":
        batch_submit(client, model, df)
        return
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
    print_summary(out_path, len(failures))


def print_summary(out_path, n_failed):
    final = pd.read_csv(out_path)
    print(f"\n{'='*58}")
    print(f"완료: {len(final):,}건 저장  |  실패: {n_failed:,}건")
    print(f"  {out_path}")
    if n_failed:
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
