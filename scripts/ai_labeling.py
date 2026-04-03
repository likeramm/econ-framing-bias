"""AI 기반 프레이밍 라벨링 스크립트

MindLogic API (gpt-5.4-mini)를 사용하여 경제뉴스 제목을 6가지 프레이밍 유형으로 분류.
토큰 절약을 위해 제목만 사용하고, 10건씩 배치 처리한다.
중단 후 재실행 시 자동으로 이어서 처리된다.
"""

import json
import time
import glob
import pandas as pd
import requests
from pathlib import Path

# ── 설정 ──
API_URL = "https://factchat-cloud.mindlogic.ai/v1/gateway/chat/completions"
API_KEY = "Wy3GiUNM0s5k9DVqH0ZBIhPIo3Gzd5fQ"
MODEL = "gpt-5.4-mini"
BATCH_SIZE = 10

RAW_DIR = Path("data/raw")
OUTPUT_PATH = Path("data/processed/ai_labeled_dataset.csv")
PROGRESS_PATH = Path("data/processed/ai_labeling_progress.json")

SYSTEM_PROMPT = """너는 경제뉴스 프레이밍 분류 전문가야.
뉴스 제목을 보고 다음 6가지 프레이밍 유형 중 하나로 분류해.

1. optimistic (낙관적): 성장, 호조, 반등, 최고치 등 긍정적 전망
2. pessimistic (비관적): 둔화, 하락, 침체, 부진 등 부정적 전망
3. alarmist (경보적): 위기, 폭락, 붕괴, 공포 등 극단적 경고
4. defensive (방어적): "~에도 불구하고", 위기 속 선방, 역경 극복
5. comparative (비교적): 국가/시기 간 비교, 대비, 순위
6. neutral (중립적): 사실 전달, 동결, 발표, 집계 등 감정 없는 보도

반드시 JSON 배열로만 답해. 설명 없이 라벨만.
예시: ["optimistic","neutral","pessimistic"]"""


def load_all_articles():
    """모든 크롤링 CSV를 합쳐서 중복 제거"""
    files = sorted(glob.glob(str(RAW_DIR / "*.csv")))
    dfs = []
    for f in files:
        try:
            df = pd.read_csv(f)
            dfs.append(df)
        except Exception as e:
            print(f"  ⚠ 파일 읽기 실패: {f} - {e}")

    combined = pd.concat(dfs, ignore_index=True)
    before = len(combined)
    combined = combined.drop_duplicates(subset=["url"], keep="first").reset_index(drop=True)
    after = len(combined)
    print(f"  총 {before:,}건 → 중복 제거 후 {after:,}건")
    return combined


def load_progress():
    """이전 진행 상황 로드"""
    if PROGRESS_PATH.exists():
        with open(PROGRESS_PATH, "r") as f:
            return json.load(f)
    return {"completed": 0, "results": [], "total_tokens": 0}


def save_progress(progress):
    PROGRESS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(PROGRESS_PATH, "w") as f:
        json.dump(progress, f, ensure_ascii=False)


def classify_batch(titles: list[str]) -> tuple[list[str], int]:
    """제목 배치를 API로 분류"""
    numbered = "\n".join(f"{i+1}. {t}" for i, t in enumerate(titles))
    user_msg = f"다음 {len(titles)}개 뉴스 제목을 분류해줘:\n{numbered}"

    resp = requests.post(
        API_URL,
        headers={
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json",
        },
        json={
            "model": MODEL,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            "max_tokens": 200,
            "temperature": 0,
        },
        timeout=30,
    )
    resp.raise_for_status()
    data = resp.json()

    content = data["choices"][0]["message"]["content"].strip()
    tokens_used = data.get("usage", {}).get("total_tokens", 0)

    # JSON 파싱
    try:
        if "```" in content:
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
        labels = json.loads(content.strip())
    except json.JSONDecodeError:
        valid = {"optimistic", "pessimistic", "alarmist", "defensive", "comparative", "neutral"}
        labels = []
        for word in content.replace('"', '').replace(',', ' ').replace('[', '').replace(']', '').split():
            w = word.strip().lower()
            if w in valid:
                labels.append(w)

    # 개수 맞추기
    if len(labels) < len(titles):
        labels.extend(["neutral"] * (len(titles) - len(labels)))
    elif len(labels) > len(titles):
        labels = labels[:len(titles)]

    return labels, tokens_used


def main():
    print(f"\n{'='*60}")
    print(f"  AI 기반 프레이밍 라벨링")
    print(f"  모델: {MODEL}")
    print(f"{'='*60}\n")

    # 1. 데이터 로드
    print("[1] 데이터 로드 중...")
    df = load_all_articles()
    titles = df["title"].tolist()
    total = len(titles)

    # 2. 이전 진행 확인
    progress = load_progress()
    start_idx = progress["completed"]
    results = progress["results"]
    total_tokens = progress["total_tokens"]

    if start_idx > 0:
        print(f"\n  ▶ 이전 진행 이어서: {start_idx}/{total}건 완료, {total_tokens:,}토큰 사용")

    total_batches = (total + BATCH_SIZE - 1) // BATCH_SIZE
    print(f"\n[2] 라벨링 시작 ({total:,}건, {BATCH_SIZE}건씩 = {total_batches}배치)\n")

    # 3. 배치 처리
    for i in range(start_idx, total, BATCH_SIZE):
        batch = titles[i:i+BATCH_SIZE]
        batch_num = i // BATCH_SIZE + 1

        try:
            labels, tokens = classify_batch(batch)
            results.extend(labels)
            total_tokens += tokens

            progress["completed"] = i + len(batch)
            progress["results"] = results
            progress["total_tokens"] = total_tokens
            save_progress(progress)

            pct = (i + len(batch)) / total * 100
            print(f"  [{batch_num}/{total_batches}] {i+len(batch):,}/{total:,}건 ({pct:.0f}%) | {tokens}토큰 | 누적 {total_tokens:,}토큰")

        except requests.exceptions.HTTPError as e:
            status = e.response.status_code if e.response else 0
            if status in (429, 402):
                print(f"\n  ⚠ 토큰 소진 또는 Rate Limit! {len(results)}건까지 저장됨.")
                print(f"  다시 실행하면 이어서 진행됩니다.")
                break
            print(f"  [{batch_num}] HTTP 오류: {e}")
            time.sleep(5)
            continue

        except Exception as e:
            print(f"  [{batch_num}] 오류: {e}, 5초 후 재시도...")
            time.sleep(5)
            try:
                labels, tokens = classify_batch(batch)
                results.extend(labels)
                total_tokens += tokens
                progress["completed"] = i + len(batch)
                progress["results"] = results
                progress["total_tokens"] = total_tokens
                save_progress(progress)
            except Exception as e2:
                print(f"  재시도 실패: {e2}, neutral로 채움")
                results.extend(["neutral"] * len(batch))

        time.sleep(0.3)

    # 4. 결과 저장
    labeled_count = min(len(results), total)
    df_out = df.iloc[:labeled_count].copy()
    df_out["framing_label"] = results[:labeled_count]

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")

    print(f"\n{'='*60}")
    print(f"  라벨링 완료: {labeled_count:,}/{total:,}건")
    print(f"  총 토큰 사용: {total_tokens:,}")
    print(f"  저장: {OUTPUT_PATH}")
    print(f"{'='*60}")

    print(f"\n  클래스별 분포:")
    print(df_out["framing_label"].value_counts().to_string())
    print()


if __name__ == "__main__":
    main()
