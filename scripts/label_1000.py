"""1000건 샘플링 + 정밀 프레이밍 라벨링

사용자 제시 6가지 프레이밍 유형 기준 (플로우차트 순서 엄수):
  1. 경보적  → 2. 방어적 → 3. 비교적
  → 4. 낙관적 → 5. 비관적 → 6. 중립적

제목 우선(80%), 본문은 참고용.
"""

import re
import pandas as pd
from pathlib import Path

# ══════════════════════════════════════════════════════════
# 1. 패턴 사전
# ══════════════════════════════════════════════════════════

# ── 경보적: 극단적 위기감, 공포 조장, 최악 시나리오 ──
ALARMIST = [
    r"폭락", r"붕괴", r"공포", r"패닉", r"쇼크", r"충격",
    r"비상(?!구|착륙|계단|탈출|약|전화|연락)",   # "비상구"등 무관 단어 제외
    r"경보", r"초비상", r"초긴장", r"사태", r"대란",
    r"시한\s*폭탄", r"뇌관", r"도미노", r"칼바람", r"쓰나미",
    r"최악", r"파국", r"나락", r"벼랑\s*끝", r"파산", r"디폴트",
    r"모라토리엄", r"아수라장", r"재앙",
    r"폭풍", r"전운", r"위험\s*수위", r"불안\s*확산",
    r"초유", r"비상등",
    r"\S+폭탄",        # 매도폭탄, 관세폭탄 등
    r"\S+대란",        # 전세대란, 물가대란 등
    r"위기",
    # 환율·물가·금리 폭등은 경보적
    r"(?:환율|물가|금리|유가)\s*\S*\s*폭등",
    r"폭등\s*\S*\s*(?:공포|충격|비상|위기)",
    r"긴급\s*(?:처방|대응|조치|점검|대책)",
]

# "위기" 단독인데 방어적 구조가 함께 있으면 방어적으로 넘김
# → classify_title 함수에서 처리

# ── 방어적: 부정 상황 + 반전 → 긍정 결과 ──
DEFENSIVE = [
    r"에도\s*불구(?:하고)?",
    r"(?:에도|속에서도|속에도|임에도)\s*.{0,20}(?:선방|버텨|방어|유지|양호|견조|반등|회복|상승|증가|개선|호조|흑자|성장|도약|돌파|상회|웃돌)",
    r"(?:침체|위기|불황|둔화|부진|악화|역풍|악재|리스크|불확실|불안|하락|감소|위축|전쟁|갈등|코로나|팬데믹|한파|냉각)\s*(?:속에서도|속에도|에도|불구)",
    r"(?:위기|역풍|악재|불황|침체|둔화|부진|악화|리스크|한파|냉각)\s*.{0,15}(?:극복|선방|회복|반등|버텨|견뎌|방어|돌파|타개|이겨|견인)",
    r"(?:불황|위기|침체|역경|어려움|난관|악재|하락)\s*.{0,8}(?:딛고|뚫고|이겨내|극복)",
    r"우려\s*(?:에도|불구)\s*.{0,15}(?:선방|회복|증가|상승|개선|유지|견조)",
    r"(?:꾸역꾸역|버텨냈|선방했|견뎌냈|꿋꿋|저력|딛고|뚫고|이겨내)",
]

# ── 비교적: 비교 자체가 기사의 핵심 주제 ──
COMPARATIVE_STRONG = [
    r"vs\.?",                                    # A vs B
    r"격차\s*.{0,10}(?:벌어|좁혀|확대|축소|역대|사상)",
    r"(?:꼴찌|최하위|최상위|1위\s*(?:탈환|달성|등극))",
    r"순위\s*.{0,10}(?:올라|내려|상승|하락|유지)",
    r"반면\s*.{0,30}(?:증가|감소|상승|하락|성장|위축|호조|부진)",
    r"(?:미국|중국|일본|EU|유럽|OECD|G7|G20|선진국|주요국)\s*.{0,10}(?:대비|비해|보다)\s*.{0,10}(?:낮|높|많|적|부진|양호|앞서|뒤처|꼴찌|1위)",
    r"(?:서울|수도권)\s*.{0,5}(?:vs|VS|대비|반면)\s*.{0,5}(?:지방|비수도권|지역)",
    r"삼성\s*.{0,5}(?:vs|VS|대비|반면|추월|앞서)\s*.{0,5}(?:TSMC|애플|엔비디아|인텔)",
]
# 비교적 제외: "전년 대비 N% 성장/감소" 같은 단순 수치 보고
COMPARATIVE_EXCLUDE = [
    r"(?:전년|전기|전월|전분기|작년|지난\s*달|지난\s*해)\s*(?:동기\s*)?대비\s*[\d\.]+",
    r"(?:전년|전월|전기)\s*대비\s*(?:보다)?\s*(?:소폭|크게|다소)?\s*(?:증가|감소|상승|하락|성장)",
]

# ── 낙관적: 긍정 성과/전망 강조 ──
OPTIMISTIC = [
    r"(?:성장|수출|실적|경기|투자|고용|소비)\s*(?:세\s*)?(?:호조|급증|급등|급성장)",
    r"호조", r"상승세", r"회복세", r"반등",
    r"최고치", r"사상\s*최대", r"역대\s*최대(?!\s*(?:적자|손실|부채|빚))",
    r"신고가", r"신기록",
    r"흑자\s*전환", r"흑자\s*(?:기록|달성|지속|유지)",
    r"상향\s*(?:조정|전망)", r"긍정적\s*(?:전망|평가|신호)",
    r"개선", r"활기", r"훈풍", r"청신호", r"순항",
    r"고공\s*행진", r"도약", r"쾌속\s*성장", r"쾌조",
    r"기대감", r"낙관", r"밝은\s*전망",
    r"플러스\s*성장", r"견조",
    r"수혜", r"성장\s*동력",
    r"턴어라운드",
    r"(?:코스피|코스닥|주가|지수)\s*.{0,5}(?:급등|사상\s*최고|신고가|돌파)",
    r"경기\s*회복", r"투자\s*확대", r"수출\s*(?:호조|급증|최대|증가)",
]
# 함정 방지: "역대 최대 적자" 등은 낙관이 아님
OPTIMISTIC_EXCLUDE = [
    r"(?:역대|사상)\s*최대\s*(?:적자|손실|감소|하락|부채|빚|실업)",
    r"(?:역대|사상)\s*최고\s*(?:적자|손실|부채|빚|실업|금리)",
]

# ── 비관적: 부정 현실/하락 보도 (공포감까지는 아님) ──
PESSIMISTIC = [
    r"둔화", r"하락", r"감소", r"위축", r"부진",
    r"침체", r"역성장",
    r"적자(?!\s*전환)",    # "흑자 전환" 제외
    r"마이너스\s*(?:성장|전환|기록)",
    r"약세", r"내리막", r"먹구름",
    r"악화", r"하향\s*(?:조정|전망)",
    r"줄어", r"떨어",
    r"감소세", r"역대\s*최저", r"최저치",
    r"급감", r"급락",
    r"(?:수출|고용|소비|투자|생산|실적)\s*.{0,8}(?:감소|둔화|위축|부진|하락)",
    r"(?:경기|경제|성장)\s*.{0,8}(?:둔화|부진|침체|위축)",
    r"(?:소비|투자)\s*심리\s*(?:위축|냉각)",
    r"내수\s*부진",
    r"(?:꺾여|꺾이)",
    r"(?:세\s*)?지속\s*(?:되는|된다|하는)?\s*(?:부진|하락|감소)",
    r"무역\s*적자", r"경상\s*적자",
    r"추락", r"곤두박질",
]

# ── 중립적: 감정 없는 사실 전달, 통계·정책 보도 ──
NEUTRAL = [
    r"동결", r"보합", r"횡보",
    r"(?:한국은행|한은|기재부|기획재정부|통계청|금융위|금통위|정부|재정부)\s*.{0,15}(?:발표|집계|전망|결정|확정|유지|동결)",
    r"집계",
    r"(?:으로|로)\s*(?:결정|확정)",
    r"\(속보\)", r"\(종합\d*\)", r"\(1보\)", r"\(2보\)", r"\(3보\)",
    r"유지\s*(?:하기로|결정|확정)",
    r"소비자물가\s*\S+\s*(?:상승|집계|발표)",
    r"기준금리\s*\S+\s*(?:유지|동결|결정)",
]


# ══════════════════════════════════════════════════════════
# 2. 판별 함수
# ══════════════════════════════════════════════════════════

def _match(text: str, patterns: list[str]) -> bool:
    for p in patterns:
        if re.search(p, text, re.IGNORECASE):
            return True
    return False


def _count(text: str, patterns: list[str]) -> int:
    total = 0
    for p in patterns:
        total += len(re.findall(p, text, re.IGNORECASE))
    return total


def classify(title: str) -> str:
    """플로우차트 순서대로 분류: 경보→방어→비교→낙관→비관→중립"""
    t = title.strip()

    # ── 1단계: 경보적 ──
    if _match(t, ALARMIST):
        # "위기" 단독이고 방어적 구조가 함께 있으면 방어적으로 넘김
        alarmist_only_crisis = (
            bool(re.search(r"위기", t))
            and _count(t, ALARMIST) == 1
        )
        if alarmist_only_crisis and _match(t, DEFENSIVE):
            pass  # 방어적 판단으로 이동
        else:
            return "alarmist"

    # ── 2단계: 방어적 ──
    if _match(t, DEFENSIVE):
        return "defensive"

    # ── 3단계: 비교적 ──
    if _match(t, COMPARATIVE_STRONG) and not _match(t, COMPARATIVE_EXCLUDE):
        return "comparative"

    # ── 4단계: 낙관적 ──
    if _match(t, OPTIMISTIC) and not _match(t, OPTIMISTIC_EXCLUDE):
        return "optimistic"

    # ── 5단계: 비관적 ──
    if _match(t, PESSIMISTIC):
        return "pessimistic"

    # ── 6단계: 중립적 ──
    return "neutral"


# ══════════════════════════════════════════════════════════
# 3. 샘플링 + 라벨링
# ══════════════════════════════════════════════════════════

def sample_and_label(
    input_path: str = "data/processed/dataset.csv",
    output_path: str = "data/labeled/labeled_1000.csv",
    n_samples: int = 1000,
    random_state: int = 42,
):
    print("=" * 60)
    print("  전체 라벨링 후 균형 샘플링 (클래스별 균등 분포)")
    print("=" * 60)

    # 1. 로드
    df = pd.read_csv(input_path)
    print(f"[1] 전체 기사: {len(df):,}건")

    # 2. 필터링
    df = df.dropna(subset=["title_clean"])
    df = df[df["title_clean"].str.len() >= 10].copy()
    df = df.drop_duplicates(subset=["title_clean"], keep="first")
    print(f"[2] 필터링 후: {len(df):,}건")

    # 3. 전체 라벨링 먼저
    print("[3] 전체 데이터 라벨링 중...")
    df["framing_label"] = df["title_clean"].apply(classify)

    # 전체 분포 출력
    full_dist = df["framing_label"].value_counts()
    print("\n  전체 분포:")
    for label, cnt in full_dist.items():
        print(f"    {label:12s}: {cnt:5d}건")

    # 4. 균형 샘플링: 클래스별 n_samples//6 건 추출
    LABELS = ["alarmist", "defensive", "comparative", "optimistic", "pessimistic", "neutral"]
    per_label = n_samples // len(LABELS)          # 기본 167건
    sampled_dfs = []

    for label in LABELS:
        subset = df[df["framing_label"] == label]
        n = min(len(subset), per_label)
        sampled_dfs.append(subset.sample(n=n, random_state=random_state))
        print(f"    {label:12s}: {n}건 선택 (가용 {len(subset)}건)")

    sampled = pd.concat(sampled_dfs, ignore_index=True)

    # 부족분은 가장 많은 neutral에서 보충
    if len(sampled) < n_samples:
        already = sampled.index.tolist()
        remain = df[~df.index.isin(already)]
        extra = remain.sample(
            n=min(n_samples - len(sampled), len(remain)),
            random_state=random_state,
        )
        sampled = pd.concat([sampled, extra], ignore_index=True)

    sampled = sampled.sample(frac=1, random_state=random_state).reset_index(drop=True)
    print(f"\n[4] 최종 샘플: {len(sampled):,}건")

    # 5. 분포 출력
    dist = sampled["framing_label"].value_counts()
    print(f"\n{'─'*50}")
    print("  최종 라벨 분포")
    print(f"{'─'*50}")
    for label, cnt in dist.items():
        pct = cnt / len(sampled) * 100
        bar = "█" * int(pct / 2)
        print(f"  {label:12s}: {cnt:4d}건 ({pct:5.1f}%) {bar}")
    print(f"{'─'*50}")
    print(f"  합계        : {len(sampled):4d}건")

    # 6. 라벨별 대표 제목 5건 출력
    print(f"\n{'='*60}")
    print("  라벨별 대표 제목 (각 5건)")
    print(f"{'='*60}")
    for label in ["alarmist", "defensive", "comparative", "optimistic", "pessimistic", "neutral"]:
        subset = sampled[sampled["framing_label"] == label]
        print(f"\n  ▶ {label} ({len(subset)}건):")
        for _, row in subset.head(5).iterrows():
            print(f"    - {row['title_clean']}")

    # 7. 저장
    keep_cols = [c for c in [
        "article_id", "title", "title_clean",
        "framing_label", "media_name", "media_group",
        "event_type", "date",
    ] if c in sampled.columns]
    result = sampled[keep_cols].reset_index(drop=True)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"\n[5] 저장 완료: {output_path} ({len(result)}건)")
    return result


if __name__ == "__main__":
    sample_and_label()
