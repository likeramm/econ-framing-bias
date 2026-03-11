"""규칙 기반 프레이밍 라벨 생성 스크립트

제목 + 본문 키워드 패턴으로 초기 프레이밍 라벨을 자동 생성하고,
고신뢰도 샘플을 선별하여 학습 데이터를 만든다.

6가지 라벨: optimistic, pessimistic, alarmist, defensive, comparative, neutral
"""

import re
import pandas as pd
from pathlib import Path


# ===== 프레이밍 유형별 키워드/패턴 사전 =====

OPTIMISTIC_PATTERNS = [
    # 성장/호조
    r"성장세\s*지속", r"호조", r"상승세", r"회복세", r"반등",
    r"최고치", r"사상\s*최대", r"역대\s*최대", r"신고가", r"상향\s*조정",
    r"상향\s*전망", r"긍정적", r"개선", r"활기", r"훈풍",
    r"청신호", r"순항", r"견조", r"호황", r"급등",
    r"폭등", r"쾌속\s*성장", r"턴어라운드", r"흑자\s*전환",
    r"기대감", r"낙관", r"밝은\s*전망", r"가속", r"고공\s*행진",
    r"사상\s*최고", r"최고\s*실적", r"흑자", r"플러스\s*성장",
    r"양호", r"선방", r"수혜", r"도약", r"성장\s*동력",
    r"수출\s*호조", r"경기\s*회복", r"투자\s*확대",
]

PESSIMISTIC_PATTERNS = [
    # 하락/부진
    r"둔화", r"하락", r"감소", r"위축", r"부진",
    r"적자", r"마이너스", r"약세", r"침체", r"역성장",
    r"역대\s*최저", r"최저치", r"급감", r"폭감", r"곤두박질",
    r"내리막", r"빨간불", r"먹구름", r"악화", r"줄어",
    r"감소세", r"하향\s*조정", r"하향\s*전망", r"우울",
    r"추락", r"급락", r"감소.*지속", r"부진.*지속",
    r"무역\s*적자", r"경상\s*적자", r"성장률.*낮", r"떨어",
    r"둔화\s*우려", r"수출.*감소", r"고용.*감소",
]

ALARMIST_PATTERNS = [
    # 위기/경고
    r"위기", r"경고", r"비상", r"폭락", r"붕괴",
    r"충격", r"공포", r"패닉", r"쇼크", r"블랙",
    r"초비상", r"초긴장", r"초유", r"사태", r"대란",
    r"경보", r"빨간\s*불", r"비상등", r"긴급", r"전운",
    r"위험\s*수위", r"뇌관", r"시한\s*폭탄", r"불안\s*확산",
    r"도미노", r"먹구름.*짙", r"칼바람", r"한파.*경제",
    r"최악", r"파국", r"벼랑\s*끝", r"나락", r"폭풍",
    r"쓰나미", r"아수라장", r"재앙", r"파산", r"디폴트",
    r"IMF", r"모라토리엄",
]

DEFENSIVE_PATTERNS = [
    # "~에도 불구하고", "우려이나 ~"
    r"불구.*(?:성장|반등|회복|양호|선방|견조)",
    r"우려.*(?:선방|양호|견조|회복|극복)",
    r"(?:어렵|힘들|악화).*(?:버텨|버텼|선방|극복|견뎌)",
    r"(?:위기|우려).*(?:기회|전환점|반전)",
    r"역풍.*(?:속|에도).*(?:성장|회복)",
    r"(?:불확실|불안).*(?:속|에도).*(?:견조|양호)",
    r"악재.*(?:속|에도).*(?:선방|반등)",
    r"(?:코로나|팬데믹).*(?:회복|극복)",
    r"(?:위기|난관).*(?:돌파|극복|타개)",
    r"꾸역꾸역", r"버텨", r"선방했", r"견뎌",
]

COMPARATIVE_PATTERNS = [
    # 비교
    r"대비", r"비교", r"대비.*(?:높|낮|양호|부진)",
    r"(?:미국|중국|일본|EU|유럽).*(?:대비|비교|반면|한편)",
    r"(?:한국|국내).*(?:반면|한편|대비)",
    r"(?:선진국|주요국|G7|OECD).*(?:대비|비교|중)",
    r"(?:전년|전기|전월|작년).*(?:대비|비교|동기)",
    r"(?:상반기|하반기|분기).*(?:대비|비교)",
    r"(?:세계|글로벌).*(?:대비|비교|순위)",
    r"(?:순위|랭킹).*(?:올라|내려|유지)",
    r"(?:격차|차이).*(?:벌어|좁혀|확대|축소)",
]

NEUTRAL_PATTERNS = [
    # 사실 보도
    r"(?:기록|집계|발표|보고)$",
    r"^\d+.*%\s*(?:기록|집계|발표)",
    r"(?:한국은행|한은|기재부|통계청).*(?:발표|집계|보고)",
    r"(?:전망치|예상치|컨센서스).*(?:부합|일치)",
    r"(?:보합|횡보|유지)",
    r"(?:동결|유지).*(?:결정|발표)",
]


def compute_pattern_score(text: str, patterns: list[str]) -> int:
    """텍스트에서 패턴 매칭 횟수를 반환"""
    score = 0
    for pattern in patterns:
        matches = re.findall(pattern, text)
        score += len(matches)
    return score


def classify_framing(title: str, content: str) -> tuple[str, float]:
    """규칙 기반 프레이밍 분류

    Returns:
        (label, confidence) - confidence는 0~1 사이 값
    """
    # 제목에 2배 가중치
    combined = f"{title} {title} {content[:500]}"

    scores = {
        "defensive": compute_pattern_score(combined, DEFENSIVE_PATTERNS),
        "alarmist": compute_pattern_score(combined, ALARMIST_PATTERNS),
        "optimistic": compute_pattern_score(combined, OPTIMISTIC_PATTERNS),
        "pessimistic": compute_pattern_score(combined, PESSIMISTIC_PATTERNS),
        "comparative": compute_pattern_score(combined, COMPARATIVE_PATTERNS),
        "neutral": compute_pattern_score(combined, NEUTRAL_PATTERNS),
    }

    # defensive 패턴은 복합 패턴이므로 보너스
    scores["defensive"] *= 2

    total = sum(scores.values())
    if total == 0:
        return "neutral", 0.3

    best_label = max(scores, key=scores.get)
    best_score = scores[best_label]

    # 2위와의 차이로 신뢰도 계산
    sorted_scores = sorted(scores.values(), reverse=True)
    gap = sorted_scores[0] - sorted_scores[1] if len(sorted_scores) > 1 else sorted_scores[0]
    confidence = min(1.0, (gap / max(total, 1)) + (best_score / max(total, 1)) * 0.5)

    return best_label, confidence


def generate_labels(
    input_path: str = "data/processed/dataset.csv",
    output_path: str = "data/labeled/framing_labels.csv",
    min_confidence: float = 0.4,
    min_content_length: int = 100,
    target_count: int = 600,
):
    """라벨링 데이터 생성"""
    df = pd.read_csv(input_path)
    print(f"전체 기사 수: {len(df)}")

    # 콘텐츠 길이 필터
    df = df[df["content_length"] >= min_content_length].copy()
    print(f"콘텐츠 길이 >= {min_content_length}: {len(df)}")

    # 프레이밍 분류 적용
    results = df.apply(
        lambda row: classify_framing(
            str(row["title_clean"]), str(row["content_clean"])
        ),
        axis=1,
    )
    df["framing_label"] = [r[0] for r in results]
    df["label_confidence"] = [r[1] for r in results]

    # 신뢰도 필터
    high_conf = df[df["label_confidence"] >= min_confidence].copy()
    print(f"신뢰도 >= {min_confidence}: {len(high_conf)}")

    # 라벨별 분포 확인
    print("\n라벨별 분포 (필터 전):")
    print(df["framing_label"].value_counts())
    print(f"\n라벨별 분포 (신뢰도 >= {min_confidence}):")
    print(high_conf["framing_label"].value_counts())

    # 라벨별 균형 샘플링
    labels = ["optimistic", "pessimistic", "alarmist", "defensive", "comparative", "neutral"]
    per_label = target_count // len(labels)

    sampled_dfs = []
    for label in labels:
        subset = high_conf[high_conf["framing_label"] == label]
        # 신뢰도 높은 순으로 정렬
        subset = subset.sort_values("label_confidence", ascending=False)
        n = min(len(subset), per_label)
        sampled_dfs.append(subset.head(n))
        print(f"  {label}: {n}건 선택 (가용: {len(subset)}건)")

    result = pd.concat(sampled_dfs, ignore_index=True)

    # 부족한 경우 남은 데이터에서 추가 샘플링
    if len(result) < target_count:
        remaining = high_conf[~high_conf.index.isin(result.index)]
        remaining = remaining.sort_values("label_confidence", ascending=False)
        extra = remaining.head(target_count - len(result))
        result = pd.concat([result, extra], ignore_index=True)

    # 셔플
    result = result.sample(frac=1, random_state=42).reset_index(drop=True)

    # 출력 컬럼 선택
    output_cols = [
        "article_id", "title", "content", "framing_label",
        "label_confidence", "media_name", "event_type", "date",
    ]
    result = result[output_cols]

    # 저장
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"\n최종 라벨링 데이터: {len(result)}건 → {output_path}")
    print("\n최종 라벨 분포:")
    print(result["framing_label"].value_counts())

    return result


if __name__ == "__main__":
    generate_labels()
