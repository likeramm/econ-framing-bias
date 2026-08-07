# NLP기반 경제 뉴스 프레이밍 편향 탐지 및 주가 영향 분석 시스템

> Economic News Framing Bias Detection and Stock Price Impact Analysis System Using NLP

한국은행이 "GDP 성장률 2.3%"를 발표하면, 어떤 언론은 **"견조한 성장세 지속"**, 다른 언론은 **"성장 둔화 우려 확대"**로 보도합니다. 본 프로젝트는 이러한 **프레이밍 편향**을 NLP 모델로 자동 탐지하고, 편향이 실제 주가에 미치는 인과적 영향을 통계적으로 검증합니다.

---

## 핵심 아이디어

```
같은 숫자, 다른 해석 → 다른 투자 심리 → 다른 주가 반응?

  GDP 2.3% 발표
       ↓
  ┌─ A언론: "견조한 성장세 지속"        (낙관적)
  ├─ B언론: "성장 둔화 우려 확대"        (비관적)
  ├─ C언론: "우려에도 불구하고 선방"     (방어적)
  └─ D언론: "OECD 평균 상회"            (비교적)
       ↓
  프레이밍 편향 점수 산출 → 주가 영향 인과분석
```

## 연구 질문

| # | 질문 |
|---|------|
| **RQ1** | 동일 경제 지표에 대해 한국 언론사들의 프레이밍은 어떻게 다른가? |
| **RQ2** | 언론사별 프레이밍 편향은 시간에 따라 일관된 패턴을 보이는가? |
| **RQ3** | 부정적 프레이밍 집중도가 높을수록 관련 섹터 주가 하락 폭이 큰가? |
| **RQ4** | 미디어 프레이밍이 소비심리지수(CCSI)를 매개하여 주가에 영향을 미치는가? |

## 프레이밍 유형 (6가지)

| 유형 | 설명 | 예시 헤드라인 |
|------|------|-------------|
| **낙관적** (Optimistic) | 긍정적 전망 강조 | "견조한 성장세 지속, 하반기 반등 기대" |
| **비관적** (Pessimistic) | 부정적 측면 부각 | "성장 둔화 우려 확대, 불확실성 고조" |
| **경고적** (Alarmist) | 위기감 조성 | "경제 위기 경고음, 금융위기 재현 우려" |
| **방어적** (Defensive) | 부정 속 긍정 강조 | "우려에도 불구하고 펀더멘털 견고" |
| **비교적** (Comparative) | 타국/과거 대비 비교 | "미국 대비 양호한 수준, OECD 평균 상회" |
| **중립적** (Neutral) | 사실 중심 전달 | "2분기 GDP 2.3% 기록, 금리 6연속 동결" |

## 시스템 파이프라인

```
[뉴스 크롤링]        [ECOS API]         [yfinance]
 38개 언론사           경제지표 7종        KOSPI + 섹터 ETF 14종
 10년간 14,135건      (금리,CPI,CCSI...)   10년간 35,821건
       │                  │                  │
       ▼                  ▼                  ▼
┌─────────────────────────────────────────────────┐
│              데이터 전처리 & 이벤트 매칭            │
└──────────────────────┬──────────────────────────┘
                       │
        ┌──────────────┼──────────────┐
        ▼              ▼              ▼
  KLUE-RoBERTa    KcELECTRA      키워드 극성
  프레이밍 6유형    감성 강도       분석
  분류             (-1 ~ +1)
        └──────────────┼──────────────┘
                       ▼
              편향 점수 산출 (-3 ~ +3)
                       │
        ┌──────────────┼──────────────┐
        ▼              ▼              ▼
   이벤트 스터디    그랜저 인과     매개분석
   (CAR 측정)     관계 검정     (편향→CCSI→주가)
        └──────────────┼──────────────┘
                       ▼
              React 대시보드 시각화
```

## 기술 스택

| 영역 | 기술 | 용도 |
|------|------|------|
| **프레이밍 분류** | KLUE-RoBERTa-large (Fine-tuned) | 6가지 프레이밍 유형 분류 |
| **감성 분석** | KcELECTRA-base-v2022 (Fine-tuned) | 감성 강도 연속 점수 산출 |
| **통계 분석** | statsmodels, linearmodels, pingouin | 이벤트 스터디, 그랜저, 패널회귀, 매개분석 |
| **데이터 수집** | Selenium, ECOS API, yfinance | 뉴스 크롤링, 경제지표, 주가 |
| **백엔드** | Django REST Framework | API 서빙, 모델 추론, DB 관리 |
| **프론트엔드** | React + Recharts | SPA 대시보드, 인터랙티브 차트 |
| **배포** | AWS EC2, Nginx, Gunicorn | 프로덕션 서비스 |

## 데이터 현황

| 데이터 | 규모 | 기간 | 출처 |
|--------|------|------|------|
| 뉴스 기사 | **14,135건** (38개 언론사, 15개 키워드) | 2016-03 ~ 2026-03 | 네이버뉴스 |
| 경제 지표 | 7종 (기준금리, 소비자물가, 전산업생산, 실업률, CCSI, 경상수지, 원/달러환율) | 2015-09 ~ 2026-03 (127개월) | 한국은행 ECOS API |
| 주가 데이터 | 14종목/지수 (KOSPI, KOSDAQ, 섹터 ETF 6종, 개별주 5종, KODEX200) | 2015-09 ~ 2026-03 (35,821건) | yfinance |
| 수동 라벨 | **3,262건** (모델 학습용) | — | 직접 라벨링 |
| 자동 라벨 | **13,886건** (fine-tuned KLUE-RoBERTa-large, F1 0.74) | — | 모델 추론 |

> 주가·지표를 2015-09부터 수집하는 이유: 뉴스 시작일(2016-03)에 대해서도 이벤트 스터디 추정기간 `[-120, -11]`을 확보하기 위함. 뉴스 × 주가 분석 가능 구간은 **10.0년**이다.

### 분석 대상 언론사 (10개)

| 보수 | 진보 | 경제지 | 중립 |
|------|------|--------|------|
| 조선일보 | 한겨레 | 한국경제 | 연합뉴스 |
| 중앙일보 | 경향신문 | 매일경제 | SBS |
| 동아일보 | | 서울경제 | |

## 프로젝트 구조

```
ver1/
├── backend/                        # Django REST API
│   ├── config/settings.py
│   └── api/                        # 모델, 뷰, 시리얼라이저
├── frontend/                       # React SPA
│   └── src/
│       ├── pages/                  # 실시간분석, 비교분석, 대시보드
│       └── api.js                  # Axios API 클라이언트
├── src/                            # ML/분석 핵심 모듈
│   ├── collection/                 # 뉴스 크롤러, ECOS, yfinance
│   ├── preprocessing/              # 텍스트 정제, 이벤트 매칭
│   ├── models/                     # 프레이밍 분류, 감성 분석, 편향 점수
│   └── analysis/                   # 이벤트 스터디, 그랜저, 매개분석, 패널회귀
├── models/                         # 학습된 모델 가중치
│   ├── framing/best/               # KLUE-RoBERTa (프레이밍)
│   └── sentiment/best/             # KcELECTRA (감성)
├── scripts/                        # 실행 스크립트
│   ├── train_framing.py            # 프레이밍 분류 모델 학습
│   ├── auto_label.py               # 전체 기사 자동 라벨링
│   ├── sentiment_score.py          # 감성 점수 산출
│   ├── compute_bias.py             # 편향 점수 산출
│   └── run_analysis.py             # Phase 3 통계 분석
├── data/                           # 데이터
│   ├── raw/                        # 원본 크롤링 CSV
│   ├── processed/                  # 전처리 완료 데이터셋
│   └── labeled/                    # 라벨링 데이터 (모델 학습용)
├── config/                         # 언론사 목록, 이벤트-섹터 매핑
├── notebooks/                      # 분석 노트북
├── docs/                           # 프로젝트 기획서
├── run_crawl.py                    # 크롤링 실행
├── build_dataset.py                # 데이터셋 구축 파이프라인
└── requirements.txt
```

## 설치 및 실행

### 환경 설정

```bash
# 저장소 클론
git clone <repo-url>
cd ver1

# Python 패키지 설치
pip install -r requirements.txt

# 환경변수 설정 (ECOS API 키 등)
cp .env.example .env
# .env 파일에 ECOS_API_KEY 입력
```

### Backend (Django)

```bash
cd backend
python manage.py migrate
python manage.py runserver
```

### Frontend (React)

```bash
cd frontend
npm install
npm start
```

### 데이터 수집

```bash
# 뉴스 크롤링 (Selenium)
python run_crawl.py

# 데이터셋 구축 (전처리 + 통합)
python build_dataset.py

# 주가 / 경제지표 수집 (기간 조정 가능)
python -m src.collection.stock_fetcher --start 2015-09-01 --end 2026-03-07
python -m src.collection.ecos_client   --start 201509     --end 202603
```

### 분석 파이프라인

```bash
python scripts/train_framing.py    # 프레이밍 분류 모델 학습
python scripts/auto_label.py       # 전체 기사 자동 라벨링
python scripts/sentiment_score.py  # 감성 점수 산출 (sentiment_score 컬럼)
python scripts/compute_bias.py     # 편향 점수 산출 (bias_score 컬럼)
python scripts/run_analysis.py     # Phase 3 통계 분석 4종
```

> `run_analysis.py`는 `sentiment_score`와 `bias_score` 컬럼을 요구한다.
> 위 순서대로 실행해야 한다.

## 개발 로드맵

- [x] **Phase 1** — 데이터 기반 구축 (뉴스 크롤러, ECOS API, 주가 수집, 6,381건 완료)
- [x] **Phase 2-1** — NLP 모델 학습 (KLUE-RoBERTa 프레이밍 분류, KcELECTRA 감성 분석)
- [ ] **Phase 2-2** — 라벨링 보강 및 모델 재학습 (defensive/neutral 클래스 개선)
- [ ] **Phase 3** — 주가 인과분석 (이벤트 스터디, 그랜저, 매개분석, 패널회귀)
- [ ] **Phase 4** — 웹 대시보드 (실시간 편향 분석, 언론사 비교, 연구 결과 시각화)
- [ ] **Phase 5** — 배포 (AWS EC2) & 논문화

## 분석 방법론

| 분석 | 목적 | 핵심 |
|------|------|------|
| **이벤트 스터디** | 경제 지표 발표 전후 주가 비정상 수익률 측정 | CAR(-1, +5), 추정기간 [-120, -11] |
| **그랜저 인과관계** | 편향 점수 → 주가 변동 시간적 선행 검증 | 시차 1~10일 |
| **매개 분석** | 편향 → CCSI → 주가 간접 경로 검증 | 직접/간접 효과 분리 |
| **패널 회귀** | 언론사 x 시간 이중 고정효과 통제 | 순수 편향 효과 추정 |
