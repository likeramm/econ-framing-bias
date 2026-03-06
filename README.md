# 경제 뉴스 프레이밍 편향 탐지기 + 주가 영향 분석

> Detecting framing bias in Korean economic news and analyzing its impact on stock prices

동일한 경제 지표 발표에 대해 언론사별 프레이밍 차이를 분석하고, 그 편향이 실제 주가에 미치는 영향을 검증하는 프로젝트입니다.

## 연구 질문

- **RQ1:** 동일 경제 지표에 대해 한국 언론사들의 프레이밍은 어떻게 다른가?
- **RQ2:** 언론사별 프레이밍 편향 점수는 시간에 따라 일관된 패턴을 보이는가?
- **RQ3:** 부정적 프레이밍의 집중도가 높을수록 관련 섹터 주가 하락 폭이 큰가?
- **RQ4:** 미디어 프레이밍이 소비심리지수(CCSI) 변화를 매개하여 주가에 영향을 미치는가?

## 기술 스택

| 영역 | 기술 |
|------|------|
| Frontend | React |
| Backend | Django REST Framework |
| 데이터 수집 | BeautifulSoup, ECOS API, pykrx |
| 프레이밍 분류 | KLUE-RoBERTa-large (Fine-tuned) |
| 감성 분석 | KcELECTRA-base |
| 통계 분석 | statsmodels, linearmodels, pingouin |
| 배포 | AWS EC2 |

## 프레이밍 유형 (6가지)

| 유형 | 예시 |
|------|------|
| 낙관적 (Optimistic) | "견조한 성장세 지속" |
| 비관적 (Pessimistic) | "성장 둔화 우려 확대" |
| 경고적 (Alarmist) | "경제 위기 신호" |
| 방어적 (Defensive) | "우려에도 불구하고..." |
| 비교적 (Comparative) | "미국 대비 양호한 수준" |
| 중립적 (Neutral) | "2분기 GDP 2.3% 기록" |

## 프로젝트 구조

```
econ-framing-bias/
├── backend/                     # Django REST API
│   ├── config/                  # Django 설정
│   └── api/                     # API 앱 (모델, 뷰, 시리얼라이저)
├── frontend/                    # React 앱
│   └── src/
│       ├── pages/               # 페이지 컴포넌트
│       └── api.js               # API 클라이언트
├── src/                         # ML/분석 모듈
│   ├── collection/              # 데이터 수집 (크롤러, API)
│   ├── preprocessing/           # 텍스트 전처리
│   ├── models/                  # NLP 모델
│   └── analysis/                # 통계 분석
├── config/                      # 설정 파일
├── data/                        # 데이터 (Git 미포함)
├── notebooks/                   # 분석 노트북
└── tests/
```

## 설치 및 실행

### Backend (Django)

```bash
# 패키지 설치
pip install -r requirements.txt

# 환경변수 설정
cp .env.example .env

# DB 마이그레이션 & 서버 실행
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

## 개발 로드맵

- [ ] **Phase 1** — 데이터 기반 구축 (뉴스 크롤러, ECOS API, 주가 수집)
- [ ] **Phase 2** — 프레이밍 분류 모델 (KLUE-RoBERTa Fine-tuning, F1 >= 0.80)
- [ ] **Phase 3** — 주가 상관분석 (이벤트 스터디, 그랜저 인과관계, 매개 분석)
- [ ] **Phase 4** — 시각화 & 대시보드 (React + Django)
- [ ] **Phase 5** — 배포 (AWS EC2) & 논문화
