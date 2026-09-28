# 경제 뉴스 편향 탐지 및 주가 영향 분석 시스템

> 순천향대학교 컴퓨터소프트웨어공학과 졸업작품 · 조아람(20233506)

한국은행이 "GDP 성장률 2.3%"를 발표하면, 어떤 언론은 **"견조한 성장세 지속"**, 다른 언론은 **"성장 둔화 우려 확대"** 로 씁니다. 숫자는 하나인데 프레임은 갈립니다.

이 프로젝트는 그 차이를 **정량화**하고, 그것이 시장 반응과 어떻게 연결되는지 검증합니다.

---

## 핵심 아이디어

사건을 고정하고 매체만 비교합니다. 그래야 "사건 때문"이 아니라 "매체 때문"인 부분을 분리할 수 있습니다.

```
2022-07-13  기준금리 0.5%p 인상  ── 12개 매체가 동시 보도

  긍정 ●●                    한국경제 · 매일경제
  중립 ●●●●●●                연합뉴스 · SBS Biz …
  부정 ●●●●                  한겨레 · 경향신문 …
         ↑
  사건은 하나인데 프레임이 갈린다 ← 측정 대상
```

수집된 데이터에서 **(이벤트 × 주) 조합 1,834개 중 935개(51%)** 를 3개 이상 매체가 동시 보도했습니다. 한 사건을 최대 14개 매체가 함께 다뤘습니다. 이 구조가 매체 간 비교를 가능하게 합니다.

## 연구 질문

| # | 질문 | 상태 |
|---|------|------|
| **RQ1** | 동일 경제 사건에 대해 매체별 프레이밍은 어떻게 다른가 | 모델 학습 완료, 분석 대기 |
| **RQ2** | 매체별 편향은 시간에 따라 일관된 패턴을 보이는가 | 미착수 |
| **RQ3** | 매체 간 프레이밍 분산이 큰 날, 시장 반응(거래량·변동성)은 어떠한가 | 코드 구현 완료, 미실행 |
| **RQ4** | 프레이밍이 소비자심리지수(CCSI)를 매개해 주가에 영향을 미치는가 | 코드 구현 완료, 미실행 |

## 분류 체계 — 3분류

| 라벨 | 정의 | 예시 |
|------|------|------|
| **긍정** | 성장·회복·개선·안도의 틀. 우려를 언급해도 완화·방어로 마무리 | "우려에도 펀더멘털 견고, 선방" |
| **중립** | 기자 해석이 최소화된 사실·수치 전달. 또는 양측 병렬 제시로 평가 유보 | "2분기 GDP 2.3% 기록" |
| **부정** | 둔화·하락·우려·위기의 틀 | "성장 둔화 우려 확대, 불확실성 고조" |

<details>
<summary><b>6분류에서 3분류로 바꾼 이유</b></summary>

초기에는 낙관·비관·중립·방어·비교·경고 6분류로 설계했고 실제로 모델까지 학습했습니다(Macro F1 0.74). 그러나 클래스별 성능을 보면 경계가 무너져 있었습니다.

| 클래스 | precision | recall | f1 | 학습셋 비중 |
|--------|-----------|--------|-----|-----------|
| neutral | 0.78 | 0.82 | 0.80 | 35.8% |
| comparative | 0.88 | 0.72 | 0.79 | 6.5% |
| alarmist | 0.78 | 0.73 | 0.76 | 23.1% |
| optimistic | 0.87 | 0.66 | 0.75 | 13.8% |
| pessimistic | 0.66 | 0.76 | 0.71 | 15.8% |
| **defensive** | **0.53** | 0.71 | **0.61** | **4.8%** |

`defensive`는 F1 0.61에 학습 데이터도 4.8%뿐이었습니다. 3분류로 통합하면 긍정 18.7% / 중립 42.4% / 부정 39.0%로 불균형이 크게 완화됩니다.

- 긍정 ← optimistic + defensive
- 중립 ← neutral + comparative
- 부정 ← pessimistic + alarmist

</details>

## 데이터 현황 (실측)

| 데이터 | 규모 | 기간 | 출처 |
|--------|------|------|------|
| 뉴스 기사 | **14,135건** · 38개 매체 · 15개 이벤트 유형 | 2016-03 ~ 2026-03 (10년) | 네이버뉴스 |
| 주가·지수 | **34,666행** · 14종목 (KOSPI/KOSDAQ, 섹터 ETF, 개별주) | 2016-01 ~ 2026-03 | pykrx |
| 경제 지표 | **984건** · 6종 (기준금리, CPI, CCSI, 경상수지, 환율, 생산지수) | 2016-01 ~ 2026-03 | 한국은행 ECOS |
| 수동 라벨 | 3,262건 | — | 직접 라벨링 |
| LLM 라벨 | 13,268건 | — | gpt-5.5 (Batch API) |

뉴스 기사의 **98%** 가 주가 데이터 기간과 겹칩니다.

### 매체 성향 분포

| 그룹 | 건수 | 비율 |
|------|------|------|
| 경제지 | 6,895 | 48.8% |
| 통신사·방송 | 4,675 | 33.1% |
| 보수 | 1,422 | 10.1% |
| 진보 | 1,071 | 7.6% |
| 기타 | 72 | 0.5% |

주요 매체: 연합뉴스(3,787) · 한국경제(1,898) · 서울경제(1,624) · 매일경제(1,370) · SBS Biz(1,042) · 한국경제TV(714) · 경향신문(602) · 조선일보(476) · 동아일보(474) · 한겨레

## 모델

| 역할 | 모델 | 비고 |
|------|------|------|
| 프레이밍 분류 | **`klue/roberta-large`** (약 337M) | Park et al. (2021), arXiv:2105.09680 |
| 감성 분석 | **`beomi/KcELECTRA-base-v2022`** | Lee et al. (2022) · Clark et al. (2020) |
| 키워드 극성 | 경제 도메인 극성 사전 198개 | KNU 감성사전 + 한국은행 용어집 기반 |

모델 선정 과정에서 `snunlp/KR-FinBert-SC`(110M, 금융 특화)를 먼저 시도했으나 Macro F1 0.63에 그쳤고, 범용 대형 모델인 `klue/roberta-large`에 도메인 파인튜닝을 적용한 쪽이 **0.74**로 더 나았습니다.

### 편향 점수

```
Bias = α·F + β·S + γ·K   (α=0.40, β=0.35, γ=0.25)

  F  프레이밍 강도   KLUE-RoBERTa
  S  감성 점수      KcELECTRA        [-1, +1]
  K  키워드 극성    극성 사전         [-1, +1]
```

가중치는 현재 잠정값이며 ① Gold Set 회귀 학습 ② 선행 연구 사전 비율 참조 ③ 민감도 분석의 3중 검증으로 확정할 예정입니다. 상세: [`docs/Bias_Score_가중치_학술근거.md`](docs/Bias_Score_가중치_학술근거.md)

## 파이프라인과 실행 상태

```
[1] 뉴스 수집          ✅ 실행 완료   14,135건
[2] 전처리·이벤트 매칭  ✅ 실행 완료
[3] LLM 라벨링         ✅ 실행 완료   13,268건 (gpt-5.5 Batch API, 3분류)
[4] 프레이밍 모델 학습  ✅ 실행 완료   Test Macro F1 0.84 (LLM 라벨 증류, 3분류)
[5] 감성 점수 산출      ✅ 실행 완료   KcELECTRA
[6] Bias Score 산출    ✅ 실행 완료   data/labeled/bias_scored.csv
[7] 통계 분석 4종      ✅ 실행 완료   data/analysis_results/analysis_results.json
[8] 웹 대시보드        🟡 일부 구현   기사 탐색 · 실시간 분류
```

## 평가자 간 신뢰도(IAA) 측정 — 진행 중

단일 라벨러의 주관성을 검증하기 위해 평가자 2인이 **각자 독립적으로** 150건을 라벨링하고 Cohen's κ를 산출합니다.

**라벨링 도구**: https://likeramm.github.io/econ-framing-bias/

```
평가자 A (조아람) 150건 ─┐
                        ├→ Cohen's κ  (목표 ≥ 0.6)
평가자 B (박세준) 150건 ─┘
         ↓
   불일치 건 조정(adjudication) → Gold Set 150건 확정
         ↓
   Gold Set ↔ LLM 자동 라벨 대조 → LLM 타당성 검증
```

가이드라인은 파일럿 검증을 거쳐 3차까지 개정됐습니다. 판정 규칙은 번호가 낮을수록 우선합니다.

| 규칙 | 내용 |
|------|------|
| **R0** | 사건이 아니라 **서술**을 본다. 평가어가 없으면 중립 |
| **R0-a** | 강도어(`급등`,`최고치`,`돌파`)는 **방향이 없다**. 금리·물가·부채면 부정, 주가·수출·고용이면 긍정 |
| **R0-b** | 큰따옴표 안은 취재원의 말. 단 제목으로 뽑았다면 R1 |
| **R1** | 제목과 본문이 충돌하면 제목 |
| **R2** | 긍·부 병기되고 결론 없으면 중립 |
| **R3** | 기자가 한쪽으로 결론냈으면 그 방향 |
| **R4** | 그래도 애매하면 중립 + 표시 |

상세: [`docs/라벨링_가이드라인_v3.md`](docs/라벨링_가이드라인_v3.md)

## 분석 방법론

| 분석 | 목적 | 근거 논문 |
|------|------|----------|
| **이벤트 스터디** | 사건 전후 비정상수익률(AR·CAR) 측정 | MacKinlay (1997), *JEL* |
| **그랜저 인과성** | Bias Score → 주가·거래량 선후 관계 검정 | Granger (1969), *Econometrica* |
| **매개 분석** | 편향 → 투자심리(CCSI) → 주가 경로 검증 | Baron & Kenny (1986), *JPSP* |
| **패널 고정효과 회귀** | 매체·시간 고정효과 통제 후 순수 편향 효과 추정 | — |

## 프로젝트 구조

```
.
├── src/
│   ├── collection/          뉴스 크롤러 · ECOS · pykrx
│   ├── preprocessing/       텍스트 정제 · 이벤트 매칭
│   ├── models/              프레이밍 분류 · 감성 분석 · 편향 점수
│   └── analysis/            이벤트 스터디 · 그랜저 · 매개 · 패널회귀
├── scripts/
│   ├── train_framing.py     KLUE-RoBERTa 학습
│   ├── auto_label.py        self-training 자동 라벨링
│   ├── llm_label.py         LLM 자동 라벨링 (재개·백오프 지원)
│   ├── sentiment_score.py   KcELECTRA 추론
│   ├── compute_bias.py      Bias Score + 극성 사전
│   ├── build_goldset.py     IAA 표본 추출
│   └── run_analysis.py      통계 분석 4종 실행
├── models/
│   ├── framing/best/        학습 완료 (3분류, LLM 라벨 증류)
│   └── sentiment/best/      학습 완료
├── data/
│   ├── processed/           dataset.csv · stock_data.csv · economic_indicators.csv
│   ├── goldset/             IAA 표본 · 파일럿 결과
│   └── _archive_6class/     6분류 시기 라벨 원본
├── backend/                 Django REST API (골격)
├── frontend/                React SPA (골격)
├── docs/
│   ├── index.html           라벨링 사이트 (GitHub Pages 배포)
│   ├── 00_마스터_아이디어_정리.md
│   ├── 라벨링_가이드라인_v3.md
│   └── Bias_Score_가중치_학술근거.md
└── config/                  매체 목록 · 이벤트-섹터 매핑
```

## 설치 및 실행

```bash
pip install -r requirements.txt
cp .env.example .env      # ECOS_API_KEY 입력
```

LLM 라벨링을 쓰려면 `.env`에 `OPENAI_API_KEY`를 추가합니다.

```bash
# 데이터 수집
python run_crawl.py                  # 뉴스 크롤링
python src/collection/ecos_client.py # 경제지표
python build_dataset.py              # 전처리·통합

# 라벨링·모델
python scripts/llm_label.py --dry-run     # LLM 라벨링 비용 추정
python scripts/llm_label.py --batch auto  # Batch API 로 전체 라벨링
python scripts/train_framing.py           # LLM 라벨로 프레이밍 모델 학습

# 분석 (순서대로)
python scripts/sentiment_score.py
python scripts/compute_bias.py
python scripts/run_analysis.py
```

### 대시보드

```bash
# 백엔드 (Django) — 최초 1회 DB 생성·데이터 적재
cd backend
python manage.py migrate
python manage.py load_articles       # bias_scored.csv 등 → DB
python manage.py runserver 8001

# 프론트엔드 (React) — 다른 터미널에서
cd frontend
npm install
REACT_APP_API_URL=http://localhost:8001/api npm start   # http://localhost:3000
```

실시간 분류는 학습된 가중치(`models/framing/best/model.safetensors`, git 미포함)가 있어야 동작합니다.

## 남은 작업

- [x] 3분류 전환 — LLM 라벨링, 모델 학습, `FRAMING_SCORES`
- [ ] IAA 측정 완료 (평가자 2인 × 150건) → Gold Set 확정
- [ ] Bias Score 가중치 3중 검증
- [x] 파이프라인 감성 점수 → Bias Score → 통계 분석 실행
- [ ] 매체 간 프레이밍 비교 분석 (RQ1·RQ2)
- [ ] 웹 대시보드 — 개요·매체 비교·시계열 화면 추가
- [ ] 최종 보고서

## 참고문헌

- Park, S. et al. (2021). KLUE: Korean Language Understanding Evaluation. *NeurIPS Datasets and Benchmarks*. arXiv:2105.09680
- Liu, Y. et al. (2019). RoBERTa: A Robustly Optimized BERT Pretraining Approach. arXiv:1907.11692
- Clark, K. et al. (2020). ELECTRA: Pre-training Text Encoders as Discriminators Rather Than Generators. *ICLR*
- Card, D. et al. (2015). The Media Frames Corpus: Annotations of Frames Across Issues. *ACL*
- Spinde, T. et al. (2022). Neural Media Bias Detection Using Distant Supervision With BABE. *EMNLP Findings*
- Gilardi, F. et al. (2023). ChatGPT outperforms crowd workers for text-annotation tasks. *PNAS*, 120(30)
- MacKinlay, A. C. (1997). Event Studies in Economics and Finance. *JEL*, 35(1)
- Granger, C. W. J. (1969). Investigating Causal Relations by Econometric Models and Cross-spectral Methods. *Econometrica*, 37(3)
- Baron, R. M. & Kenny, D. A. (1986). The moderator-mediator variable distinction. *JPSP*, 51(6)
- Landis, J. R. & Koch, G. G. (1977). The measurement of observer agreement for categorical data. *Biometrics*, 33(1)
