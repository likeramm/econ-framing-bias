import { useState, useEffect } from 'react';
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';
import { getAnalysisResults } from '../api';
import { fmt, stars } from '../constants';
import './Results.css';

// 이벤트 스터디는 15개 이벤트를 동시에 검정하므로 개별 p값을 그대로 읽으면
// 다중비교 문제가 생긴다. Bonferroni 보정 임계값을 함께 보여준다.
const N_TESTS = 15;
const BONFERRONI = 0.05 / N_TESTS;

function Section({ title, note, children }) {
  return (
    <section className="result-panel">
      <h3>{title}</h3>
      {note && <p className="panel-note">{note}</p>}
      {children}
    </section>
  );
}

function EventStudy({ data }) {
  const rows = Object.entries(data || {})
    .filter(([, v]) => v && v.mean_car !== null && v.mean_car !== undefined)
    .map(([event, v]) => ({
      event,
      car: v.mean_car,
      p: v.p_value,
      n: v.n,
      model: v.model,
    }))
    .sort((a, b) => a.car - b.car);

  if (rows.length === 0) return <p className="empty">결과 없음</p>;

  return (
    <>
      <ResponsiveContainer width="100%" height={340}>
        <BarChart data={rows} margin={{ top: 8, right: 16, bottom: 60, left: 0 }}>
          <CartesianGrid strokeDasharray="3 3" vertical={false} />
          <XAxis dataKey="event" tick={{ fontSize: 11 }} interval={0} angle={-35} height={80} textAnchor="end" />
          <YAxis tick={{ fontSize: 12 }} />
          <Tooltip
            formatter={(v) => [fmt(v, 4), 'CAR']}
            labelFormatter={(l) => {
              const r = rows.find((x) => x.event === l);
              return r ? `${l} — p=${fmt(r.p, 4)}, n=${r.n}` : l;
            }}
          />
          <ReferenceLine y={0} stroke="#666" />
          <Bar dataKey="car" name="CAR">
            {rows.map((r) => (
              <Cell
                key={r.event}
                fill={r.p < BONFERRONI ? '#1a3d6d' : r.p < 0.05 ? '#7aa3cc' : '#cbd5e0'}
              />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>

      <table className="result-table">
        <thead>
          <tr>
            <th>이벤트</th>
            <th>모형</th>
            <th className="num">CAR</th>
            <th className="num">p</th>
            <th className="num">n</th>
            <th>보정 후</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((r) => (
            <tr key={r.event}>
              <td>{r.event}</td>
              <td className="dim">{r.model === 'mean_adjusted' ? '평균조정' : '시장모형'}</td>
              <td className="num">{fmt(r.car, 4)}</td>
              <td className="num">
                {fmt(r.p, 4)} {stars(r.p)}
              </td>
              <td className="num dim">{r.n}</td>
              <td>{r.p < BONFERRONI ? <span className="pass">유의</span> : <span className="dim">—</span>}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </>
  );
}

function Mediation({ data }) {
  if (!data || data.total_effect_c === undefined) return <p className="empty">결과 없음</p>;
  const paths = [
    ['총효과 (c)', data.total_effect_c, data.total_effect_p],
    ['경로 a (편향→CCSI)', data.path_a, data.path_a_p],
    ['경로 b (CCSI→주가)', data.path_b, data.path_b_p],
    ["직접효과 (c')", data.direct_effect, data.direct_effect_p],
    ['간접효과 (a×b)', data.indirect_effect, data.sobel_p],
  ];
  return (
    <table className="result-table">
      <thead>
        <tr>
          <th>경로</th>
          <th className="num">계수</th>
          <th className="num">p</th>
        </tr>
      </thead>
      <tbody>
        {paths.map(([label, coef, p]) => (
          <tr key={label}>
            <td>{label}</td>
            <td className="num">{fmt(coef, 6)}</td>
            <td className={`num ${p < 0.05 ? 'pass' : ''}`}>
              {fmt(p, 4)} {stars(p)}
            </td>
          </tr>
        ))}
        <tr className="verdict">
          <td>매개 성립</td>
          <td colSpan={2}>{data.significant_mediation ? '예' : '아니오'}</td>
        </tr>
      </tbody>
    </table>
  );
}

function Granger({ data }) {
  const fwd = data?.bias_to_stock;
  if (!fwd?.lag_results) return <p className="empty">결과 없음</p>;
  return (
    <table className="result-table">
      <thead>
        <tr>
          <th>시차</th>
          <th className="num">F</th>
          <th className="num">p</th>
        </tr>
      </thead>
      <tbody>
        {Object.entries(fwd.lag_results).map(([lag, r]) => (
          <tr key={lag}>
            <td>lag {lag}</td>
            <td className="num">{fmt(r.f_stat, 3)}</td>
            <td className={`num ${r.p_value < 0.05 ? 'pass' : ''}`}>
              {fmt(r.p_value, 4)} {stars(r.p_value)}
            </td>
          </tr>
        ))}
        <tr className="verdict">
          <td>인과 유의</td>
          <td colSpan={2}>{fwd.significant ? '예' : '아니오'}</td>
        </tr>
      </tbody>
    </table>
  );
}

function Panel({ data }) {
  if (!data?.coefficients) return <p className="empty">결과 없음</p>;
  return (
    <>
      <p className="panel-stat">
        관측치 {data.n_obs}개 · 섹터 {data.n_entities}개 · R²(within) {fmt(data.r2_within, 4)} ·
        F {fmt(data.f_stat, 3)} (p={fmt(data.f_pvalue, 4)})
      </p>
      <table className="result-table">
        <thead>
          <tr>
            <th>변수</th>
            <th className="num">β</th>
            <th className="num">t</th>
            <th className="num">p</th>
          </tr>
        </thead>
        <tbody>
          {Object.entries(data.coefficients).map(([name, c]) => (
            <tr key={name}>
              <td>{name}</td>
              <td className="num">{fmt(c.coefficient, 6)}</td>
              <td className="num">{fmt(c.t_stat, 3)}</td>
              <td className={`num ${c.p_value < 0.05 ? 'pass' : ''}`}>
                {fmt(c.p_value, 4)} {stars(c.p_value)}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </>
  );
}

function Results() {
  const [data, setData] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    getAnalysisResults()
      .then(setData)
      .catch((e) =>
        setError(e.response?.data?.hint || e.message)
      );
  }, []);

  if (error) return <div className="empty-state">분석 결과를 불러오지 못했습니다: {error}</div>;
  if (!data) return <div className="empty-state">불러오는 중…</div>;

  return (
    <div className="results">
      <h2>연구 결과</h2>

      <Section
        title="1. 이벤트 스터디 — 누적 비정상 수익률(CAR)"
        note={`대상이 벤치마크(KOSPI) 자신인 이벤트는 평균조정 모형으로 계산했다. 15개를 동시 검정하므로 Bonferroni 보정 임계값 p < ${BONFERRONI.toFixed(4)}를 함께 표시한다. 진한 막대만 보정 후에도 유의하다.`}
      >
        <EventStudy data={data.event_study} />
      </Section>

      <Section
        title="2. 그랜저 인과관계 — 편향이 주가에 선행하는가"
        note="일별 편향 점수와 KOSPI 수익률. 귀무가설은 '편향이 주가를 예측하지 못한다'이다."
      >
        <Granger data={data.granger} />
      </Section>

      <Section
        title="3. 매개분석 — 편향 → CCSI → 주가"
        note="월별 데이터. 매개가 성립하려면 경로 a와 b가 모두 유의해야 한다."
      >
        <Mediation data={data.mediation} />
      </Section>

      <Section
        title="4. 패널 회귀 — 섹터 × 시간 고정효과"
        note="분석 단위가 언론사이면 종속변수가 같은 달의 모든 언론사에 대해 동일해져 시간 고정효과가 이를 모두 흡수한다. 섹터 단위로 바꿔 식별 문제를 해소했다."
      >
        <Panel data={data.panel} />
      </Section>
    </div>
  );
}

export default Results;
