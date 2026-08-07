import { useState, useEffect } from 'react';
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  Legend,
  Line,
  LineChart,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';
import { getStats, getBiasSummary, getFramingDistribution, getBiasTimeseries } from '../api';
import { CATEGORY_LABEL, FRAMING, FRAMING_ORDER, fmt, fmtInt } from '../constants';
import './Dashboard.css';

function StatCard({ label, value, sub }) {
  return (
    <div className="card">
      <h3>{label}</h3>
      <p className="card-value">{value}</p>
      {sub && <p className="card-sub">{sub}</p>}
    </div>
  );
}

function Dashboard() {
  const [stats, setStats] = useState(null);
  const [bias, setBias] = useState([]);
  const [framing, setFraming] = useState([]);
  const [series, setSeries] = useState([]);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    Promise.all([
      getStats(),
      getBiasSummary(true),
      getFramingDistribution(),
      getBiasTimeseries(),
    ])
      .then(([s, b, f, t]) => {
        setStats(s);
        setBias(b);
        setFraming(f);
        setSeries(t);
      })
      .catch((e) => setError(e.message))
      .finally(() => setLoading(false));
  }, []);

  if (loading) return <div className="state">불러오는 중…</div>;
  if (error) {
    return (
      <div className="state error">
        <p>API에 연결하지 못했습니다: {error}</p>
        <p className="hint">
          백엔드를 실행하세요: <code>cd backend &amp;&amp; python manage.py runserver</code>
        </p>
      </div>
    );
  }

  const period =
    stats.period_start && stats.period_end
      ? `${stats.period_start.slice(0, 7)} ~ ${stats.period_end.slice(0, 7)}`
      : '—';

  const framingData = [...framing].sort(
    (a, b) => FRAMING_ORDER.indexOf(a.framing_type) - FRAMING_ORDER.indexOf(b.framing_type)
  );

  return (
    <div className="dashboard">
      <h2>대시보드</h2>

      <div className="card-grid">
        <StatCard label="수집 기사" value={fmtInt(stats.article_count)} sub={period} />
        <StatCard
          label="분석 완료"
          value={fmtInt(stats.analyzed_count)}
          sub={`평균 편향 ${fmt(stats.avg_bias)}`}
        />
        <StatCard
          label="언론사"
          value={fmtInt(stats.media_count)}
          sub={`연구 대상 ${stats.core_media_count}개사`}
        />
        <StatCard label="경제 지표 발표" value={fmtInt(stats.event_count)} sub="ECOS 7종" />
      </div>

      <section className="panel">
        <h3>언론사별 평균 편향 점수</h3>
        <p className="panel-note">
          연구 대상 10개사. 음수는 부정적 프레이밍, 양수는 긍정적 프레이밍으로 기운다는 뜻이다.
        </p>
        <ResponsiveContainer width="100%" height={320}>
          <BarChart data={bias} margin={{ top: 8, right: 16, bottom: 8, left: 0 }}>
            <CartesianGrid strokeDasharray="3 3" vertical={false} />
            <XAxis dataKey="media" tick={{ fontSize: 12 }} interval={0} angle={-25} height={60} textAnchor="end" />
            <YAxis tick={{ fontSize: 12 }} />
            <Tooltip
              formatter={(v, name) => [fmt(v), name === 'avg_bias' ? '평균 편향' : name]}
              labelFormatter={(l) => {
                const row = bias.find((r) => r.media === l);
                return row ? `${l} (${CATEGORY_LABEL[row.category] || row.category}, n=${fmtInt(row.count)})` : l;
              }}
            />
            <ReferenceLine y={0} stroke="#666" />
            <Bar dataKey="avg_bias" name="평균 편향">
              {bias.map((r) => (
                <Cell key={r.media} fill={r.avg_bias < 0 ? '#c0453b' : '#1a7f5a'} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </section>

      <section className="panel">
        <h3>프레이밍 유형 분포</h3>
        <ResponsiveContainer width="100%" height={300}>
          <BarChart data={framingData} margin={{ top: 8, right: 16, bottom: 8, left: 0 }}>
            <CartesianGrid strokeDasharray="3 3" vertical={false} />
            <XAxis dataKey="label" tick={{ fontSize: 12 }} />
            <YAxis tick={{ fontSize: 12 }} />
            <Tooltip
              formatter={(v, n, p) => [
                `${fmtInt(v)}건 (${(p.payload.share * 100).toFixed(1)}%)`,
                '기사 수',
              ]}
              labelFormatter={(l) => {
                const row = framingData.find((r) => r.label === l);
                return row ? `${l} — 평균 편향 ${fmt(row.avg_bias)}` : l;
              }}
            />
            <Bar dataKey="count" name="기사 수">
              {framingData.map((r) => (
                <Cell key={r.framing_type} fill={FRAMING[r.framing_type]?.color || '#888'} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </section>

      <section className="panel">
        <h3>월별 편향 추이</h3>
        <p className="panel-note">연구 대상 10개사 전체 평균. 편향 점수와 감성 점수를 함께 표시한다.</p>
        <ResponsiveContainer width="100%" height={320}>
          <LineChart data={series} margin={{ top: 8, right: 16, bottom: 8, left: 0 }}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="month" tick={{ fontSize: 11 }} minTickGap={40} />
            <YAxis tick={{ fontSize: 12 }} />
            <Tooltip formatter={(v) => fmt(v)} />
            <Legend />
            <ReferenceLine y={0} stroke="#666" />
            <Line
              type="monotone"
              dataKey="avg_bias"
              name="편향 점수"
              stroke="#2c5f8d"
              dot={false}
              strokeWidth={2}
            />
            <Line
              type="monotone"
              dataKey="avg_sentiment"
              name="감성 점수"
              stroke="#c0453b"
              dot={false}
              strokeWidth={1}
              strokeDasharray="4 3"
            />
          </LineChart>
        </ResponsiveContainer>
      </section>
    </div>
  );
}

export default Dashboard;
