// 프레이밍 라벨 표시용 공통 정의
export const LABELS = {
  positive: { ko: '긍정', className: 'label-positive' },
  neutral: { ko: '중립', className: 'label-neutral' },
  negative: { ko: '부정', className: 'label-negative' },
};

export const LABEL_ORDER = ['positive', 'neutral', 'negative'];

export const eventName = (eventType) => eventType?.replaceAll('_', ' ') ?? '';

export function LabelBadge({ label }) {
  const info = LABELS[label];
  if (!info) return null;
  return <span className={`label-badge ${info.className}`}>{info.ko}</span>;
}

// 편향 점수(-3 ~ +3)를 가운데 0 기준 막대로 표시
export function BiasMeter({ value }) {
  if (value == null) return null;
  const pct = Math.min(Math.abs(value) / 3, 1) * 50;
  const side = value >= 0 ? 'pos' : 'neg';
  return (
    <div className="bias-meter" title={`편향 점수 ${value.toFixed(2)} (범위 -3 ~ +3)`}>
      <div className="bias-track">
        <div className={`bias-fill ${side}`} style={{ width: `${pct}%` }} />
        <div className="bias-zero" />
      </div>
      <span className="bias-value">{value > 0 ? '+' : ''}{value.toFixed(2)}</span>
    </div>
  );
}
