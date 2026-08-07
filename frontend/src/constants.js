// 프레이밍 유형 표시명과 색상.
// 낙관 → 중립 → 경고 순서로 발산형 색을 배정해, 차트에서 편향 방향이
// 색으로 읽히도록 한다.
export const FRAMING = {
  optimistic: { label: '낙관적', color: '#1a7f5a' },
  defensive: { label: '방어적', color: '#5aa77f' },
  comparative: { label: '비교적', color: '#8f9aa8' },
  neutral: { label: '중립적', color: '#b0b7c0' },
  pessimistic: { label: '비관적', color: '#d98a4a' },
  alarmist: { label: '경고적', color: '#c0453b' },
};

export const FRAMING_ORDER = [
  'optimistic',
  'defensive',
  'comparative',
  'neutral',
  'pessimistic',
  'alarmist',
];

export const CATEGORY_LABEL = {
  conservative: '보수',
  progressive: '진보',
  economic: '경제지',
  broadcast: '방송',
  wire: '통신사',
  neutral: '기타',
};

export const fmt = (n, digits = 3) =>
  n === null || n === undefined || Number.isNaN(n) ? '—' : Number(n).toFixed(digits);

export const fmtInt = (n) =>
  n === null || n === undefined ? '—' : Number(n).toLocaleString('ko-KR');

// p값에 유의성 별표를 붙인다
export const stars = (p) => {
  if (p === null || p === undefined) return '';
  if (p < 0.01) return '***';
  if (p < 0.05) return '**';
  if (p < 0.1) return '*';
  return '';
};
