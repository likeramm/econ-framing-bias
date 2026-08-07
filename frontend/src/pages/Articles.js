import { useState, useEffect, useCallback } from 'react';
import { getArticles } from '../api';
import { FRAMING, FRAMING_ORDER, fmt, fmtInt } from '../constants';
import './Articles.css';

const CORE_MEDIA = [
  '조선일보', '중앙일보', '동아일보',
  '한겨레', '경향신문',
  '한국경제', '매일경제', '서울경제',
  '연합뉴스', 'SBS',
];

const ORDERINGS = [
  { value: '-published_at', label: '최신순' },
  { value: 'published_at', label: '오래된순' },
  { value: 'framing__bias_score', label: '편향 낮은순 (부정)' },
  { value: '-framing__bias_score', label: '편향 높은순 (긍정)' },
];

function Articles() {
  const [data, setData] = useState({ results: [], count: 0 });
  const [filters, setFilters] = useState({
    media: '',
    framing: '',
    search: '',
    ordering: '-published_at',
  });
  const [page, setPage] = useState(1);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  const load = useCallback(() => {
    setLoading(true);
    const params = { core_only: true, page, ordering: filters.ordering };
    if (filters.media) params.media = filters.media;
    if (filters.framing) params.framing = filters.framing;
    if (filters.search) params.search = filters.search;

    getArticles(params)
      .then(setData)
      .catch((e) => setError(e.message))
      .finally(() => setLoading(false));
  }, [filters, page]);

  useEffect(load, [load]);

  const update = (key) => (e) => {
    setFilters((f) => ({ ...f, [key]: e.target.value }));
    setPage(1);
  };

  const pageSize = 20;
  const totalPages = Math.max(1, Math.ceil(data.count / pageSize));

  return (
    <div className="articles">
      <h2>기사 분석</h2>

      <div className="filters">
        <select value={filters.media} onChange={update('media')}>
          <option value="">전체 언론사</option>
          {CORE_MEDIA.map((m) => (
            <option key={m} value={m}>{m}</option>
          ))}
        </select>

        <select value={filters.framing} onChange={update('framing')}>
          <option value="">전체 프레이밍</option>
          {FRAMING_ORDER.map((k) => (
            <option key={k} value={k}>{FRAMING[k].label}</option>
          ))}
        </select>

        <select value={filters.ordering} onChange={update('ordering')}>
          {ORDERINGS.map((o) => (
            <option key={o.value} value={o.value}>{o.label}</option>
          ))}
        </select>

        <input
          type="search"
          placeholder="제목 검색"
          value={filters.search}
          onChange={update('search')}
        />
      </div>

      <p className="result-count">{fmtInt(data.count)}건</p>

      {error && <div className="empty-state">API 오류: {error}</div>}

      {!error && loading && <div className="empty-state">불러오는 중…</div>}

      {!error && !loading && data.results.length === 0 && (
        <div className="empty-state">
          <p>조건에 맞는 기사가 없습니다.</p>
        </div>
      )}

      {!error && !loading && data.results.length > 0 && (
        <>
          <div className="article-list">
            {data.results.map((article) => {
              const f = article.framing;
              const meta = f ? FRAMING[f.framing_type] : null;
              return (
                <div key={article.id} className="article-card">
                  <div className="article-meta">
                    <span className="media-name">{article.media?.name}</span>
                    <span className="date">
                      {new Date(article.published_at).toLocaleDateString('ko-KR')}
                    </span>
                    {article.event_type && (
                      <span className="event-tag">{article.event_type}</span>
                    )}
                  </div>
                  <h3>
                    <a href={article.url} target="_blank" rel="noreferrer">
                      {article.title}
                    </a>
                  </h3>
                  {f && (
                    <div className="framing-badge">
                      <span
                        className="badge"
                        style={{ background: meta?.color || '#888' }}
                      >
                        {meta?.label || f.framing_type}
                      </span>
                      <span>편향 {fmt(f.bias_score, 2)}</span>
                      <span>감성 {fmt(f.sentiment_score, 2)}</span>
                      <span className="confidence">신뢰도 {fmt(f.confidence, 2)}</span>
                    </div>
                  )}
                </div>
              );
            })}
          </div>

          <div className="pagination">
            <button disabled={page <= 1} onClick={() => setPage((p) => p - 1)}>
              이전
            </button>
            <span>
              {page} / {fmtInt(totalPages)}
            </span>
            <button disabled={page >= totalPages} onClick={() => setPage((p) => p + 1)}>
              다음
            </button>
          </div>
        </>
      )}
    </div>
  );
}

export default Articles;
