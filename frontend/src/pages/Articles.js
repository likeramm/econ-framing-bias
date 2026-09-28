import { useCallback, useEffect, useState } from 'react';
import { useSearchParams } from 'react-router-dom';
import api from '../api';
import { BiasMeter, LABELS, LABEL_ORDER, LabelBadge, eventName } from '../labels';
import './Articles.css';

const PAGE_SIZE = 20; // backend REST_FRAMEWORK PAGE_SIZE 와 동일
const FILTER_KEYS = ['search', 'media', 'group', 'event_type', 'label', 'date_from', 'date_to', 'ordering'];

const ORDERING_OPTIONS = [
  { value: '-date', label: '최신순' },
  { value: 'date', label: '오래된순' },
  { value: '-bias', label: '편향 점수 높은순 (긍정)' },
  { value: 'bias', label: '편향 점수 낮은순 (부정)' },
  { value: 'confidence', label: 'LLM 확신도 낮은순 (검수용)' },
];

function Articles() {
  const [params, setParams] = useSearchParams();
  const [options, setOptions] = useState(null);
  const [data, setData] = useState(null);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(false);
  const [selected, setSelected] = useState(null);
  const [searchText, setSearchText] = useState(params.get('search') || '');

  const page = Number(params.get('page') || 1);
  const closeDetail = useCallback(() => setSelected(null), []);

  // 실행 시점의 실제 URL을 기준으로 갱신한다. 디바운스된 검색어 반영은 입력 당시 렌더의
  // 클로저에서 실행되므로, 렌더 상태(params)를 쓰면 그 사이 고른 필터를 덮어쓰게 된다.
  const updateParams = (changes) => {
    const next = new URLSearchParams(window.location.search);
    Object.entries(changes).forEach(([k, v]) => (v ? next.set(k, v) : next.delete(k)));
    if (!('page' in changes)) next.delete('page'); // 필터가 바뀌면 1페이지로
    setParams(next);
  };

  // 필터 선택지 (한 번만)
  useEffect(() => {
    api.get('/filters/')
      .then((res) => setOptions(res.data))
      .catch(() => setError('API 서버에 연결할 수 없습니다. 백엔드(python manage.py runserver)가 실행 중인지 확인하세요.'));
  }, []);

  // 검색어 입력 디바운스
  useEffect(() => {
    const t = setTimeout(() => {
      if (searchText !== (params.get('search') || '')) updateParams({ search: searchText.trim() });
    }, 400);
    return () => clearTimeout(t);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [searchText]);

  // 기사 목록
  useEffect(() => {
    const query = { page };
    FILTER_KEYS.forEach((k) => params.get(k) && (query[k] = params.get(k)));
    setLoading(true);
    api.get('/articles/', { params: query })
      .then((res) => { setData(res.data); setError(null); })
      .catch(() => setError('기사 목록을 불러오지 못했습니다.'))
      .finally(() => setLoading(false));
  }, [params, page]);

  const totalPages = data ? Math.max(1, Math.ceil(data.count / PAGE_SIZE)) : 1;
  const hasFilters = FILTER_KEYS.some((k) => params.get(k));

  return (
    <div className="articles">
      <h2 className="page-title">기사 탐색</h2>
      <p className="page-desc">
        {options ? `${options.total.toLocaleString()}건` : ''}의 경제 뉴스를 매체·이벤트·프레임별로 찾아보고,
        LLM(gpt-5.5)이 붙인 프레이밍 라벨과 판단 근거를 확인할 수 있습니다.
      </p>

      {error && <div className="error-box">{error}</div>}

      <div className="filter-bar card">
        <input
          className="search-input"
          type="search"
          placeholder="제목·본문 검색 (예: 기준금리 인상)"
          value={searchText}
          onChange={(e) => setSearchText(e.target.value)}
          aria-label="검색어"
        />

        <div className="filter-row">
          <select value={params.get('media') || ''} onChange={(e) => updateParams({ media: e.target.value })} aria-label="언론사">
            <option value="">전체 언론사</option>
            {options?.media.map((m) => (
              <option key={m.name} value={m.name}>{m.name} ({m.count.toLocaleString()})</option>
            ))}
          </select>

          <select value={params.get('group') || ''} onChange={(e) => updateParams({ group: e.target.value })} aria-label="언론사 그룹">
            <option value="">전체 성향</option>
            {options?.groups.map((g) => (
              <option key={g.group} value={g.group}>{g.group} ({g.count.toLocaleString()})</option>
            ))}
          </select>

          <select value={params.get('event_type') || ''} onChange={(e) => updateParams({ event_type: e.target.value })} aria-label="이벤트">
            <option value="">전체 이벤트</option>
            {options?.event_types.map((ev) => (
              <option key={ev.event_type} value={ev.event_type}>{eventName(ev.event_type)} ({ev.count.toLocaleString()})</option>
            ))}
          </select>

          <input
            type="date"
            value={params.get('date_from') || ''}
            min={options?.date_range.min}
            max={options?.date_range.max}
            onChange={(e) => updateParams({ date_from: e.target.value })}
            aria-label="시작일"
          />
          <span className="date-sep">~</span>
          <input
            type="date"
            value={params.get('date_to') || ''}
            min={options?.date_range.min}
            max={options?.date_range.max}
            onChange={(e) => updateParams({ date_to: e.target.value })}
            aria-label="종료일"
          />
        </div>

        <div className="filter-row">
          <div className="label-chips" role="group" aria-label="프레이밍 라벨">
            <button
              className={`chip ${!params.get('label') ? 'active' : ''}`}
              onClick={() => updateParams({ label: '' })}
            >
              전체
            </button>
            {LABEL_ORDER.map((l) => (
              <button
                key={l}
                className={`chip ${LABELS[l].className} ${params.get('label') === l ? 'active' : ''}`}
                onClick={() => updateParams({ label: params.get('label') === l ? '' : l })}
              >
                {LABELS[l].ko}
              </button>
            ))}
          </div>

          <select
            className="ordering"
            value={params.get('ordering') || '-date'}
            onChange={(e) => updateParams({ ordering: e.target.value === '-date' ? '' : e.target.value })}
            aria-label="정렬"
          >
            {ORDERING_OPTIONS.map((o) => <option key={o.value} value={o.value}>{o.label}</option>)}
          </select>

          {hasFilters && (
            <button className="btn reset" onClick={() => { setSearchText(''); setParams({}); }}>
              필터 초기화
            </button>
          )}
        </div>
      </div>

      {data && (
        <div className="result-summary">
          검색 결과 <strong>{data.count.toLocaleString()}</strong>건
          {loading && <span className="loading-dot"> · 불러오는 중…</span>}
        </div>
      )}

      {data && data.results.length === 0 && (
        <div className="empty-state card">조건에 맞는 기사가 없습니다.</div>
      )}

      <div className="article-list">
        {data?.results.map((a) => (
          <button key={a.article_id} className="article-card card" onClick={() => setSelected(a.article_id)}>
            <div className="article-meta">
              <span className="media-name">{a.media.name}</span>
              <span className="media-group">{a.media.group}</span>
              <span className="event">{eventName(a.event_type)}</span>
              <span className="date">{a.date}</span>
            </div>
            <div className="article-main">
              <h3>{a.title}</h3>
              <LabelBadge label={a.framing.label} />
            </div>
            <div className="article-foot">
              <span className="reason">{a.framing.reason}</span>
              <BiasMeter value={a.framing.bias_score} />
            </div>
          </button>
        ))}
      </div>

      {data && data.count > PAGE_SIZE && (
        <div className="pagination">
          <button className="btn" disabled={page <= 1} onClick={() => updateParams({ page: String(page - 1) })}>이전</button>
          <span>{page} / {totalPages.toLocaleString()}</span>
          <button className="btn" disabled={page >= totalPages} onClick={() => updateParams({ page: String(page + 1) })}>다음</button>
        </div>
      )}

      {selected && <ArticleDetail articleId={selected} onClose={closeDetail} />}
    </div>
  );
}

function ArticleDetail({ articleId, onClose }) {
  const [article, setArticle] = useState(null);

  useEffect(() => {
    api.get(`/articles/${articleId}/`).then((res) => setArticle(res.data));
    const onKey = (e) => e.key === 'Escape' && onClose();
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [articleId, onClose]);

  const f = article?.framing;

  return (
    <div className="drawer-backdrop" onClick={onClose}>
      <aside className="drawer" onClick={(e) => e.stopPropagation()} role="dialog" aria-label="기사 상세">
        <button className="drawer-close" onClick={onClose} aria-label="닫기">×</button>
        {!article ? (
          <p className="page-desc">불러오는 중…</p>
        ) : (
          <>
            <div className="article-meta">
              <span className="media-name">{article.media.name}</span>
              <span className="media-group">{article.media.group}</span>
              <span className="event">{eventName(article.event_type)}</span>
              <span className="date">{article.date}</span>
            </div>
            <h3 className="drawer-title">{article.title}</h3>

            <section className="score-grid">
              <div>
                <div className="score-label">프레이밍 (LLM)</div>
                <LabelBadge label={f.label} />
                {f.confidence != null && <span className="score-sub">확신도 {f.confidence.toFixed(2)}</span>}
              </div>
              <div>
                <div className="score-label">편향 점수</div>
                <BiasMeter value={f.bias_score} />
              </div>
              <div>
                <div className="score-label">감성 점수 (KcELECTRA)</div>
                <span className="score-num">{f.sentiment_score.toFixed(2)}</span>
              </div>
              <div>
                <div className="score-label">키워드 극성</div>
                <span className="score-num">{f.keyword_polarity.toFixed(2)}</span>
              </div>
            </section>

            {f.reason && (
              <section className="reason-box">
                <div className="score-label">LLM 판단 근거</div>
                <p>{f.reason}</p>
              </section>
            )}

            <section>
              <div className="score-label">본문</div>
              <p className="drawer-content">{article.content || '본문이 수집되지 않은 기사입니다.'}</p>
            </section>

            {article.url && (
              <a className="btn original-link" href={article.url} target="_blank" rel="noopener noreferrer">
                원문 보기 ↗
              </a>
            )}
          </>
        )}
      </aside>
    </div>
  );
}

export default Articles;
