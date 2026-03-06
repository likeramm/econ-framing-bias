import { useState, useEffect } from 'react';
import api from '../api';
import './Articles.css';

function Articles() {
  const [articles, setArticles] = useState([]);

  useEffect(() => {
    api.get('/articles/')
      .then(res => setArticles(res.data.results || []))
      .catch(() => setArticles([]));
  }, []);

  return (
    <div className="articles">
      <h2>기사 분석</h2>
      {articles.length === 0 ? (
        <div className="empty-state">
          <p>수집된 기사가 없습니다.</p>
          <p>크롤러를 실행하여 기사를 수집해주세요.</p>
        </div>
      ) : (
        <div className="article-list">
          {articles.map(article => (
            <div key={article.id} className="article-card">
              <div className="article-meta">
                <span className="media-name">{article.media?.name}</span>
                <span className="date">{new Date(article.published_at).toLocaleDateString('ko-KR')}</span>
              </div>
              <h3>{article.title}</h3>
              {article.framing && (
                <div className="framing-badge">
                  <span className={`badge ${article.framing.framing_type}`}>
                    {article.framing.framing_type}
                  </span>
                  <span>편향 점수: {article.framing.bias_score?.toFixed(2)}</span>
                </div>
              )}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

export default Articles;
