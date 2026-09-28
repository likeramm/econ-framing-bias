import { useState } from 'react';
import api from '../api';
import { LABELS, LABEL_ORDER, LabelBadge } from '../labels';
import './Classify.css';

// 같은 사실을 다르게 프레이밍한 예시 (README 의 GDP 사례)
const EXAMPLES = [
  { title: '2분기 GDP 2.3% 성장…견조한 성장세 지속', content: '' },
  { title: '2분기 GDP 2.3%…성장 둔화 우려 확대, 불확실성 고조', content: '' },
  { title: '한국은행, 2분기 실질 GDP 전분기 대비 2.3% 성장 발표', content: '' },
];

function Classify() {
  const [title, setTitle] = useState('');
  const [content, setContent] = useState('');
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(false);

  const submit = async (t = title, c = content) => {
    if (!t.trim()) return;
    setLoading(true);
    setError(null);
    try {
      const res = await api.post('/classify/', { title: t, content: c });
      setResult({ ...res.data, title: t });
    } catch (e) {
      setResult(null);
      setError(e.response?.data?.detail || 'API 서버에 연결할 수 없습니다. 백엔드가 실행 중인지 확인하세요.');
    } finally {
      setLoading(false);
    }
  };

  const runExample = (ex) => {
    setTitle(ex.title);
    setContent(ex.content);
    submit(ex.title, ex.content);
  };

  return (
    <div className="classify">
      <h2 className="page-title">실시간 프레이밍 분류</h2>
      <p className="page-desc">
        기사 제목(과 본문)을 입력하면, gpt-5.5 라벨로 학습한 KLUE-RoBERTa-large 모델이 프레임을 바로 분류합니다.
        API 비용 없이 로컬 GPU에서 동작합니다.
      </p>

      <div className="classify-grid">
        <form
          className="card classify-form"
          onSubmit={(e) => { e.preventDefault(); submit(); }}
        >
          <label htmlFor="cls-title">기사 제목 *</label>
          <input
            id="cls-title"
            value={title}
            onChange={(e) => setTitle(e.target.value)}
            placeholder="예: 수출 3개월 연속 증가…반도체 호조"
            maxLength={500}
          />

          <label htmlFor="cls-content">본문 (선택 · 앞 500자만 사용)</label>
          <textarea
            id="cls-content"
            value={content}
            onChange={(e) => setContent(e.target.value)}
            rows={8}
            placeholder="본문을 넣으면 제목과 함께 판단합니다. 제목과 본문이 충돌하면 제목의 프레임을 우선합니다."
          />

          <button className="btn btn-primary" type="submit" disabled={loading || !title.trim()}>
            {loading ? '분류 중…' : '분류하기'}
          </button>

          <div className="examples">
            <span className="examples-label">같은 숫자, 다른 프레임 — 예시로 해보기</span>
            {EXAMPLES.map((ex) => (
              <button type="button" key={ex.title} className="example-btn" onClick={() => runExample(ex)}>
                {ex.title}
              </button>
            ))}
          </div>
        </form>

        <div className="card classify-result" aria-live="polite">
          {error && <div className="error-box">{error}</div>}

          {!error && !result && (
            <p className="placeholder">왼쪽에 제목을 입력하고 분류해보세요.<br />첫 요청은 모델을 불러오느라 몇 초 걸릴 수 있습니다.</p>
          )}

          {result && (
            <>
              <div className="result-head">
                <span className="score-label">분류 결과</span>
                <div className="result-label">
                  <LabelBadge label={result.label} />
                  <span className="result-prob">{(result.probabilities[result.label] * 100).toFixed(1)}%</span>
                </div>
                <p className="result-title">“{result.title}”</p>
              </div>

              <div className="prob-list">
                {LABEL_ORDER.map((l) => {
                  const p = result.probabilities[l] ?? 0;
                  return (
                    <div key={l} className="prob-row">
                      <span className="prob-name">{LABELS[l].ko}</span>
                      <div className="prob-track">
                        <div className={`prob-fill ${LABELS[l].className}`} style={{ width: `${p * 100}%` }} />
                      </div>
                      <span className="prob-value">{(p * 100).toFixed(1)}%</span>
                    </div>
                  );
                })}
              </div>

              <p className="result-note">
                {result.used_content ? '제목 + 본문으로 판단했습니다.' : '제목만으로 판단했습니다.'}
                {' '}이 모델은 LLM 판단을 재현하도록 학습됐으며(테스트 Macro-F1 0.84), 사람 기준 정확도는 골든셋으로 검증 중입니다.
              </p>
            </>
          )}
        </div>
      </div>
    </div>
  );
}

export default Classify;
