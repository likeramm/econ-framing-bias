import { useState, useEffect } from 'react';
import api from '../api';
import './Dashboard.css';

function Dashboard() {
  const [health, setHealth] = useState(null);

  useEffect(() => {
    api.get('/health/')
      .then(res => setHealth(res.data.status))
      .catch(() => setHealth('disconnected'));
  }, []);

  return (
    <div className="dashboard">
      <h2>대시보드</h2>
      <div className="status-card">
        <p>
          API 상태: {' '}
          <span className={health === 'ok' ? 'status-ok' : 'status-error'}>
            {health === 'ok' ? 'Connected' : 'Disconnected'}
          </span>
        </p>
      </div>
      <div className="card-grid">
        <div className="card">
          <h3>수집된 기사</h3>
          <p className="card-value">-</p>
        </div>
        <div className="card">
          <h3>분석 완료</h3>
          <p className="card-value">-</p>
        </div>
        <div className="card">
          <h3>경제 이벤트</h3>
          <p className="card-value">-</p>
        </div>
        <div className="card">
          <h3>언론사</h3>
          <p className="card-value">10</p>
        </div>
      </div>
    </div>
  );
}

export default Dashboard;
