import { BrowserRouter as Router, Routes, Route, Link } from 'react-router-dom';
import Dashboard from './pages/Dashboard';
import Articles from './pages/Articles';
import './App.css';

function App() {
  return (
    <Router>
      <div className="App">
        <nav className="navbar">
          <h1 className="logo">경제뉴스 프레이밍 편향 탐지기</h1>
          <div className="nav-links">
            <Link to="/">대시보드</Link>
            <Link to="/articles">기사 분석</Link>
          </div>
        </nav>
        <main className="main-content">
          <Routes>
            <Route path="/" element={<Dashboard />} />
            <Route path="/articles" element={<Articles />} />
          </Routes>
        </main>
      </div>
    </Router>
  );
}

export default App;
