import { BrowserRouter as Router, Routes, Route, NavLink } from 'react-router-dom';
import Articles from './pages/Articles';
import Classify from './pages/Classify';
import './App.css';

function App() {
  return (
    <Router>
      <div className="App">
        <nav className="navbar">
          <h1 className="logo">경제 뉴스 편향 탐지 및 주가 영향 분석 시스템</h1>
          <div className="nav-links">
            <NavLink to="/" end>기사 탐색</NavLink>
            <NavLink to="/classify">실시간 분류</NavLink>
          </div>
        </nav>
        <main className="main-content">
          <Routes>
            <Route path="/" element={<Articles />} />
            <Route path="/classify" element={<Classify />} />
          </Routes>
        </main>
      </div>
    </Router>
  );
}

export default App;
