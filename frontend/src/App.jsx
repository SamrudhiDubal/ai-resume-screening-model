import { useState } from 'react';
import ResumeScreener from './components/ResumeScreener.jsx';
import CandidateRanking from './components/CandidateRanking.jsx';

const TABS = { screen: 'Screen resume', rank: 'Rank candidates' };

export default function App() {
  const [tab, setTab] = useState('screen');
  return (
    <main className="app">
      <header>
        <h1>AI Resume Screening</h1>
        <nav role="tablist">
          {Object.entries(TABS).map(([key, label]) => (
            <button key={key} role="tab" aria-selected={tab === key} onClick={() => setTab(key)}>{label}</button>
          ))}
        </nav>
      </header>
      {tab === 'screen' ? <ResumeScreener /> : <CandidateRanking />}
    </main>
  );
}
