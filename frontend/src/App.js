import React, { useState } from 'react';
import axios from 'axios';
import Plot from 'react-plotly.js';

function App() {
  const [ticker, setTicker] = useState('AAPL');
  const [startDate, setStartDate] = useState('2015-01-01');
  const [endDate, setEndDate] = useState('2024-12-31');
  const [useBacktrader, setUseBacktrader] = useState(true);
  const [loading, setLoading] = useState(false);
  const [charts, setCharts] = useState(null);
  const [stats, setStats] = useState(null);
  const [error, setError] = useState(null);

  const handleAnalyze = async () => {
    setLoading(true);
    setError(null);
    try {
      const response = await axios.post('http://127.0.0.1:5000/api/analyze', {
        ticker,
        start_date: startDate,
        end_date: endDate,
        use_backtrader: useBacktrader
      });

      setCharts({
        fig_price: JSON.parse(response.data.fig_price),
        fig_post: JSON.parse(response.data.fig_post),
        fig_perf: JSON.parse(response.data.fig_perf),
      });
      setStats(response.data.stats);
    } catch (err) {
      setError('Failed to fetch data. Check backend console/logs.');
    }
    setLoading(false);
  };

  return (
    <div style={{ padding: '20px', fontFamily: 'Arial' }}>
      <h1>Regime Detection Dashboard</h1>

      {/* Input Section */}
      <div style={{ marginBottom: '20px' }}>
        <input value={ticker} onChange={e => setTicker(e.target.value)} placeholder="Ticker" />
        <input type="date" value={startDate} onChange={e => setStartDate(e.target.value)} />
        <input type="date" value={endDate} onChange={e => setEndDate(e.target.value)} />
        <label style={{ marginLeft: '10px' }}>
          <input
            type="checkbox"
            checked={useBacktrader}
            onChange={e => setUseBacktrader(e.target.checked)}
          /> Use Backtrader
        </label>
        <button onClick={handleAnalyze} disabled={loading} style={{ marginLeft: '10px' }}>
          {loading ? 'Analyzing...' : 'Analyze'}
        </button>
      </div>

      {/* Error */}
      {error && <p style={{ color: 'red' }}>{error}</p>}

      {/* Stats Section */}
      {stats && (
        <div style={{ display: 'flex', gap: '20px', marginBottom: '20px' }}>
          <StatCard title="Final Portfolio Value" value={`$${stats.final_value?.toFixed(2)}`} />
          <StatCard title="Total Return" value={`${stats.total_return?.toFixed(2)}%`} />
          <StatCard title="Sharpe Ratio" value={stats.sharpe_ratio?.toFixed(2)} />
          <StatCard title="Max Drawdown" value={`${stats.max_drawdown?.toFixed(2)}%`} />
        </div>
      )}

      {/* Charts */}
      {charts && (
        <>
          <Chart title="Price Chart with Regimes" fig={charts.fig_price} />
          <Chart title="Posterior Probabilities" fig={charts.fig_post} />
          <Chart title="Strategy vs Buy & Hold" fig={charts.fig_perf} />
        </>
      )}
    </div>
  );
}

function Chart({ title, fig }) {
  return (
    <div style={{ marginBottom: '40px' }}>
      <h2>{title}</h2>
      <Plot data={fig.data} layout={{ ...fig.layout, autosize: true }} useResizeHandler style={{ width: "100%" }} />
    </div>
  );
}

function StatCard({ title, value }) {
  return (
    <div style={{
      border: '1px solid #ddd',
      padding: '15px',
      borderRadius: '8px',
      minWidth: '180px',
      textAlign: 'center',
      background: '#f8f8f8',
      boxShadow: '0 2px 4px rgba(0,0,0,0.1)'
    }}>
      <h4 style={{ marginBottom: '5px' }}>{title}</h4>
      <p style={{ fontSize: '1.5rem', fontWeight: 'bold' }}>{value}</p>
    </div>
  );
}

export default App;
