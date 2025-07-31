// src/api.js
import axios from 'axios';

const API = axios.create({
  baseURL: process.env.REACT_APP_API_URL,
});

export const analyzeTicker = async (ticker, useBacktrader = true) => {
  const res = await API.post('/api/analyze', {
    ticker,
    start_date: '2015-01-01',
    end_date: '2024-12-31',
    use_backtrader: useBacktrader
  });
  return res.data;
};
