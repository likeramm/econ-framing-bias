import axios from 'axios';

const api = axios.create({
  baseURL: process.env.REACT_APP_API_URL || 'http://localhost:8000/api',
});

export const getStats = () => api.get('/stats/').then((r) => r.data);

export const getBiasSummary = (coreOnly = true) =>
  api.get('/bias-summary/', { params: { core_only: coreOnly } }).then((r) => r.data);

export const getFramingDistribution = (media) =>
  api
    .get('/framing-distribution/', { params: { core_only: true, media } })
    .then((r) => r.data);

export const getBiasTimeseries = (by) =>
  api.get('/bias-timeseries/', { params: { by } }).then((r) => r.data);

export const getStock = (ticker = 'KOSPI') =>
  api.get('/stock/', { params: { ticker } }).then((r) => r.data);

export const getAnalysisResults = () => api.get('/analysis-results/').then((r) => r.data);

export const getArticles = (params) => api.get('/articles/', { params }).then((r) => r.data);

export default api;
