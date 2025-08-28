import axios from 'axios';

const API_BASE_URL = 'https://badly-powerful-marmot.ngrok-free.app';

const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

export const medicalApi = {
  // Process clinical text through the full extraction pipeline
  extractFullPipeline: async (text, options = {}) => {
    try {
      const response = await api.post('/api/extract/full', {
        text,
        threshold: options.threshold || 0.5,
        max_tokens: options.maxTokens || 512,
        store_in_graph: options.storeInGraph !== false, // default to true
      });
      return response.data;
    } catch (error) {
      console.error('Full extraction failed:', error);
      throw error;
    }
  },

  // Query the graph for specific data
  queryGraph: async (query, parameters = {}) => {
    try {
      const response = await api.post('/api/graph/query', {
        query,
        parameters,
      });
      return response.data;
    } catch (error) {
      console.error('Graph query failed:', error);
      throw error;
    }
  },

  // Get graph statistics
  getGraphStats: async () => {
    try {
      const response = await api.get('/api/graph/stats');
      return response.data;
    } catch (error) {
      console.error('Failed to get graph stats:', error);
      throw error;
    }
  },

  // Health check
  healthCheck: async () => {
    try {
      const response = await api.get('/health');
      return response.data;
    } catch (error) {
      console.error('Health check failed:', error);
      throw error;
    }
  },

  // Search entities
  searchEntities: async (searchParams = {}) => {
    try {
      const response = await api.get('/api/graph/entities/search', {
        params: searchParams,
      });
      return response.data;
    } catch (error) {
      console.error('Entity search failed:', error);
      throw error;
    }
  },
};

export default medicalApi;