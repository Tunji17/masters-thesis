# Medical Graph Extraction Client

A minimalist React application for testing the Medical Graph Extraction API. This client allows users to input clinical text, extract medical entities and relationships, and visualize the results.

## Features

- **Clean Interface**: Simple textarea for clinical text input
- **Real-time Processing**: Submit clinical notes to the extraction API
- **Entity Extraction**: View extracted medical entities with UMLS linking
- **Relationship Display**: See relationships between medical entities
- **Graph Visualization**: Display data stored in Neo4j knowledge graph
- **Responsive Design**: Works on desktop and mobile devices

## Prerequisites

- Node.js (v18 or higher)
- Running Medical Graph Extraction API on `http://localhost:8000`
- Neo4j database (for graph storage functionality)

## Installation

1. Install dependencies:
```bash
npm install
```

2. Start the development server:
```bash
npm run dev
```

3. Open your browser to `http://localhost:5173`

## Usage

1. **Enter Clinical Text**: Type or paste clinical notes into the textarea
   - Example: "Patient presents with chest pain and shortness of breath. Diagnosed with myocardial infarction."

2. **Analyze Text**: Click "Analyze Text" to process the clinical notes
   - The system will extract entities and relationships
   - Data will be automatically stored in Neo4j if available

3. **View Results**: Review the extracted information including:
   - **Entities**: Medical terms with UMLS CUI codes and semantic types
   - **Relationships**: Connections between medical entities
   - **Graph Data**: Information stored in the Neo4j knowledge graph

## API Integration

This client communicates with the Medical Graph Extraction API endpoints:

- `POST /api/extract/full` - Full extraction pipeline
- `POST /api/graph/query` - Query graph data
- `GET /api/graph/stats` - Get graph statistics
- `GET /health` - Health check

## Development

The application is built with:
- **React 18** with Hooks
- **Vite** for fast development and building
- **Axios** for API communication
- **Modern CSS** with responsive design

## Project Structure

```
src/
├── App.jsx          # Main application component
├── App.css          # Application styles
├── main.jsx         # React entry point
└── api/
    └── medicalApi.js # API service functions
```

## Configuration

The API base URL is set to `http://localhost:8000`. Modify this in `src/api/medicalApi.js` if your API runs on a different host/port.