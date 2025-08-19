# Medical Graph Extraction API

A FastAPI service that extracts medical entities and relationships from clinical notes using GLiNER, SciSpacy, and Gemma models, with knowledge graph storage in Neo4j.

## 🚀 Quick Start

1. **Set up environment:**
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

2. **Start services:**
   ```bash
   docker-compose up -d
   ```

3. **Access the API:**
   - API Documentation: http://localhost:8000/docs
   - Neo4j Browser: http://localhost:7474 (neo4j/password123)
   - Health Check: http://localhost:8000/health

## 📁 Project Structure

```
api/
├── main.py                        # FastAPI application
├── config.py                      # Configuration
├── models.py                      # Pydantic models
├── requirements.txt               # Dependencies
├── services/
│   ├── entity_extractor.py        # GLiNER + SciSpacy
│   ├── relation_extractor.py      # Gemma model
│   └── neo4j_service.py          # Neo4j integration
├── routers/
│   ├── extraction.py             # Extraction endpoints
│   ├── graph.py                  # Graph endpoints
│   └── health.py                 # Health checks
└── docker-compose.yml            # Container orchestration
```

## 🔗 API Endpoints

### Entity Extraction
- `POST /api/extract/entities` - Extract medical entities
- `POST /api/extract/relationships` - Extract relationships
- `POST /api/extract/full` - Complete pipeline with Neo4j storage

### Graph Operations
- `POST /api/graph/store` - Store entities and relationships
- `GET /api/graph/query` - Execute Cypher queries
- `GET /api/graph/stats` - Graph statistics

### Health & Monitoring
- `GET /health` - Service health check
- `GET /health/models` - Model loading status
- `GET /health/neo4j` - Neo4j connection status

## 💡 Usage Examples

### Extract entities from clinical text:
```bash
curl -X POST "http://localhost:8000/api/extract/entities" \
  -H "Content-Type: application/json" \
  -d '{"text": "Patient has diabetes and hypertension.", "threshold": 0.5}'
```

### Complete extraction pipeline:
```bash
curl -X POST "http://localhost:8000/api/extract/full" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Patient presents with chest pain. ECG shows ST elevation. Administered aspirin and transferred to cathlab.",
    "store_in_graph": true
  }'
```

### Query the knowledge graph:
```bash
curl -X POST "http://localhost:8000/api/graph/query" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "MATCH (e:MedicalEntity) RETURN e.name, e.cui LIMIT 10"
  }'
```

## 🔧 Configuration

Environment variables (see `.env.example`):

```env
# Neo4j
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=password123

# Models
GEMMA_MODEL=google/gemma-3-4b-it
GLINER_MODEL=Ihor/gliner-biomed-bi-large-v1.0
SCISPACY_MODEL=en_core_sci_lg

# API
API_PORT=8000
LOG_LEVEL=INFO
CACHE_SIZE=10000
```

## 🏥 Models Used

- **GLiNER**: `Ihor/gliner-biomed-bi-large-v1.0` - Medical entity extraction
- **SciSpacy**: `en_core_sci_lg` - UMLS entity linking
- **Gemma**: `google/gemma-3-4b-it` - Relationship extraction

## 📊 Neo4j Graph Structure

- **Nodes**: `MedicalEntity` with properties:
  - `name`: Entity text
  - `cui`: UMLS CUI identifier
  - `canonical_name`: UMLS canonical name
  - `semantic_types`: UMLS semantic types
  - `linking_score`: Confidence score

- **Relationships**: `RELATIONSHIP` with `type` property

## 🔍 Development

### Run locally (without Docker):
```bash
cd api
pip install -r requirements.txt
python -m spacy download en_core_sci_lg
uvicorn main:app --reload
```

### View logs:
```bash
docker-compose logs -f api
```

### Reset Neo4j data:
```bash
curl -X DELETE "http://localhost:8000/api/graph/clear"
```

## 🎯 Features

- ✅ Medical entity extraction with UMLS linking
- ✅ Relationship extraction using LLM
- ✅ Knowledge graph storage in Neo4j
- ✅ Batch processing support
- ✅ Comprehensive health checks
- ✅ Caching for performance
- ✅ Docker containerization
- ✅ Production-ready with proper error handling