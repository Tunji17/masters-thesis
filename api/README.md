# Medical Graph Extraction API

A FastAPI service that extracts medical entities and relationships from clinical notes using GLiNER, SciSpacy, and Gemma models, with knowledge graph storage in Neo4j.

## 🚀 Quick Start

### Using the Makefile (Recommended)

1. **Complete setup:**
   ```bash
   make setup
   ```

2. **Start the system:**
   ```bash
   make run-all
   ```

3. **Access the API:**
   - API Documentation: http://localhost:8000/docs
   - Neo4j Browser: http://localhost:7474 (neo4j/password123)
   - Health Check: http://localhost:8000/health

### Manual Setup

1. **Set up environment:**
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

2. **Start services:**
   ```bash
   docker-compose up -d
   ```

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

Environment variables (used by Makefile and docker-compose):

```env
# Neo4j
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=password123

# Models
GEMMA_MODEL=google/gemma-3-4b-it
NER_MODEL=mlx-ner-medical
SCISPACY_MODEL=en_core_sci_lg

# API
API_PORT=8000
LOG_LEVEL=INFO
CACHE_SIZE=10000
```

**Note:** The Makefile automatically sets these environment variables when running `make run-api`. For manual setup, create a `.env` file or export these variables.

## 🏥 Models Used

- **MLX-NER-Medical**: `mlx-ner-medical` - Medical entity extraction (optimized for Apple Silicon)
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

### Makefile Commands

View all available commands:
```bash
make help
```

**Setup Commands:**
- `make setup` - Set up both API and notebook environments
- `make setup-api` - Set up API virtual environment and dependencies  
- `make setup-src` - Set up source/notebook environment

**Running Commands:**
- `make run-neo4j` - Start Neo4j database container
- `make run-api` - Run API server locally (requires Neo4j)
- `make run-all` - Start Neo4j and API together
- `make stop` - Stop Neo4j container

**Development Commands:**
- `make test` - Run API tests
- `make lint` - Run code linting
- `make clean` - Clean up containers, volumes, and cache
- `make dev-api` - Quick start for API development
- `make dev-notebook` - Start Jupyter for notebook development
- `make status` - Show status of services

### Manual Development Setup

Run locally (without Docker):
```bash
cd api
pip install -r requirements.txt
python -m spacy download en_core_sci_lg
uvicorn main:app --reload
```

View logs:
```bash
docker-compose logs -f api
```

Reset Neo4j data:
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