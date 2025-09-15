# 🚀 Smart Product Discovery AI Agent

**Migrated from Colab to Production-Ready FastAPI + TiDB Cloud**

An intelligent AI-powered product recommendation system that solves real-world product discovery challenges using vector similarity search, built with TiDB Cloud Vector Search and FastAPI.

## 🎥 Demonstration Video  
Watch the demo (https://drive.google.com/file/d/1g5ZC6a1r5azBSwWAwMaanozIDMuLMBQ5/view?usp=sharing)


## TiDB Cloud Account Email: wilsonbmiles@gmail.com


## Project Repository URL: https://github.com/Dessyonpoint/smart-product-ai

## 🎯 Problem Solved

E-commerce platforms struggle with helping users discover relevant products through natural language queries. This AI agent provides intelligent product recommendations by understanding user intent through semantic search.

## 🏗️ Competition Requirements ✅

### ✅ **Component 1: Data Ingestion & Vector Indexing**
- Ingests product data (title, description, metadata)
- Generates embeddings using Sentence Transformers
# Smart Product Discovery AI Agent

## Competition Submission - TiDB Cloud AI Challenge

### Overview
AI-powered product recommendation system using TiDB Vector Search, semantic similarity, and external API integration for e-commerce product discovery.

### Architecture
- **Database**: TiDB Cloud with VECTOR(384) embeddings
- **ML Model**: SentenceTransformer for semantic search
- **LLM**: OpenAI GPT-3.5 for product analysis (optional)
- **APIs**: External price comparison and inventory data
- **Framework**: FastAPI with async operations

---

## Quick Start

### Prerequisites
```bash
pip install -r requirements.txt
```

**Required packages:**
- fastapi
- uvicorn
- mysql-connector-python
- sentence-transformers
- openai
- numpy
- python-dotenv
- aiohttp

### Environment Setup
Create `.env` file:
```
TIDB_HOST=gateway01.us-west-2.prod.aws.tidbcloud.com
TIDB_PORT=4000
TIDB_USER=your_tidb_user
TIDB_PASSWORD=your_tidb_password
TIDB_DATABASE=recommenddb
OPENAI_API_KEY=sk-your_openai_key_here
```

### Launch Application
```bash
uvicorn main:app --reload
```

Server runs on: `http://127.0.0.1:8000`

---

## Demo Instructions

### 1. Check System Status
```bash
curl http://127.0.0.1:8000/
```
Returns competition compliance status and component health.

### 2. Load Demo Products
```bash
curl -X POST "http://127.0.0.1:8000/ingest/batch"
```
Loads 7 sample products with embeddings into TiDB.

### 3. Test AI Product Search
```bash
curl -X POST "http://127.0.0.1:8000/search" \
  -H "Content-Type: application/json" \
  -d '{"query": "wireless headphones", "top_k": 5}'
```

**Search Features:**
- Semantic similarity matching
- Category filtering
- Price range filtering
- AI analysis (if OpenAI configured)
- External API enrichment

### 4. View Database Statistics
```bash
curl http://127.0.0.1:8000/stats
```

### 5. Test Single Product Ingestion
```bash
curl -X POST "http://127.0.0.1:8000/ingest/product" \
  -H "Content-Type: application/json" \
  -d '{
    "id": "demo001",
    "title": "Smart Watch Pro",
    "description": "Advanced fitness tracking with GPS and heart rate monitor",
    "category": "Wearables",
    "price": 299.99,
    "brand": "TechCorp"
  }'
```

---

## Competition Requirements Met

### ✅ TiDB Cloud Integration
- SSL-secured connection to TiDB Cloud
- Vector embeddings stored as VECTOR(384)
- Dynamic connection fallback methods

### ✅ TiDB Vector Search
- VEC_COSINE_DISTANCE for similarity search
- VEC_L2_DISTANCE fallback
- Manual similarity calculation backup

### ✅ Data Ingestion Pipeline
- Real-time product indexing
- Automatic embedding generation
- Batch and single product support

### ✅ LLM Integration
- OpenAI GPT-3.5 product analysis
- Contextual recommendations
- Graceful degradation if unavailable

### ✅ External API Integration
- Price comparison data
- Inventory status simulation
- Market trends analysis
- Real-time data enrichment

### ✅ Real-World Application
- E-commerce product discovery
- Multi-step AI agent workflow
- Production-ready error handling

---

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | System health and compliance status |
| `/search` | POST | AI-powered product search |
| `/ingest/product` | POST | Add single product |
| `/ingest/batch` | POST | Load demo products |
| `/stats` | GET | Database and system statistics |
| `/validate-keys` | GET | API key validation |

---

## Search Query Parameters

```json
{
  "query": "search text",
  "top_k": 5,
  "category_filter": "Electronics",
  "price_max": 500.0,
  "include_external_data": true,
  "include_ai_analysis": true
}
```

---

## Sample Search Response

```json
{
  "query": "wireless headphones",
  "results": [
    {
      "id": "item001",
      "title": "Wireless Bluetooth Headphones",
      "description": "High-quality wireless headphones...",
      "category": "Audio",
      "price": 299.99,
      "brand": "SoundMax",
      "similarity_score": 0.92
    }
  ],
  "total_results": 3,
  "processing_time": "0.145s",
  "method_used": "VEC_COSINE_DISTANCE",
  "ai_analysis": "These headphones match your query perfectly...",
  "external_insights": {
    "price_comparison": [...],
    "availability_status": [...],
    "market_trends": {...}
  }
}
```

---

## Technical Highlights

### Vector Search Implementation
- 384-dimensional embeddings using SentenceTransformer
- TiDB native vector functions for optimal performance
- Fallback mechanisms for reliability

### AI Agent Workflow
1. **Query Processing**: Semantic embedding generation
2. **Vector Search**: TiDB similarity matching
3. **LLM Analysis**: OpenAI contextual insights
4. **API Enrichment**: External data integration
5. **Response Assembly**: Unified recommendation output

### Production Features
- SSL certificate management
- Connection retry logic
- Error handling and logging
- Async operations for scalability

---

## Notes for Judges

- **OpenAI Quota**: Core functionality works without OpenAI (vector search independent)
- **TiDB Connection**: Multiple SSL methods ensure reliable connectivity
- **Demo Data**: 13 products pre-loaded for immediate testing
- **Real-time**: All operations are live, no cached responses
- **Scalable**: Async FastAPI design supports concurrent requests

**Competition Status**: ✅ FULLY COMPLIANT - Ready for evaluation

---

**Contact**: AI Agent demonstrates professional-grade e-commerce search with TiDB Cloud vector capabilities.
