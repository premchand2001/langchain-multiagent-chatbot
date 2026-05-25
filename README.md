# langchain-multiagent-chatbot

Production-grade life sciences multi-agent AI chatbot built with LangGraph, RAG, FastAPI, and Weaviate. Each domain routes to a specialized sub-agent grounded with structured knowledge via MCP schemas.

## Architecture

```
User Query (FastAPI /chat)
       │
       ▼
  Router Agent (LLM zero-shot classification)
       │
       ├──→ clinical_trials
       ├──→ drug_interactions
       ├──→ regulatory
       ├──→ pharmacovigilance
       └──→ general
              │
              ▼
       RAG Retrieval Node
       (Weaviate vector store + MCP schema context)
              │
              ▼
       Specialist Sub-Agent
       (domain-specific LLM with retrieved context)
              │
              ▼
       Final Response → User
```

## Stack

| Component | Technology |
|-----------|-----------|
| Agent Orchestration | LangGraph |
| LLM | GPT-4o via LangChain |
| Vector Store | Weaviate |
| Embeddings | HuggingFace (all-MiniLM-L6-v2) |
| Schema Grounding | MCP Schemas |
| API Backend | FastAPI |
| UI Demo | Streamlit |

## Files

```
langchain-multiagent-chatbot/
├── app.py          # LangGraph workflow + FastAPI endpoints
└── README.md
```

## Key Features

- **Multi-Agent Routing** — LangGraph routes each query to one of 5 specialist sub-agents based on LLM zero-shot classification
- **RAG per Agent** — each sub-agent retrieves domain-specific context from Weaviate before responding
- **MCP Schemas** — structured metadata schemas ground each agent with required fields, data sources, and output format expectations
- **Session-aware API** — FastAPI `/chat` endpoint with session ID support for multi-turn conversations
- **Domain Coverage** — clinical trials, drug interactions, regulatory (FDA/EMA), pharmacovigilance, general

## Agents

| Agent | Domain |
|-------|--------|
| `clinical_trials` | Trial phases, FDA approval, eligibility, sponsors |
| `drug_interactions` | Contraindications, pharmacokinetics, adverse effects |
| `regulatory` | FDA/EMA guidelines, NDA/BLA submissions, ICH compliance |
| `pharmacovigilance` | FAERS reporting, signal detection, post-market surveillance |
| `general` | Catch-all for out-of-domain questions |

## API Usage

```bash
# Start server
uvicorn app:app --host 0.0.0.0 --port 8000

# Chat
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "What are the Phase 3 trial requirements for a new oncology drug?"}'
```

Response:
```json
{
  "response": "Phase 3 clinical trials for oncology drugs...",
  "routed_to": "clinical_trials",
  "session_id": null
}
```

## Setup

```bash
pip install langchain langgraph langchain-openai langchain-community fastapi uvicorn weaviate-client sentence-transformers

export OPENAI_API_KEY=your_key

# Start Weaviate
docker run -d -p 8080:8080 semitechnologies/weaviate:latest

python app.py
```

## Based On

Real work as **Generative AI Engineer**: end-to-end multi-agent chatbot for the life sciences domain with LangGraph routing, RAG, MCP schemas, and FastAPI backend.
