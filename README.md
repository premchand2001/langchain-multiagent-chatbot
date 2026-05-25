# LangGraph Multi-Agent Routing — Reference Implementation

A minimal, readable implementation of a multi-agent routing architecture for life sciences queries. Built to demonstrate how LangGraph handles intent classification and domain-specific sub-agent delegation with RAG grounding.

If you want the full production version with Weaviate, PDF ingestion, streaming, and a Streamlit UI, see [life-science-chatbot](https://github.com/premchand2001/life-science-chatbot).

---

## Architecture

```
User Query (FastAPI /chat)
       │
       ▼
  Router Agent (LLM zero-shot classification)
       │
       ├── clinical_trials
       ├── drug_interactions
       ├── regulatory
       ├── pharmacovigilance
       └── general
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
       Response
```

## Stack

| Component | Technology |
|---|---|
| Agent orchestration | LangGraph |
| LLM | GPT-4o via LangChain |
| Vector store | Weaviate |
| Embeddings | HuggingFace all-MiniLM-L6-v2 |
| Schema grounding | MCP Schemas |
| API backend | FastAPI |

## Agents

| Agent | Handles |
|---|---|
| `clinical_trials` | Trial phases, FDA approval, eligibility, sponsors |
| `drug_interactions` | Contraindications, pharmacokinetics, adverse effects |
| `regulatory` | FDA/EMA guidelines, NDA/BLA submissions, ICH compliance |
| `pharmacovigilance` | FAERS reporting, signal detection, post-market surveillance |
| `general` | Catch-all for out-of-domain questions |

## What This Demonstrates

- LangGraph supervisor node doing zero-shot intent classification
- Per-agent RAG retrieval from Weaviate before responding
- MCP schemas grounding each agent with structured field definitions and output format expectations
- Session-aware FastAPI endpoint for multi-turn conversations

## Usage

```bash
pip install langchain langgraph langchain-openai langchain-community fastapi uvicorn weaviate-client sentence-transformers

export OPENAI_API_KEY=your_key

docker run -d -p 8080:8080 semitechnologies/weaviate:latest

python app.py
```

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "What are the Phase 3 trial requirements for a new oncology drug?"}'
```

```json
{
  "response": "Phase 3 clinical trials for oncology drugs...",
  "routed_to": "clinical_trials",
  "session_id": null
}
```
