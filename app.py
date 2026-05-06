"""
Life Sciences Multi-Agent AI Chatbot
LangGraph + RAG + FastAPI + Weaviate + MCP Schemas
Author: Premchand Kothapalli
"""

import logging
from typing import Annotated, TypedDict, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import Weaviate
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages

import weaviate

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ─── Config ───────────────────────────────────────────────────────────────────

WEAVIATE_URL   = "http://localhost:8080"
LLM_MODEL      = "gpt-4o"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

AGENT_TOPICS = {
    "clinical_trials": "clinical trials, trial phases, eligibility, FDA approval, investigational drugs",
    "drug_interactions": "drug interactions, contraindications, pharmacokinetics, adverse effects",
    "regulatory":       "FDA regulations, EMA guidelines, 510k, NDA, BLA, compliance, submissions",
    "pharmacovigilance":"adverse event reporting, FAERS, signal detection, post-market surveillance",
    "general":          "general life sciences questions not covered by other agents",
}

# ─── State Schema ─────────────────────────────────────────────────────────────

class AgentState(TypedDict):
    messages:       Annotated[list, add_messages]
    user_query:     str
    routed_to:      Optional[str]
    rag_context:    Optional[str]
    final_response: Optional[str]


# ─── Weaviate Vector Store ─────────────────────────────────────────────────────

def get_vectorstore(collection_name: str) -> Weaviate:
    client = weaviate.Client(WEAVIATE_URL)
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    return Weaviate(client=client, index_name=collection_name, text_key="content", embedding=embeddings)


# ─── MCP Schema Definitions ───────────────────────────────────────────────────

MCP_SCHEMAS = {
    "clinical_trials": {
        "schema_version": "1.0",
        "domain": "clinical_trials",
        "required_fields": ["trial_id", "phase", "indication", "sponsor", "status"],
        "context_sources": ["clinicaltrials.gov", "internal_trial_db", "fda_database"],
        "output_format": "structured_trial_summary",
    },
    "drug_interactions": {
        "schema_version": "1.0",
        "domain": "drug_interactions",
        "required_fields": ["drug_a", "drug_b", "interaction_type", "severity", "mechanism"],
        "context_sources": ["drugbank", "fda_label_database", "pubmed"],
        "output_format": "interaction_report",
    },
    "regulatory": {
        "schema_version": "1.0",
        "domain": "regulatory",
        "required_fields": ["submission_type", "agency", "guideline_reference", "applicability"],
        "context_sources": ["fda_guidance", "ema_guidance", "ich_guidelines"],
        "output_format": "regulatory_summary",
    },
}


# ─── Router Agent ─────────────────────────────────────────────────────────────

def router_agent(state: AgentState) -> AgentState:
    """
    Classifies user query and routes to the appropriate specialist sub-agent.
    Uses LLM zero-shot classification against known topic descriptions.
    """
    logger.info("[ROUTER] Classifying query...")

    llm = ChatOpenAI(model=LLM_MODEL, temperature=0)
    topic_list = "\n".join([f"- {k}: {v}" for k, v in AGENT_TOPICS.items()])

    prompt = ChatPromptTemplate.from_messages([
        ("system", f"""You are a routing agent for a life sciences chatbot.
Classify the user's question into exactly ONE of these topics:
{topic_list}

Respond with ONLY the topic key (e.g., 'clinical_trials'). No explanation."""),
        ("human", "{query}"),
    ])

    chain = prompt | llm
    result = chain.invoke({"query": state["user_query"]})
    routed_to = result.content.strip().lower()

    if routed_to not in AGENT_TOPICS:
        routed_to = "general"

    logger.info(f"[ROUTER] Routed to: {routed_to}")
    return {**state, "routed_to": routed_to}


# ─── RAG Retrieval ────────────────────────────────────────────────────────────

def rag_retrieval_agent(state: AgentState) -> AgentState:
    """
    Retrieves relevant domain knowledge from Weaviate vector store.
    Grounds the specialist agent with structured domain context + MCP schema.
    """
    domain = state["routed_to"]
    logger.info(f"[RAG] Retrieving context for domain: {domain}")

    try:
        vectorstore = get_vectorstore(collection_name=domain.upper())
        retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
        docs = retriever.get_relevant_documents(state["user_query"])
        context = "\n\n".join([d.page_content for d in docs])
    except Exception as e:
        logger.warning(f"[RAG] Retrieval failed: {e}. Proceeding without context.")
        context = "No additional context available."

    mcp = MCP_SCHEMAS.get(domain, {})
    mcp_context = f"\nMCP Schema for {domain}: {mcp}" if mcp else ""

    return {**state, "rag_context": context + mcp_context}


# ─── Specialist Sub-Agents ─────────────────────────────────────────────────────

def build_specialist_agent(domain: str):
    """Factory that builds a specialist agent node for a given domain."""

    def specialist_agent(state: AgentState) -> AgentState:
        logger.info(f"[AGENT:{domain.upper()}] Generating response")

        llm = ChatOpenAI(model=LLM_MODEL, temperature=0.2)
        prompt = ChatPromptTemplate.from_messages([
            ("system", f"""You are a specialist in {domain.replace('_', ' ')} for a life sciences company.
Use the provided context to answer the user's question accurately and concisely.
Always cite relevant guidelines, sources, or data when available.

Context:
{{rag_context}}"""),
            ("human", "{query}"),
        ])

        chain = prompt | llm
        result = chain.invoke({"query": state["user_query"], "rag_context": state.get("rag_context", "")})
        response = result.content

        logger.info(f"[AGENT:{domain.upper()}] Response generated ({len(response)} chars)")
        return {
            **state,
            "final_response": response,
            "messages": state["messages"] + [AIMessage(content=response)],
        }

    specialist_agent.__name__ = f"{domain}_agent"
    return specialist_agent


# ─── LangGraph Workflow ────────────────────────────────────────────────────────

def build_graph() -> StateGraph:
    graph = StateGraph(AgentState)

    # Nodes
    graph.add_node("router",   router_agent)
    graph.add_node("rag",      rag_retrieval_agent)
    for domain in AGENT_TOPICS:
        graph.add_node(domain, build_specialist_agent(domain))

    # Entry + routing
    graph.set_entry_point("router")
    graph.add_edge("router", "rag")

    def route_after_rag(state: AgentState) -> str:
        return state["routed_to"]

    graph.add_conditional_edges("rag", route_after_rag, {d: d for d in AGENT_TOPICS})

    for domain in AGENT_TOPICS:
        graph.add_edge(domain, END)

    return graph.compile()


# ─── FastAPI App ──────────────────────────────────────────────────────────────

app = FastAPI(title="Life Sciences Multi-Agent Chatbot", version="1.0.0")
chatbot = build_graph()


class ChatRequest(BaseModel):
    query:    str
    session_id: Optional[str] = None


class ChatResponse(BaseModel):
    response:   str
    routed_to:  str
    session_id: Optional[str]


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Main chat endpoint — routes query through multi-agent LangGraph pipeline."""
    logger.info(f"[API] Received query: {request.query[:80]}...")
    try:
        initial_state: AgentState = {
            "messages":       [HumanMessage(content=request.query)],
            "user_query":     request.query,
            "routed_to":      None,
            "rag_context":    None,
            "final_response": None,
        }
        result = chatbot.invoke(initial_state)
        return ChatResponse(
            response=result["final_response"],
            routed_to=result["routed_to"],
            session_id=request.session_id,
        )
    except Exception as e:
        logger.error(f"[API] Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health():
    return {"status": "ok", "agents": list(AGENT_TOPICS.keys())}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
