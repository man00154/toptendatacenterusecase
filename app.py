import os
import streamlit as st
import requests
import json
import numpy as np
from numpy.linalg import norm
from langgraph.graph import StateGraph, END
from langchain.agents import initialize_agent, Tool
from langchain.llms.base import LLM
from typing import List

# 🔑 Load API key
if "GOOGLE_API_KEY" in st.secrets:
    GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
else:
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")

# 🌐 Google API config
MODEL_NAME = "gemini-2.0-flash-lite"
API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL_NAME}:generateContent?key={GOOGLE_API_KEY}"

# 📚 Knowledge base
KNOWLEDGE_BASE = [
    "Energy Optimization & Cooling: AI can optimize HVAC and airflow for efficiency.",
    "Predictive Maintenance: Machine learning predicts equipment failures before they happen.",
    "Automated Workload Management: AI distributes workloads across servers dynamically.",
    "Security & Threat Detection: AI detects anomalies and cyber threats in real time.",
    "Capacity Planning & Forecasting: AI forecasts demand and optimizes hardware scaling.",
    "Network Traffic Optimization: AI balances traffic to reduce congestion and latency.",
    "Data Center Digital Twin: Virtual replica helps simulate changes and predict issues.",
    "Incident Response & Self-Healing: Automated scripts resolve common incidents.",
    "AI-Augmented Monitoring & Alerts: AI reduces false alarms and improves monitoring.",
    "Sustainability & Green Initiatives: AI improves energy efficiency and reduces carbon footprint."
]

# --- Helper: Call Gemini API for text completion ---
def call_gemini_api(prompt: str) -> str:
    headers = {"Content-Type": "application/json"}
    payload = {"contents": [{"parts": [{"text": prompt}]}]}
    response = requests.post(API_URL, headers=headers, data=json.dumps(payload))
    response.raise_for_status()
    return response.json()["candidates"][0]["content"]["parts"][0]["text"]

# --- Embeddings ---
def get_embedding(text: str) -> np.ndarray:
    url = f"https://generativelanguage.googleapis.com/v1beta/models/embedding-001:embedContent?key={GOOGLE_API_KEY}"
    payload = {
        "model": "embedding-001",   # ✅ FIX: remove "models/" prefix
        "content": {"parts": [{"text": text}]}
    }
    headers = {"Content-Type": "application/json"}
    response = requests.post(url, headers=headers, data=json.dumps(payload))

    if not response.ok:
        st.error(f"❌ Embedding API failed: {response.status_code}, {response.text}")
        return np.zeros(768, dtype="float32")  # fallback to dummy vector

    emb = response.json()["embedding"]["value"]
    return np.array(emb, dtype="float32")

# --- Build knowledge embeddings once ---
@st.cache_resource
def build_embeddings():
    return [get_embedding(t) for t in KNOWLEDGE_BASE]

EMBEDDINGS = build_embeddings()

# --- Simple RAG retriever (cosine similarity) ---
def rag_retriever(query: str, k: int = 2) -> str:
    q_emb = get_embedding(query)
    sims = [np.dot(q_emb, e) / (norm(q_emb) * norm(e)) for e in EMBEDDINGS]
    topk_idx = np.argsort(sims)[-k:][::-1]
    return " ".join([KNOWLEDGE_BASE[i] for i in topk_idx])

# --- Simple NLP (keyword-based retrieval) ---
def simple_nlp_retriever(query: str) -> str:
    q = query.lower()
    matches = [item for item in KNOWLEDGE_BASE if any(word in item.lower() for word in q.split())]
    return " ".join(matches) if matches else "General data center practices apply."

# --- Agentic AI (strong prompt executor) ---
def agentic_ai_executor(query: str) -> str:
    context = rag_retriever(query)
    strong_prompt = f"""
    ROLE: Data Center AI Expert
    OBJECTIVE: Provide highly accurate, actionable, and structured insights.
    RULES:
    - Use best practices from data center operations.
    - Give step-by-step recommendations.
    - Be professional and concise.
    CONTEXT: {context}
    QUESTION: {query}
    ANSWER:
    """
    return call_gemini_api(strong_prompt)

# --- LangChain Custom LLM Wrapper ---
class GeminiLLM(LLM):
    def _call(self, prompt: str, stop: List[str] = None) -> str:
        return call_gemini_api(prompt)

    @property
    def _identifying_params(self):
        return {"name": "gemini-llm"}

    @property
    def _llm_type(self):
        return "custom"

# --- LangChain Agent ---
llm = GeminiLLM()
tools = [
    Tool(name="Keyword NLP Retriever", func=simple_nlp_retriever, description="Retrieve context with simple NLP."),
    Tool(name="RAG Retriever", func=rag_retriever, description="Retrieve semantic context using embeddings."),
]
agent = initialize_agent(tools, llm, agent="zero-shot-react-description", verbose=False)

# --- LangGraph flow ---
def build_graph():
    sg = StateGraph(dict)

    def start(state):
        state["context"] = rag_retriever(state["query"])
        return state

    def reasoning(state):
        state["answer"] = agentic_ai_executor(state["query"])
        return state

    sg.add_node("start", start)
    sg.add_node("reasoning", reasoning)
    sg.set_entry_point("start")
    sg.add_edge("start", "reasoning")
    sg.add_edge("reasoning", END)

    return sg.compile()

# --- Streamlit UI ---
st.title("🤖 MANISH SINGH - Data Center Agentic AI Assistant (Simple RAG LLM NLP LANGGRAPH)")
st.write("This assistant combines: 1 AI Agent (LangChain), 1 Agentic AI, 1 Simple NLP retriever, 1 LangGraph pipeline.")

query = st.text_input("💬 Ask a question:")

if st.button("Ask"):
    if not GOOGLE_API_KEY:
        st.error("❌ Missing Google API Key! Add it to secrets.toml")
    elif query:
        # Run LangChain Agent
        agent_answer = agent.run(query)

        # Run LangGraph + Agentic AI
        graph = build_graph()
        result = graph.invoke({"query": query})

        st.subheader("🔎 LangChain Agent Answer")
        st.info(agent_answer)

        st.subheader("⚡ Agentic AI Answer")
        st.success(result["answer"])
