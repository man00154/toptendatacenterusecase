# app.py
# Before running:
# pip install streamlit langchain langchain_community faiss-cpu google-cloud-aiplatform

import os
import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain.chat_models import ChatVertexAI
from langchain.embeddings import VertexAIEmbeddings

# --------------------------
# Use Cases
# --------------------------
use_cases = [
    "Energy Optimization & Cooling",
    "Predictive Maintenance",
    "Automated Workload Management",
    "Security & Threat Detection",
    "Capacity Planning & Forecasting",
    "Network Traffic Optimization",
    "Data Center Digital Twin",
    "Incident Response & Self-Healing",
    "AI-Augmented Monitoring & Alerts",
    "Sustainability & Green Initiatives",
]

# --------------------------
# Prompt Template
# --------------------------
strong_prompt = """
You are an expert AI assistant specialized in AI-driven data centers.
Answer queries based on the selected use case.
Provide detailed, clear, and actionable insights with examples or best practices.
"""

# --------------------------
# Knowledge Base
# --------------------------
docs = [
    "Energy Optimization & Cooling: Use AI to reduce power consumption of cooling systems.",
    "Predictive Maintenance: AI predicts equipment failures before they happen.",
    "Automated Workload Management: AI balances server loads dynamically.",
    "Security & Threat Detection: AI monitors and alerts on anomalies.",
    "Capacity Planning & Forecasting: AI predicts future resource needs.",
    "Network Traffic Optimization: AI routes traffic efficiently to avoid congestion.",
    "Data Center Digital Twin: AI simulates the full data center digitally.",
    "Incident Response & Self-Healing: AI auto-resolves common issues.",
    "AI-Augmented Monitoring & Alerts: AI enhances monitoring dashboards with predictions.",
    "Sustainability & Green Initiatives: AI optimizes energy and reduces carbon footprint.",
]

# --------------------------
# Split text into chunks
# --------------------------
text_splitter = CharacterTextSplitter(chunk_size=200, chunk_overlap=20)
split_docs = text_splitter.split_text(" ".join(docs))

# --------------------------
# Load Google Gemini API key from Streamlit secrets
# --------------------------
try:
    GEMINI_API_KEY = st.secrets["GOOGLE_API_KEY"]
    os.environ["GOOGLE_API_KEY"] = GEMINI_API_KEY
except KeyError:
    st.error("❌ Google Gemini API key not found. Add `GOOGLE_API_KEY` to Streamlit Secrets")
    st.stop()

# --------------------------
# Create embeddings and FAISS index
# --------------------------
embeddings = VertexAIEmbeddings(api_key=GEMINI_API_KEY)
vectorstore = FAISS.from_texts(split_docs, embeddings)

# --------------------------
# Helper: normalize text
# --------------------------
def simple_nlp(text: str) -> str:
    return " ".join(text.lower().strip().split())

# --------------------------
# AI Agent
# --------------------------
class SimpleAIAgent:
    def __init__(self):
        self.llm = ChatVertexAI(api_key=GEMINI_API_KEY, temperature=0.3)

    def respond(self, prompt: str) -> str:
        return self.llm.predict(prompt)

# --------------------------
# Agentic AI wrapper
# --------------------------
class AgenticAI:
    def __init__(self, agent: SimpleAIAgent):
        self.agent = agent

    def handle_query(self, query: str, context: str) -> str:
        final_prompt = f"{strong_prompt}\nContext: {context}\nQuestion: {query}\nAnswer:"
        return self.agent.respond(final_prompt)

# --------------------------
# Streamlit UI
# --------------------------
st.title("AI-Driven Data Center Assistant (Google Gemini)")

selected_use_case = st.selectbox("Select a Use Case:", use_cases)
user_query = st.text_input("Ask your question about the use case:")

if user_query:
    clean_query = simple_nlp(user_query)
    retrieved_docs = vectorstore.similarity_search(clean_query, k=2)
    context = " ".join([doc.page_content for doc in retrieved_docs])

    ai_agent = SimpleAIAgent()
    agentic_ai = AgenticAI(ai_agent)
    answer = agentic_ai.handle_query(clean_query, context)

    st.write("### AI Answer:")
    st.write(answer)
