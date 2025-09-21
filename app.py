# Install these packages before running
# pip install streamlit langchain langchain_community openai faiss-cpu sentence-transformers

import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.chat_models import ChatOpenAI

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
    "Sustainability & Green Initiatives"
]

# --------------------------
# Strong Prompt Template
# --------------------------
strong_prompt = """
You are an expert AI assistant specialized in AI-driven data centers. 
You will answer queries based on the selected use case.
Provide detailed, clear, and actionable insights.
Include examples or best practices wherever possible.
"""

# --------------------------
# FAISS + Simple RAG Setup
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
    "Sustainability & Green Initiatives: AI optimizes energy and reduces carbon footprint."
]

text_splitter = CharacterTextSplitter(chunk_size=200, chunk_overlap=20)
split_docs = text_splitter.split_text(" ".join(docs))

# --------------------------
# OpenAI Embeddings Setup
# --------------------------
try:
    OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
except KeyError:
    st.error(
        "OpenAI API key not found. Please add your `OPENAI_API_KEY` to the `secrets.toml` file or "
        "as a secret in your Streamlit Cloud app settings."
    )
    st.stop()

embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
vectorstore = FAISS.from_texts(split_docs, embeddings)

# --------------------------
# Simple NLP Preprocessing
# --------------------------
def simple_nlp(text):
    return " ".join(text.lower().strip().split())

# --------------------------
# Simple Agent using OpenAI LLM
# --------------------------
class SimpleAIAgent:
    def __init__(self, name="AI Agent"):
        self.name = name
        self.llm = ChatOpenAI(
            model_name="gpt-3.5-turbo",
            temperature=0.3,
            openai_api_key=OPENAI_API_KEY
        )
    def respond(self, prompt):
        return self.llm.predict(prompt)

# --------------------------
# Agentic AI (delegates tasks)
# --------------------------
class AgenticAI:
    def __init__(self, agent):
        self.agent = agent
    def handle_query(self, query, context):
        final_prompt = f"{strong_prompt}\nContext: {context}\nQuestion: {query}\nAnswer:"
        return self.agent.respond(final_prompt)

# --------------------------
# LangGraph Simple Simulation
# --------------------------
class LangGraph:
    def __init__(self):
        self.nodes = []
    def add_node(self, name, result):
        self.nodes.append({"node": name, "result": result})
    def show_graph(self):
        st.write("### LangGraph Execution Trace")
        for n in self.nodes:
            st.write(f"Node: {n['node']} -> Result: {n['result'][:100]}...")

# --------------------------
# Streamlit UI
# --------------------------
st.title("MANISH SINGH - AI-Driven Data Center Assistant with Agent & LangGraph")

selected_use_case = st.selectbox("Select a Use Case:", use_cases)
user_query = st.text_input("Ask your question about the use case:")

if user_query:
    clean_query = simple_nlp(user_query)
    
    # Retrieve relevant docs from FAISS
    docs = vectorstore.similarity_search(clean_query, k=2)
    context = " ".join([doc.page_content for doc in docs])

    # Initialize Agent & Agentic AI
    ai_agent = SimpleAIAgent()
    agentic_ai = AgenticAI(ai_agent)

    # Get AI response
    answer = agentic_ai.handle_query(clean_query, context)

    # Log in LangGraph
    lg = LangGraph()
    lg.add_node("Retrieve Context", context)
    lg.add_node("AI Response", answer)

    # Display AI answer
    st.write("### AI Answer:")
    st.write(answer)
    
    # Display LangGraph trace
    lg.show_graph()
