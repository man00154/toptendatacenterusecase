import streamlit as st
import yaml
import os
from langchain.prompts import PromptTemplate
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import RetrievalQA
from langgraph.graph import StateGraph, END
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain.agents import Tool, AgentExecutor
from langchain.base_language import BaseLanguageModel

# -------------------------------
# Load secrets.yaml
# -------------------------------
if not os.path.exists("secrets.yaml"):
    st.error("secrets.yaml not found! Please provide secrets.yaml with model and FAISS paths.")
    st.stop()

with open("secrets.yaml", "r") as f:
    secrets = yaml.safe_load(f)

MODEL_PATH = secrets.get("local_llm_model_path")
VECTORSTORE_PATH = secrets.get("faiss_index_path", "faiss_index")

if not os.path.exists(MODEL_PATH):
    st.error(f"Local Gemini model not found at: {MODEL_PATH}")
    st.stop()

# -------------------------------
# Knowledge Base
# -------------------------------
data_center_knowledge = """
Use Cases of AI-Driven Data Centers:
1. Energy Optimization & Cooling
2. Predictive Maintenance
3. Automated Workload Management
4. Security & Threat Detection
5. Capacity Planning & Forecasting
6. Network Traffic Optimization
7. Data Center Digital Twin
8. Incident Response & Self-Healing
9. AI-Augmented Monitoring & Alerts
10. Sustainability & Green Initiatives
"""

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

# -------------------------------
# Load CPU-only Gemini LLM
# -------------------------------
try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH)
except Exception as e:
    st.error(f"Error loading local Gemini model: {e}")
    st.stop()

llm_pipeline = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    device=-1  # CPU only
)

def llm_generate(prompt: str) -> str:
    output = llm_pipeline(prompt, max_length=200, do_sample=True, temperature=0.3)
    return str(output[0]['generated_text'])

# -------------------------------
# Local LLM for AgentExecutor
# -------------------------------
class LocalLLM(BaseLanguageModel):
    @property
    def _llm_type(self) -> str:
        return "local"

    def _call(self, prompt: str, stop=None) -> str:
        return llm_generate(prompt)

llm_local = LocalLLM()

# -------------------------------
# FAISS Knowledge Store
# -------------------------------
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

try:
    vectordb = FAISS.load_local(VECTORSTORE_PATH, embeddings)
except Exception:
    st.warning("FAISS index not found, creating a new one...")
    splitter = CharacterTextSplitter(chunk_size=200, chunk_overlap=0)
    docs = splitter.create_documents([data_center_knowledge])
    vectordb = FAISS.from_documents(docs, embeddings)
    vectordb.save_local(VECTORSTORE_PATH)
    st.success("FAISS index created successfully.")

retriever = vectordb.as_retriever()
qa = RetrievalQA.from_chain_type(llm=llm_local, retriever=retriever)

# -------------------------------
# Agent Tool
# -------------------------------
def agent_1_tool(task: str) -> str:
    template = """
    Agent Tool: Analyze the task using NLP + RAG + LLM reasoning and provide key insights.
    Task: {task}
    """
    prompt = PromptTemplate(template=template, input_variables=["task"])
    return llm_generate(prompt.format(task=task))

agent_1 = Tool(
    name="Agent_1_NLP_Tool",
    func=agent_1_tool,
    description="Analyzes the use case and summarizes key points with NLP + RAG + LLM."
)

# -------------------------------
# LangGraph Workflow
# -------------------------------
def build_langgraph_pipeline():
    graph = StateGraph(dict)

    def retrieve_knowledge(state):
        query = state["query"]
        state["rag_result"] = qa.run(query)
        return state

    def run_agent_executor(state):
        query = state["query"]
        executor = AgentExecutor.from_tools([agent_1], llm=llm_local, verbose=False)
        state["agent_result"] = executor.run(query)
        return state

    def agentic_summary(state):
        summary_prompt = f"Summarize actionable hardware insights from RAG and AgentExecutor outputs:\n\nRAG:\n{state['rag_result']}\n\nAgentExecutor Output:\n{state['agent_result']}"
        state["summary_result"] = llm_generate(summary_prompt)
        return state

    graph.add_node("retriever", retrieve_knowledge)
    graph.add_node("agent_executor", run_agent_executor)
    graph.add_node("agentic_summary", agentic_summary)

    graph.set_entry_point("retriever")
    graph.add_edge("retriever", "agent_executor")
    graph.add_edge("agent_executor", "agentic_summary")
    graph.add_edge("agentic_summary", END)

    return graph.compile()

# -------------------------------
# Streamlit UI
# -------------------------------
st.title("MANISH SINGH - AI-Driven Data Center Use Case Insights (CPU-only)")
st.write("Select a Data Center Use Case and generate insights using RAG + AgentExecutor + Agentic AI.")

selected_use_case = st.selectbox("Choose a Data Center Use Case:", use_cases)

if st.button("Generate Insights") and selected_use_case.strip():
    workflow = build_langgraph_pipeline()
    result = workflow.invoke({"query": selected_use_case})

    st.subheader("🔎 RAG Retrieved Knowledge")
    st.write(result["rag_result"])

    st.subheader("🤖 AgentExecutor AI Response")
    st.write(result["agent_result"])

    st.subheader("🧠 Agentic AI Summary")
    st.write(result["summary_result"])

    st.subheader("⚙️ Hardware Integration Points")
    st.write("""
    Connect to:
    - SNMP / IPMI sensors (temperature, fans, power usage)
    - REST APIs from cooling/UPS vendors
    - Prometheus / Grafana for monitoring
    - Orchestration tools (Kubernetes, VMware, OpenStack)
    """)
