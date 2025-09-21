import streamlit as st
from langchain.prompts import PromptTemplate
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import RetrievalQA
from langgraph.graph import StateGraph, END
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from langchain.agents import Tool, AgentExecutor
from langchain.base_language import BaseLanguageModel
import torch
import os

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

# -------------------------------
# Load Local LLM (free model)
# -------------------------------
MODEL_NAME = "google/flan-t5-small"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
llm_pipeline = pipeline(
    "text2text-generation",
    model=model,
    tokenizer=tokenizer,
    device=0 if torch.cuda.is_available() else -1
)

def llm_generate(prompt: str) -> str:
    output = llm_pipeline(prompt, max_length=200, do_sample=True, temperature=0.3)
    return str(output[0]['generated_text'])

# -------------------------------
# Fixed LocalLLM for AgentExecutor
# -------------------------------
class LocalLLM(BaseLanguageModel):
    @property
    def _llm_type(self) -> str:
        return "local"

    def _call(self, prompt: str, stop=None) -> str:
        result = llm_generate(prompt)
        return str(result)

llm_local = LocalLLM()

# -------------------------------
# FAISS Persistence
# -------------------------------
VECTORSTORE_PATH = "faiss_index"
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

try:
    vectordb = FAISS.load_local(VECTORSTORE_PATH, embeddings)
except Exception:
    splitter = CharacterTextSplitter(chunk_size=200, chunk_overlap=0)
    docs = splitter.create_documents([data_center_knowledge])
    vectordb = FAISS.from_documents(docs, embeddings)
    vectordb.save_local(VECTORSTORE_PATH)

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

    # Step 1: Retrieve Knowledge
    def retrieve_knowledge(state):
        query = state["query"]
        state["rag_result"] = qa.run(query)
        return state

    # Step 2: Run AgentExecutor
    def run_agent_executor(state):
        query = state["query"]
        executor = AgentExecutor.from_tools([agent_1], llm=llm_local, verbose=False)
        state["agent_result"] = executor.run(query)
        return state

    # Step 3: Agentic AI Summary
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
st.title("MANISH SINGH - AI-Driven Data Center: AgentExecutor + Agentic AI")
st.write("RAG + NLP + LLM + Agentic AI + LangGraph + Persistent FAISS")

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

choice = st.selectbox("Choose a Data Center Use Case:", use_cases)

if st.button("Run Use Case"):
    workflow = build_langgraph_pipeline()
    result = workflow.invoke({"query": choice})

    st.subheader("🔎 RAG Retrieved Knowledge")
    st.write(result["rag_result"])

    st.subheader("🤖 AgentExecutor AI Response")
    st.write(result["agent_result"])

    st.subheader("🧠 Agentic AI Summary")
    st.write(result["summary_result"])

    st.subheader("⚙️ Hardware Integration Points")
    st.write("""
    Here you can connect to:
    - SNMP / IPMI sensors (temperature, fans, power usage)
    - REST APIs from cooling/UPS vendors
    - Prometheus / Grafana for monitoring
    - Orchestration tools (Kubernetes, VMware, OpenStack)
    """)
