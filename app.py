import streamlit as st
import yaml
import os
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langgraph.graph import StateGraph, END
from langchain.agents import Tool, AgentExecutor
from langchain.base_language import BaseLanguageModel
import openai

# -------------------------------
# Load secrets.yaml
# -------------------------------
if not os.path.exists("secrets.yaml"):
    st.error("secrets.yaml not found! Provide secrets.yaml with OpenAI API key and FAISS path.")
    st.stop()

with open("secrets.yaml", "r") as f:
    secrets = yaml.safe_load(f)

OPENAI_API_KEY = secrets.get("openai_api_key")
VECTORSTORE_PATH = secrets.get("faiss_index_path", "faiss_index")

if not OPENAI_API_KEY:
    st.error("API key not found in secrets.yaml (openai_api_key)")
    st.stop()

# Set OpenAI API key
openai.api_key = OPENAI_API_KEY

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
# API-based LLM Wrapper
# -------------------------------
class APILLM(BaseLanguageModel):
    @property
    def _llm_type(self):
        return "openai_api"

    def _call(self, prompt: str, stop=None) -> str:
        response = openai.Completion.create(
            engine="text-davinci-003",
            prompt=prompt,
            temperature=0.3,
            max_tokens=200
        )
        return response.choices[0].text.strip()

llm_api = APILLM()

# -------------------------------
# FAISS Knowledge Store using OpenAI Embeddings
# -------------------------------
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

if os.path.exists(VECTORSTORE_PATH):
    vectordb = FAISS.load_local(VECTORSTORE_PATH, embeddings)
else:
    st.warning("FAISS index not found, creating a new one...")
    splitter = CharacterTextSplitter(chunk_size=200, chunk_overlap=0)
    docs = splitter.create_documents([data_center_knowledge])
    vectordb = FAISS.from_documents(docs, embeddings)
    vectordb.save_local(VECTORSTORE_PATH)
    st.success("FAISS index created.")

retriever = vectordb.as_retriever()
qa = RetrievalQA.from_chain_type(llm=llm_api, retriever=retriever)

# -------------------------------
# Agent Tool
# -------------------------------
def agent_1_tool(task: str) -> str:
    template = """
    Agent Tool: Analyze the task using RAG + LLM reasoning and provide key insights.
    Task: {task}
    """
    prompt = PromptTemplate(template=template, input_variables=["task"])
    return llm_api(prompt.format(task=task))

agent_1 = Tool(
    name="Agent_1_API_Tool",
    func=agent_1_tool,
    description="Analyzes the use case and summarizes key points with RAG + API-based LLM."
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
        executor = AgentExecutor.from_tools([agent_1], llm=llm_api, verbose=False)
        state["agent_result"] = executor.run(query)
        return state

    def agentic_summary(state):
        summary_prompt = f"Summarize actionable insights from RAG and AgentExecutor outputs:\n\nRAG:\n{state['rag_result']}\n\nAgentExecutor:\n{state['agent_result']}"
        state["summary_result"] = llm_api(summary_prompt)
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
st.title("MANISH SINGH - AI-Driven Data Center Insights (OpenAI Embeddings + API LLM)")
st.write("Select a Data Center Use Case and generate insights using RAG + AgentExecutor + API-based LLM.")

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
