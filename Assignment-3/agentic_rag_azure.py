# agentic_rag_azure.py
"""
Agentic RAG System using LangGraph + Azure OpenAI + Pinecone + MLflow
Workflow:
  Retriever → Answer → Critique → (Refine if needed) → Output
"""

import os, json, argparse
from datetime import datetime
import mlflow
import pinecone
from typing import List, Dict, Optional

from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END
from pydantic import BaseModel, Field

# ---------- State Definition ----------
class RAGState(BaseModel):
    query: str
    retrieved: List[Dict] = Field(default_factory=list)
    initial_answer: Optional[str] = None
    critique: Optional[str] = None
    refined_answer: Optional[str] = None

# ---------- Utils ----------
def pinecone_index():
    pinecone.init(api_key=os.environ["PINECONE_API_KEY"], environment=os.environ["PINECONE_ENV"])
    return pinecone.Index(os.environ["PINECONE_INDEX"])

def embed_query(text):
    embeddings = AzureOpenAIEmbeddings(
        azure_deployment=os.environ["AZURE_EMBED_DEPLOYMENT"],
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        api_key=os.environ["AZURE_OPENAI_KEY"],
    )
    return embeddings.embed_query(text)

def snippets_to_text(snippets):
    return "\n\n".join([f"[{s['id']}] {s['metadata'].get('title','')}\n{s['metadata'].get('text','')}" for s in snippets])

def chat_model():
    return AzureChatOpenAI(
        azure_deployment=os.environ["AZURE_CHAT_DEPLOYMENT"],
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        api_key=os.environ["AZURE_OPENAI_KEY"],
        temperature=0.0,
    )

# ---------- Node Implementations ----------
def retriever_node(state: RAGState) -> RAGState:
    index = pinecone_index()
    vec = embed_query(state.query)
    res = index.query(vector=vec, top_k=5, include_metadata=True)
    state.retrieved = [{"id": m.id, "score": m.score, "metadata": m.metadata} for m in res.matches]
    return state

def answer_node(state: RAGState) -> RAGState:
    llm = chat_model()
    sys = SystemMessage(content="You are an expert assistant. Always cite KB ids like [KB001].")
    user = HumanMessage(content=f"KB:\n{snippets_to_text(state.retrieved)}\n\nUser Query: {state.query}\n\nAnswer:")
    state.initial_answer = llm([sys, user]).content
    return state

def critique_node(state: RAGState) -> RAGState:
    llm = chat_model()
    sys = SystemMessage(content="You are a reviewer. Reply only COMPLETE or REFINE.")
    user = HumanMessage(content=f"Query: {state.query}\n\nAnswer: {state.initial_answer}\n\nKB:\n{snippets_to_text(state.retrieved)}")
    resp = llm([sys, user]).content.strip().upper()
    state.critique = "COMPLETE" if "COMPLETE" in resp else "REFINE"
    return state

def refine_node(state: RAGState) -> RAGState:
    if state.critique == "COMPLETE":
        return state
    index = pinecone_index()
    vec = embed_query(state.query)
    res = index.query(vector=vec, top_k=10, include_metadata=True)
    existing_ids = {m["id"] for m in state.retrieved}
    extra = next(({"id": m.id, "score": m.score, "metadata": m.metadata} for m in res.matches if m.id not in existing_ids), None)
    if extra:
        new_retrieved = state.retrieved + [extra]
        llm = chat_model()
        sys = SystemMessage(content="You are an expert assistant. Always cite KB ids like [KB001].")
        user = HumanMessage(content=f"KB:\n{snippets_to_text(new_retrieved)}\n\nUser Query: {state.query}\n\nAnswer:")
        state.refined_answer = llm([sys, user]).content
    return state

# ---------- Graph ----------
def build_graph():
    graph = StateGraph(RAGState)
    graph.add_node("retriever", retriever_node)
    graph.add_node("answer", answer_node)
    graph.add_node("critique", critique_node)
    graph.add_node("refine", refine_node)

    graph.set_entry_point("retriever")
    graph.add_edge("retriever", "answer")
    graph.add_edge("answer", "critique")

    def route_decision(state: RAGState):
        return "refine" if state.critique == "REFINE" else END

    graph.add_conditional_edges("critique", route_decision, {"refine": "refine", END: END})
    graph.add_edge("refine", END)
    return graph.compile()

# ---------- Runner ----------
def run_agentic_rag(query: str):
    graph = build_graph()
    init_state = RAGState(query=query)
    final_state = graph.invoke(init_state)

    result = final_state.dict()
    result["timestamp"] = datetime.utcnow().isoformat() + "Z"

    with mlflow.start_run(run_name="agentic_rag_langgraph"):
        mlflow.log_param("query", query)
        mlflow.log_param("decision", result.get("critique"))
        mlflow.log_dict(result, "result.json")

    fname = f"rag_result_{datetime.utcnow().strftime('%Y%m%dT%H%M%S')}.json"
    with open(fname, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    return result

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--query", required=True)
    args = p.parse_args()
    res = run_agentic_rag(args.query)
    print(f"Done. Critique={res['critique']}")

if __name__ == "__main__":
    main()
