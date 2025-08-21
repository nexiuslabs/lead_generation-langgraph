# app/main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from langserve import add_routes
from app.pre_sdr_graph import build_presdr_graph

app = FastAPI(title="Pre-SDR LangGraph Server")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Agent Chat UI dev
    allow_credentials=True, allow_methods=["*"], allow_headers=["*"]
)

graph = build_presdr_graph()
add_routes(app, graph, path="/agent")   # Assistant/Graph ID = "agent"

@app.get("/health")
def health(): return {"ok": True}
