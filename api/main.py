"""
FastAPI REST API for Darukaa.Earth Biodiversity Intelligence System.
Exposes endpoints for Conversational Chat, Structured JSON Recommendations,
Spatial Coordinate Resolution, and Knowledge Base Vector Search.
"""

import sys
import os
from typing import Dict, Any, Optional
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.schemas import (
    IntelligenceResponse,
    StructuredEnvironmentalInput,
    GeoCoordinates
)
from core.intelligence_agent import BiodiversityIntelligenceAgent
from knowledge_base.vector_store import EnvironmentalVectorStore
from core.geo_spatial import GeoSpatialResolver

app = FastAPI(
    title="Darukaa.Earth AI Biodiversity Intelligence API",
    description="Scientific Reasoning & Multi-Metric RAG System for Ecosystem Restoration",
    version="1.0.0"
)

# Enable CORS for web clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory session store for multi-turn conversations
agent_sessions: Dict[str, BiodiversityIntelligenceAgent] = {}
shared_vector_store = EnvironmentalVectorStore()
geo_resolver = GeoSpatialResolver()


def get_or_create_agent(session_id: str) -> BiodiversityIntelligenceAgent:
    if session_id not in agent_sessions:
        agent_sessions[session_id] = BiodiversityIntelligenceAgent(vector_store=shared_vector_store)
    return agent_sessions[session_id]


class ChatRequest(BaseModel):
    message: str = Field(..., example="Biodiversity is declining on my land")
    session_id: str = Field(default="default_session", description="Session identifier for multi-turn memory")
    coordinates: Optional[GeoCoordinates] = None
    structured_overrides: Optional[StructuredEnvironmentalInput] = None


class KnowledgeSearchResponse(BaseModel):
    query: str
    total_results: int
    results: list


@app.get("/api/health")
def health_check():
    return {
        "status": "healthy",
        "service": "Darukaa.Earth AI Biodiversity Intelligence",
        "knowledge_base_documents": len(shared_vector_store.documents),
        "vocabulary_size": len(shared_vector_store.vocabulary),
    }


@app.post("/api/chat", response_model=IntelligenceResponse)
def chat_endpoint(request: ChatRequest):
    """
    Conversational endpoint supporting multi-turn dialogue with memory,
    incomplete input detection, and multi-metric scientific reasoning.
    """
    agent = get_or_create_agent(request.session_id)
    coords_dict = None
    if request.coordinates:
        coords_dict = {
            "latitude": request.coordinates.latitude,
            "longitude": request.coordinates.longitude
        }
    response = agent.chat(
        user_message=request.message,
        structured_input=request.structured_overrides,
        coordinates=coords_dict
    )
    return response


@app.post("/api/recommend", response_model=IntelligenceResponse)
def structured_recommend_endpoint(payload: StructuredEnvironmentalInput):
    """
    Direct structured JSON input endpoint. Bypasses chat parsing and directly
    models provided environmental variables into evidence-backed interventions.
    """
    # Create temporary single-turn agent
    agent = BiodiversityIntelligenceAgent(vector_store=shared_vector_store)
    # Synthesize synthetic query summary
    query_parts = []
    if payload.crop:
        query_parts.append(f"Crop: {payload.crop}")
    if payload.soil_organic_carbon_pct is not None:
        query_parts.append(f"Soil organic carbon: {payload.soil_organic_carbon_pct}%")
    if payload.annual_rainfall_mm is not None:
        query_parts.append(f"Rainfall: {payload.annual_rainfall_mm} mm")
    elif payload.rainfall_pattern:
        query_parts.append(f"Rainfall: {payload.rainfall_pattern}")
    if payload.region:
        query_parts.append(f"Region: {payload.region}")

    synthetic_query = ", ".join(query_parts) or "Structured environmental evaluation"
    response = agent.chat(user_message=synthetic_query, structured_input=payload)
    return response


@app.get("/api/knowledge/search", response_model=KnowledgeSearchResponse)
def search_knowledge_base(
    q: str = Query(..., description="Query terms to search scientific knowledge base"),
    top_k: int = Query(3, ge=1, le=10)
):
    """
    Direct inspection endpoint for RAG vector store and scientific literature.
    """
    results = shared_vector_store.search(query=q, top_k=top_k)
    return KnowledgeSearchResponse(
        query=q,
        total_results=len(results),
        results=results
    )


@app.post("/api/spatial/resolve")
def resolve_spatial_coordinates(coords: GeoCoordinates):
    """
    Resolves GPS coordinates into an Agro-Ecological Zone with baseline rainfall,
    soil orders, and flagship species.
    """
    return geo_resolver.resolve_coordinates(coords.latitude, coords.longitude)


@app.post("/api/session/reset")
def reset_session(session_id: str = Query("default_session")):
    if session_id in agent_sessions:
        agent_sessions[session_id].reset_memory()
    return {"message": f"Session '{session_id}' memory reset successfully."}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host="127.0.0.1", port=8000, reload=True)
