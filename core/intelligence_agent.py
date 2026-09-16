"""
Main Orchestration Intelligence Agent for Darukaa.Earth.
Combines conversational memory, parameter extraction, clarifying engine,
RAG vector retrieval, and multi-metric reasoning into an AI Environmental Scientist.
"""

import sys
import os
import json
from typing import Dict, Any, List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.schemas import (
    IntelligenceResponse,
    ChatMessage,
    StructuredEnvironmentalInput,
    ClarifyingQuestion
)
from core.parameter_extractor import ParameterExtractor
from core.clarifying_engine import ClarifyingEngine
from core.reasoning_engine import MultiMetricReasoningEngine
from core.geo_spatial import GeoSpatialResolver
from knowledge_base.vector_store import EnvironmentalVectorStore


class BiodiversityIntelligenceAgent:
    """
    Autonomous AI Environmental Scientist.
    Manages multi-turn memory, extracts ecological metrics, handles clarifying inquiries,
    and constructs multi-variable evidence-based restoration interventions.
    """

    def __init__(
        self,
        gemini_api_key: Optional[str] = None,
        vector_store: Optional[EnvironmentalVectorStore] = None
    ):
        self.vector_store = vector_store or EnvironmentalVectorStore()
        self.reasoning_engine = MultiMetricReasoningEngine(vector_store=self.vector_store)
        self.geo_resolver = GeoSpatialResolver()
        self.gemini_api_key = gemini_api_key or os.environ.get("GEMINI_API_KEY")
        self.conversation_memory: List[ChatMessage] = []
        self.accumulated_params: Dict[str, Any] = {}

    def reset_memory(self):
        """Clears dialog memory and accumulated environmental state."""
        self.conversation_memory = []
        self.accumulated_params = {}

    def chat(
        self,
        user_message: str,
        structured_input: Optional[StructuredEnvironmentalInput] = None,
        coordinates: Optional[Dict[str, float]] = None
    ) -> IntelligenceResponse:
        """
        Processes a conversational message in a multi-turn context.
        """
        # 1. Record user turn in memory
        self.conversation_memory.append(ChatMessage(role="user", content=user_message))

        # 2. Extract parameters from the latest text
        new_params = ParameterExtractor.extract_from_text(user_message)

        # 3. Merge with any structured input provided
        merged = ParameterExtractor.merge_inputs(new_params, structured_input)

        # 4. Handle coordinates if provided in call or in structured_input
        coords = coordinates
        if not coords and structured_input and structured_input.coordinates:
            coords = {
                "latitude": structured_input.coordinates.latitude,
                "longitude": structured_input.coordinates.longitude
            }

        if coords:
            spatial_res = self.geo_resolver.resolve_coordinates(coords["latitude"], coords["longitude"])
            if "region" not in merged:
                merged["region"] = spatial_res.get("zone_name")
            if "rainfall_pattern" not in merged:
                merged["rainfall_pattern"] = spatial_res.get("rainfall_pattern")

        # 5. Accumulate into agent state (multi-turn memory)
        self.accumulated_params.update(merged)

        # 6. Evaluate completeness across 5 ecological dimensions
        is_complete, clarifying_q = ClarifyingEngine.evaluate_completeness(self.accumulated_params)

        if not is_complete and clarifying_q:
            # Under constraint: < 3 environmental variables, must ask clarifying questions
            response = IntelligenceResponse(
                status="NEEDS_CLARIFICATION",
                user_query=user_message,
                extracted_variables=self.accumulated_params,
                detected_variable_count=len(self.accumulated_params),
                clarifying_question=clarifying_q,
                interventions=[],
                overall_scientific_summary=(
                    "Incomplete environmental profile. An environmental scientist requires at least 3 "
                    "coupled variables (Soil health, Climate/Water regime, and Land use) to generate "
                    "scientifically grounded, non-obvious recommendations."
                ),
                retrieved_knowledge_chunks=[]
            )
            # Record assistant turn
            self.conversation_memory.append(ChatMessage(role="assistant", content=clarifying_q.question))
            return response

        # 7. When complete (>= 3 variables), execute Multi-Metric Reasoning
        response = self.reasoning_engine.reason(
            user_query=user_message,
            params=self.accumulated_params,
            coordinates=coords
        )

        # Record assistant answer in memory
        self.conversation_memory.append(
            ChatMessage(role="assistant", content=response.overall_scientific_summary or "Recommendation formulated.")
        )
        return response


if __name__ == "__main__":
    agent = BiodiversityIntelligenceAgent()
    print("--- TURN 1: Incomplete Query ---")
    r1 = agent.chat("Biodiversity is declining on my land")
    print("STATUS:", r1.status)
    if r1.clarifying_question:
        print("AGENT ASKS:\n", r1.clarifying_question.question)

    print("\n--- TURN 2: User provides parameters (multi-turn memory test) ---")
    r2 = agent.chat("Soil organic carbon: 0.3%, Rainfall: low, Crop: monoculture wheat, Region: semi-arid")
    print("STATUS:", r2.status)
    print("INTERVENTIONS GENERATED:", len(r2.interventions))
    for it in r2.interventions:
        print(f"• {it.title} (Coupled: {it.environmental_variables_coupled})")
