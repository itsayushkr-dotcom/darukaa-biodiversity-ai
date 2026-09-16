"""
Pydantic Schemas for Darukaa.Earth Biodiversity Intelligence System.
Defines input payloads, environmental variables, multi-metric causal links, and structured outputs.
"""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field


class GeoCoordinates(BaseModel):
    latitude: float = Field(..., ge=-90.0, le=90.0, description="Latitude in decimal degrees")
    longitude: float = Field(..., ge=-180.0, le=180.0, description="Longitude in decimal degrees")
    location_name: Optional[str] = Field(None, description="Human readable name or district")


class EnvironmentalVariable(BaseModel):
    category: str = Field(..., description="soil | climate | land_use | biodiversity | human_impact")
    name: str = Field(..., description="Variable name e.g. soil_organic_carbon_pct, rainfall_pattern")
    value: Any = Field(..., description="Value extracted or provided")
    unit: Optional[str] = Field(None, description="e.g. %, mm, pH units")
    confidence: float = Field(default=1.0, description="Confidence of extraction 0.0-1.0")


class StructuredEnvironmentalInput(BaseModel):
    # Soil Metrics
    soil_organic_carbon_pct: Optional[float] = Field(None, ge=0.0, le=10.0, description="Soil Organic Carbon (SOC) percentage")
    soil_ph: Optional[float] = Field(None, ge=0.0, le=14.0, description="Soil pH level")
    soil_moisture: Optional[str] = Field(None, description="low | moderate | high | arid")
    soil_texture: Optional[str] = Field(None, description="sandy | loam | clay | silt")

    # Climate & Water
    annual_rainfall_mm: Optional[float] = Field(None, ge=0.0, description="Annual rainfall in mm")
    rainfall_pattern: Optional[str] = Field(None, description="low | erratic | semi-arid | seasonal | heavy")
    mean_temperature_c: Optional[float] = Field(None, description="Mean temperature in Celsius")

    # Land Use & Vegetation
    land_use_type: Optional[str] = Field(None, description="cropland | pasture | orchard | degraded | forest_fringe")
    crop: Optional[str] = Field(None, description="e.g. monoculture wheat, cotton, rice, fallow")
    cropping_system: Optional[str] = Field(None, description="monoculture | polyculture | agroforestry | rotational")
    tillage_practice: Optional[str] = Field(None, description="conventional_deep_tillage | minimum_till | zero_till")

    # Biodiversity & Ecosystem
    pollinator_activity: Optional[str] = Field(None, description="severely_depleted | moderate | healthy")
    species_decline_symptoms: Optional[str] = Field(None, description="Description of declining flora/fauna")

    # Human Impact
    chemical_intensity: Optional[str] = Field(None, description="high_synthetic | moderate | organic | regenerative")

    # Spatial Context (Bonus)
    region: Optional[str] = Field(None, description="e.g. semi-arid, Mediterranean, Sahel, Deccan")
    coordinates: Optional[GeoCoordinates] = None


class ChatMessage(BaseModel):
    role: str = Field(..., description="user | assistant | system")
    content: str = Field(...)


class MetricImpact(BaseModel):
    metric_name: str
    projected_improvement: str
    time_horizon: str = Field(..., description="short-term (0-6 mo) | medium-term (1-3 yrs) | long-term (3-5+ yrs)")
    confidence_level: str = Field(default="High", description="High | Medium | Moderate")


class ScientificCitation(BaseModel):
    source: str
    title: str
    year: int
    document_id: Optional[str] = None


class EnvironmentalIntervention(BaseModel):
    title: str
    what_to_do: str
    why_it_works: str
    environmental_variables_coupled: List[str] = Field(
        ...,
        min_length=3,
        description="Must connect at least 3 environmental variables together"
    )
    impacted_metrics: List[MetricImpact]
    scientific_citations: List[ScientificCitation]
    time_horizon_summary: str
    confidence_score: float = Field(default=0.88, ge=0.0, le=1.0)


class ClarifyingQuestion(BaseModel):
    question: str
    missing_variables: List[str]
    scientific_rationale: str


class IntelligenceResponse(BaseModel):
    status: str = Field(..., description="'RECOMMENDATION_READY' or 'NEEDS_CLARIFICATION'")
    user_query: str
    extracted_variables: Dict[str, Any]
    detected_variable_count: int
    clarifying_question: Optional[ClarifyingQuestion] = None
    interventions: List[EnvironmentalIntervention] = Field(default_factory=list)
    multi_variable_causal_chain: Optional[str] = None
    overall_scientific_summary: Optional[str] = None
    retrieved_knowledge_chunks: List[Dict[str, Any]] = Field(default_factory=list)
    spatial_context_resolved: Optional[Dict[str, Any]] = None
