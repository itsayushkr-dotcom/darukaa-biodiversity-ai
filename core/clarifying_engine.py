"""
Clarifying Engine for Darukaa.Earth Biodiversity Intelligence System.
Detects incomplete environmental data and generates scientifically targeted follow-up questions
when fewer than 3 environmental variables are provided.
"""

from typing import Dict, Any, Optional, Tuple, List
from core.schemas import ClarifyingQuestion
from core.parameter_extractor import ParameterExtractor


class ClarifyingEngine:
    """
    Evaluates ecological parameter completeness and generates scientifically rigorous
    clarifying questions before recommendations are generated.
    """

    @classmethod
    def evaluate_completeness(cls, params: Dict[str, Any]) -> Tuple[bool, Optional[ClarifyingQuestion]]:
        """
        Returns (is_complete, clarifying_question_if_incomplete).
        Completeness requires at least 3 distinct environmental dimensions.
        """
        dim_count = ParameterExtractor.count_distinct_environmental_dimensions(params)

        if dim_count >= 3:
            return True, None

        # Determine missing critical dimensions
        missing_vars: List[str] = []
        clarify_prompts: List[str] = []

        if not any(k in params for k in ["soil_organic_carbon_pct", "soil_ph", "soil_moisture"]):
            missing_vars.append("soil_organic_carbon_%_or_pH")
            clarify_prompts.append("soil organic carbon % (or pH)")

        if not any(k in params for k in ["annual_rainfall_mm", "rainfall_pattern", "region"]):
            missing_vars.append("rainfall_pattern_or_annual_precipitation")
            clarify_prompts.append("rainfall pattern (e.g., low, seasonal, mm/year)")

        if not any(k in params for k in ["crop", "cropping_system", "land_use_type"]):
            missing_vars.append("current_crop_or_land_use_type")
            clarify_prompts.append("land use type or current crop system (e.g., monoculture wheat, pasture, fallow)")

        # Fallback if only one or two were picked
        if len(clarify_prompts) < 2:
            if "tillage_practice" not in params:
                missing_vars.append("tillage_intensity")
                clarify_prompts.append("tillage practice (conventional vs no-till)")

        # Formulate scientific question matching the exact hackathon standard
        question_text = (
            f"To diagnose the ecological decline scientifically and model multi-variable interactions, "
            f"could you please provide:\n"
            f"• **{', '.join(clarify_prompts[:3])}**?"
        )

        rationale = (
            "Multi-metric ecological modeling requires cross-coupling between soil biogeochemistry "
            "(e.g., SOC/pH), hydrologic regime (rainfall), and vegetation structure (land use/crop) "
            "to prevent generic or misleading recommendations."
        )

        return False, ClarifyingQuestion(
            question=question_text,
            missing_variables=missing_vars,
            scientific_rationale=rationale
        )


if __name__ == "__main__":
    # Test with ambiguous input
    sample_params = {"biodiversity_status": "declining"}
    is_ready, question = ClarifyingEngine.evaluate_completeness(sample_params)
    print("Ready:", is_ready)
    if question:
        print("\nClarifying Question:\n", question.question)
        print("\nRationale:\n", question.scientific_rationale)
