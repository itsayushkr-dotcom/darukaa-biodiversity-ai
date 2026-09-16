"""
Parameter Extractor & Entity Recognizer for Environmental Variables.
Parses natural language queries to extract soil metrics, climate variables, land use types,
biodiversity indicators, and management practices.
"""

import re
from typing import Dict, Any, Optional
from core.schemas import StructuredEnvironmentalInput


class ParameterExtractor:
    """
    Extracts structured environmental parameters from natural language user queries
    using regex and ecological semantic patterns.
    """

    @classmethod
    def extract_from_text(cls, text: str) -> Dict[str, Any]:
        extracted: Dict[str, Any] = {}
        text_lower = text.lower()

        # 1. Soil Organic Carbon (SOC)
        soc_match = re.search(
            r'(?:soil\s+(?:organic\s+)?carbon|soc|organic\s+carbon|carbon)\s*(?:is|was|=|:|at|around|\s)\s*([0-9]+(?:\.[0-9]+)?)\s*%',
            text_lower
        )
        if soc_match:
            extracted["soil_organic_carbon_pct"] = float(soc_match.group(1))
        elif "low soil carbon" in text_lower or "depleted carbon" in text_lower or "low organic carbon" in text_lower:
            extracted["soil_organic_carbon_pct"] = 0.4  # typical depleted baseline

        # 2. Soil pH
        ph_match = re.search(
            r'(?:soil\s*)?ph\s*(?:is|was|=|:|at|around|\s)\s*([0-9]+(?:\.[0-9]+)?)',
            text_lower
        )
        if ph_match:
            extracted["soil_ph"] = float(ph_match.group(1))
        elif "alkaline" in text_lower or "sodic" in text_lower:
            extracted["soil_ph"] = 8.6
        elif "acidic" in text_lower:
            extracted["soil_ph"] = 5.2

        # 3. Rainfall / Climate
        rainfall_mm = re.search(
            r'(?:annual\s+)?(?:rainfall|precipitation)\s*(?:is|was|=|:|at|around|\s)\s*([0-9]+)\s*(?:mm)?',
            text_lower
        )
        if rainfall_mm:
            extracted["annual_rainfall_mm"] = float(rainfall_mm.group(1))
            extracted["rainfall_pattern"] = "low" if float(rainfall_mm.group(1)) < 600 else "adequate"
        else:
            rf_match = re.search(
                r'(?:annual\s+)?rainfall\s*(?:is|was|=|:|at|around|\s)\s*(low|moderate|high|erratic|seasonal)',
                text_lower
            )
            if rf_match:
                extracted["rainfall_pattern"] = rf_match.group(1)
            elif "low rainfall" in text_lower or "drought" in text_lower or "water scarce" in text_lower or "rainfall is low" in text_lower:
                extracted["rainfall_pattern"] = "low"
            elif "erratic rainfall" in text_lower or "irregular rain" in text_lower:
                extracted["rainfall_pattern"] = "erratic"

        # 4. Region / Biome / Climate zone
        region_match = re.search(r'region[:\s]+([a-zA-Z\-\s]+?)(?:,|$|\n|\.)', text_lower)
        if region_match:
            extracted["region"] = region_match.group(1).strip()
        elif "semi-arid" in text_lower or "semi arid" in text_lower:
            extracted["region"] = "semi-arid"
        elif "arid" in text_lower:
            extracted["region"] = "arid"
        elif "tropical" in text_lower:
            extracted["region"] = "tropical"
        elif "temperate" in text_lower:
            extracted["region"] = "temperate"

        # 5. Crop & Monoculture
        crop_match = re.search(r'crop[:\s]+([a-zA-Z\-\s]+?)(?:,|$|\n|\.)', text_lower)
        if crop_match:
            crop_val = crop_match.group(1).strip()
            extracted["crop"] = crop_val
            if "monoculture" in crop_val:
                extracted["cropping_system"] = "monoculture"
        else:
            # Common crops check
            crops_found = []
            for crop_name in ["wheat", "rice", "paddy", "cotton", "maize", "corn", "soybean", "sugarcane", "barley", "millet"]:
                if crop_name in text_lower:
                    crops_found.append(crop_name)
            if crops_found:
                extracted["crop"] = ", ".join(crops_found)
            if "monoculture" in text_lower:
                extracted["cropping_system"] = "monoculture"
                if "crop" in extracted and "monoculture" not in extracted["crop"]:
                    extracted["crop"] = f"monoculture {extracted['crop']}"

        # 6. Land Use / Soil Moisture / Tillage
        if "tillage" in text_lower or "plowing" in text_lower:
            if "no-till" in text_lower or "zero till" in text_lower:
                extracted["tillage_practice"] = "zero_till"
            else:
                extracted["tillage_practice"] = "conventional_deep_tillage"

        if "dry soil" in text_lower or "low moisture" in text_lower or "dryland" in text_lower:
            extracted["soil_moisture"] = "low"

        # 7. Biodiversity Status
        if "biodiversity is declining" in text_lower or "declining biodiversity" in text_lower or "loss of species" in text_lower:
            extracted["biodiversity_status"] = "declining"
        if "pollinator" in text_lower or "bee" in text_lower:
            extracted["pollinator_activity"] = "severely_depleted"

        # 8. Human Impact & Chemical Pollution (Pillar 5)
        if any(w in text_lower for w in ["pesticide", "chemical", "fertilizer overuse", "nitrate", "runoff", "pollution"]):
            extracted["chemical_intensity"] = "high_synthetic"
        if any(w in text_lower for w in ["deforest", "forest clearing", "cleared land", "edge effect", "logging"]):
            extracted["deforestation_history"] = "severe_fragmentation"

        return extracted

    @classmethod
    def merge_inputs(
        cls,
        text_extracted: Dict[str, Any],
        structured_input: Optional[StructuredEnvironmentalInput] = None
    ) -> Dict[str, Any]:
        """Combines extracted text variables with explicitly provided structured inputs."""
        merged = dict(text_extracted)
        if structured_input:
            dumped = structured_input.model_dump(exclude_none=True)
            for k, v in dumped.items():
                if v is not None:
                    merged[k] = v
        return merged

    @classmethod
    def count_distinct_environmental_dimensions(cls, params: Dict[str, Any]) -> int:
        """
        Calculates how many distinct core environmental dimensions are represented:
        1. Soil (SOC %, pH, moisture, texture)
        2. Climate/Water (rainfall, temperature, aridity)
        3. Land Use/Crop (crop, monoculture, tillage, land_use_type)
        4. Biodiversity (pollinators, species decline, vegetative cover)
        5. Human Impact/Chemicals (pesticides, fertilizer, deforestation)
        """
        dimensions_present = set()

        # Dimension 1: Soil
        if any(k in params for k in ["soil_organic_carbon_pct", "soil_ph", "soil_moisture", "soil_texture"]):
            dimensions_present.add("soil_health")

        # Dimension 2: Climate & Water
        if any(k in params for k in ["annual_rainfall_mm", "rainfall_pattern", "mean_temperature_c", "region"]):
            dimensions_present.add("climate_water")

        # Dimension 3: Land Use
        if any(k in params for k in ["crop", "cropping_system", "land_use_type", "tillage_practice"]):
            dimensions_present.add("land_use")

        # Dimension 4: Biodiversity
        if any(k in params for k in ["biodiversity_status", "pollinator_activity", "species_decline_symptoms"]):
            dimensions_present.add("biodiversity")

        # Dimension 5: Human Impact
        if any(k in params for k in ["chemical_intensity"]):
            dimensions_present.add("human_impact")

        return len(dimensions_present)


if __name__ == "__main__":
    # Test example from problem statement
    sample = "Soil organic carbon: 0.3%, Rainfall: low, Crop: monoculture wheat, Region: semi-arid"
    res = ParameterExtractor.extract_from_text(sample)
    print("Extracted from sample:", res)
    dim_count = ParameterExtractor.count_distinct_environmental_dimensions(res)
    print(f"Distinct dimensions: {dim_count}")
