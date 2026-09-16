"""
Geo-Spatial Context Resolver for Darukaa.Earth Biodiversity Intelligence System.
Resolves GPS coordinates (lat/long) into Agro-Ecological Zones (AEZ), baseline rainfall,
soil classifications, and native ecosystem species.
"""

import os
import json
from typing import Optional, Dict, Any


class GeoSpatialResolver:
    """
    Maps geographical coordinates (latitude, longitude) or regional descriptors
    into validated agro-ecological zones with baseline environmental metrics.
    """

    def __init__(self, aez_dataset_path: Optional[str] = None):
        if aez_dataset_path is None:
            aez_dataset_path = os.path.join(
                os.path.dirname(os.path.dirname(__file__)),
                "knowledge_base",
                "raw_documents",
                "agro_ecological_zones.json"
            )
        self.aez_data = []
        if os.path.exists(aez_dataset_path):
            try:
                with open(aez_dataset_path, "r", encoding="utf-8") as f:
                    self.aez_data = json.load(f)
            except Exception as e:
                print(f"Error loading AEZ dataset: {e}")

    def resolve_coordinates(self, lat: float, lon: float) -> Dict[str, Any]:
        """
        Matches coordinates to known agro-ecological bounding zones or nearest biome.
        """
        for zone in self.aez_data:
            bbox = zone.get("bounding_box_sample")
            if bbox:
                if (bbox["lat_min"] <= lat <= bbox["lat_max"] and
                    bbox["lon_min"] <= lon <= bbox["lon_max"]):
                    return {
                        "matched": True,
                        "zone_id": zone["zone_id"],
                        "zone_name": zone["name"],
                        "estimated_annual_rainfall_mm": f"{zone['rainfall_range_mm'][0]}-{zone['rainfall_range_mm'][1]} mm",
                        "rainfall_pattern": "semi-arid" if zone['rainfall_range_mm'][1] < 700 else "seasonal_moderate",
                        "mean_temperature_c": f"{zone['mean_temp_celsius'][0]}-{zone['mean_temp_celsius'][1]} °C",
                        "predominant_soils": zone["predominant_soils"],
                        "flagship_restoration_species": zone.get("flagship_restoration_species", []),
                        "degradation_risks": zone.get("primary_degradation_risks", []),
                        "coordinates": {"latitude": lat, "longitude": lon}
                    }

        # Fallback approximation based on latitude and general global biomes
        if 15.0 <= abs(lat) <= 35.0:
            default_zone = self.aez_data[0] if self.aez_data else {}
            zone_name = "Subtropical / Semi-Arid Dryland Transition"
            rf = [350, 700]
        elif abs(lat) < 15.0:
            zone_name = "Tropical Agro-Forestry Belt"
            rf = [800, 1500]
        else:
            zone_name = "Temperate Agricultural Zone"
            rf = [500, 900]

        return {
            "matched": False,
            "zone_id": "AEZ_GENERAL_APPROXIMATION",
            "zone_name": zone_name,
            "estimated_annual_rainfall_mm": f"{rf[0]}-{rf[1]} mm",
            "rainfall_pattern": "low-to-moderate" if rf[1] < 700 else "adequate",
            "predominant_soils": ["Mineral soils with variable organic matter"],
            "flagship_restoration_species": ["Native nitrogen-fixing legumes", "Local deep-rooted shrubs"],
            "coordinates": {"latitude": lat, "longitude": lon}
        }

    def resolve_by_region_keyword(self, keyword: str) -> Optional[Dict[str, Any]]:
        """Matches a string like 'semi-arid', 'deccan', 'punjab', or 'temperate' to AEZ."""
        kw = keyword.lower()
        for zone in self.aez_data:
            if kw in zone.get("name", "").lower() or kw in zone.get("zone_id", "").lower():
                return zone
            for reg in zone.get("typical_regions", []):
                if kw in reg.lower():
                    return zone
        return None


if __name__ == "__main__":
    resolver = GeoSpatialResolver()
    # Test Rajasthan, India (Semi-Arid)
    res = resolver.resolve_coordinates(26.9124, 75.7873)
    print("Resolved coordinates:", res["zone_name"])
    print("Rainfall:", res["estimated_annual_rainfall_mm"])
    print("Species:", res["flagship_restoration_species"])
