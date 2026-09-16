"""
Darukaa.Earth AI Biodiversity Intelligence Benchmark & Evaluation Suite.
Validates the system against all 5 evaluation criteria and constraints outlined in the challenge:
1. Depth of Reasoning (30%): Non-obvious recommendations, >= 3 variables coupled together.
2. Scientific Grounding (25%): Credible sources (FAO, IPCC, IPBES) and explainable mechanisms.
3. Knowledge System Design (20%): Vector store retrieval pipeline and structured dataset indexing.
4. Conversational Intelligence (15%): Clarifying questions on incomplete input, multi-turn memory.
5. Output Clarity (10%): Structured, readable, actionable outputs with quantified metrics.
"""

import sys
import os
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.intelligence_agent import BiodiversityIntelligenceAgent
from core.schemas import StructuredEnvironmentalInput, GeoCoordinates
from knowledge_base.vector_store import EnvironmentalVectorStore
from core.geo_spatial import GeoSpatialResolver
from core.parameter_extractor import ParameterExtractor
from core.clarifying_engine import ClarifyingEngine


class TestDarukaaBiodiversitySystem(unittest.TestCase):

    def setUp(self):
        self.vector_store = EnvironmentalVectorStore()
        self.agent = BiodiversityIntelligenceAgent(vector_store=self.vector_store)
        self.geo_resolver = GeoSpatialResolver()

    def test_criterion_1_and_2_primary_problem_statement_usecase(self):
        """
        Evaluation Criteria 1 (Depth of Reasoning - 30%) & 2 (Scientific Grounding - 25%):
        Test the exact use case from Page 3:
        Input: Soil organic carbon: 0.3%, Rainfall: low, Crop: monoculture wheat, Region: semi-arid
        Expected: Suggest agroforestry/intercropping, explain soil carbon & biodiversity impact,
                  provide measurable estimates, reference FAO & IPCC.
        """
        query = "Soil organic carbon: 0.3%, Rainfall: low, Crop: monoculture wheat, Region: semi-arid"
        res = self.agent.chat(query)

        self.assertEqual(res.status, "RECOMMENDATION_READY")
        self.assertGreaterEqual(len(res.interventions), 2)

        # Check for agroforestry / intercropping recommendation
        titles = [it.title.lower() for it in res.interventions]
        has_agroforestry_or_intercropping = any(
            "agroforestry" in t or "intercropping" in t or "cover cropping" in t for t in titles
        )
        self.assertTrue(has_agroforestry_or_intercropping, "Must recommend agroforestry or intercropping")

        # Check measurable improvement estimates
        all_metrics = []
        for it in res.interventions:
            for m in it.impacted_metrics:
                all_metrics.append(m.projected_improvement)
        has_quantified_estimates = any("%" in m for m in all_metrics)
        self.assertTrue(has_quantified_estimates, "Must provide measurable/quantified improvement estimates")

        # Check credible references (FAO, IPCC)
        all_sources = []
        for it in res.interventions:
            for c in it.scientific_citations:
                all_sources.append(c.source)
        sources_str = " ".join(all_sources)
        self.assertTrue("FAO" in sources_str or "Food and Agriculture" in sources_str, "Must cite FAO")
        self.assertTrue("IPCC" in sources_str or "Intergovernmental Panel" in sources_str, "Must cite IPCC")

    def test_criterion_3_knowledge_system_retrieval(self):
        """
        Evaluation Criteria 3 (Knowledge System Design - 20%):
        Verify vector retrieval matches relevant peer-reviewed literature and returns similarity scores.
        """
        docs = self.vector_store.search("soil organic carbon legume cover crops semi-arid", top_k=3)
        self.assertGreaterEqual(len(docs), 1)
        top = docs[0]
        self.assertIn("relevance_score", top)
        self.assertGreater(top["relevance_score"], 0.0)
        self.assertIn("quantified_impact", top)
        self.assertIn("scientific_mechanism", top)

    def test_criterion_4_incomplete_input_clarifying_question(self):
        """
        Evaluation Criteria 4 (Conversational Intelligence - 15%):
        Test exact example from Page 2:
        User: 'Biodiversity is declining on my land'
        System must ask clarifying questions for missing variables rather than giving generic advice.
        """
        agent = BiodiversityIntelligenceAgent(vector_store=self.vector_store)
        res = agent.chat("Biodiversity is declining on my land")

        self.assertEqual(res.status, "NEEDS_CLARIFICATION")
        self.assertIsNotNone(res.clarifying_question)
        self.assertGreater(len(res.clarifying_question.missing_variables), 0)

        # Verify no shallow interventions are produced
        self.assertEqual(len(res.interventions), 0)

        q_text = res.clarifying_question.question.lower()
        self.assertTrue(
            "soil" in q_text or "rainfall" in q_text or "crop" in q_text or "land use" in q_text,
            "Clarifying question must ask for soil, rainfall, or land use variables"
        )

    def test_criterion_4_multi_turn_conversation_memory(self):
        """
        Evaluation Criteria 4 (Conversational Intelligence - Memory):
        Verify that agent remembers context across turns and seamlessly resolves once complete.
        """
        agent = BiodiversityIntelligenceAgent(vector_store=self.vector_store)
        # Turn 1: Incomplete
        t1 = agent.chat("Biodiversity is declining on my land")
        self.assertEqual(t1.status, "NEEDS_CLARIFICATION")

        # Turn 2: Follow-up with missing parameters
        t2 = agent.chat("My soil carbon is 0.3%, annual rainfall is low, and I grow monoculture wheat")
        self.assertEqual(t2.status, "RECOMMENDATION_READY")
        self.assertGreaterEqual(len(t2.interventions), 2)
        # Verify accumulated parameters
        self.assertEqual(agent.accumulated_params.get("soil_organic_carbon_pct"), 0.3)
        self.assertEqual(agent.accumulated_params.get("crop"), "monoculture wheat")

    def test_constraint_multi_metric_coupling_at_least_3_variables(self):
        """
        Constraint from Page 4:
        'Must handle at least 3 environmental variables together.
         This is the core differentiator—no single-variable answers.'
        """
        agent = BiodiversityIntelligenceAgent(vector_store=self.vector_store)
        res = agent.chat("Soil organic carbon: 0.3%, Rainfall: low, Crop: monoculture wheat, Region: semi-arid")

        for it in res.interventions:
            coupled = it.environmental_variables_coupled
            self.assertGreaterEqual(
                len(coupled), 3,
                f"Intervention '{it.title}' must couple at least 3 environmental variables together!"
            )

    def test_bonus_geo_spatial_context_resolution(self):
        """
        Bonus Requirement (Page 2):
        'Bonus: Geo-coordinates or spatial context'
        """
        # Test coordinates for Northwestern India / Rajasthan (Semi-Arid)
        spatial = self.geo_resolver.resolve_coordinates(26.9124, 75.7873)
        self.assertTrue(spatial["matched"])
        self.assertEqual(spatial["zone_id"], "AEZ_SEMI_ARID")
    def test_structured_json_input_support(self):
        """
        Input Requirement (Page 2):
        'Support at least: Text input (mandatory), Structured input (JSON or similar)'
        """
        payload = StructuredEnvironmentalInput(
            soil_organic_carbon_pct=0.35,
            annual_rainfall_mm=380.0,
            rainfall_pattern="low",
            crop="monoculture wheat",
            region="semi-arid",
            coordinates=GeoCoordinates(latitude=26.9124, longitude=75.7873)
        )
        agent = BiodiversityIntelligenceAgent(vector_store=self.vector_store)
        res = agent.chat("Evaluate my land", structured_input=payload)
        self.assertEqual(res.status, "RECOMMENDATION_READY")
        self.assertGreaterEqual(len(res.interventions), 2)

    def test_pillar_5_human_impact_and_pollution_scenario(self):
        """
        Knowledge System Pillar 5: Human Impact (Pollution, Agrochemical Runoff, Deforestation).
        Verifies system diagnoses high chemical pesticide/fertilizer use and generates
        evidence-backed riparian bio-filtration and biochar sorption solutions.
        """
        query = "Soil organic carbon: 0.4%, Rainfall: low, Crop: monoculture cotton, High chemical pesticide and fertilizer runoff"
        agent = BiodiversityIntelligenceAgent(vector_store=self.vector_store)
        res = agent.chat(query)

        self.assertEqual(res.status, "RECOMMENDATION_READY")
        # Must generate 4 interventions (including riparian bio-filter)
        self.assertGreaterEqual(len(res.interventions), 3)

        has_pollution_remediation = any(
            "riparian" in it.title.lower() or "biochar" in it.title.lower() or "filter" in it.title.lower()
            for it in res.interventions
        )
        self.assertTrue(has_pollution_remediation, "Must formulate riparian bio-filter intervention for high chemical pollution")

        # Verify citation of UNEP / FAO Soil Pollution
        sources = [c.source for it in res.interventions for c in it.scientific_citations]
        self.assertTrue(any("UNEP" in s or "Environment Programme" in s for s in sources), "Must cite UNEP for soil pollution")

    def test_output_quality_confidence_and_time_horizon(self):
        """
        Output Quality Requirement (Page 2):
        'Each response must clearly include: Recommendation, Impacted metrics,
         Time horizon (short/medium/long term), Confidence level (optional but valuable)'
        """
        agent = BiodiversityIntelligenceAgent(vector_store=self.vector_store)
        res = agent.chat("Soil organic carbon: 0.3%, Rainfall: low, Crop: monoculture wheat, Region: semi-arid")

        for it in res.interventions:
            # Confidence score check
            self.assertGreaterEqual(it.confidence_score, 0.70)
            self.assertLessEqual(it.confidence_score, 1.0)

            # Time horizon summary check
            self.assertTrue(len(it.time_horizon_summary) > 10)

            # Impacted metrics check
            for m in it.impacted_metrics:
                self.assertIn(
                    m.time_horizon,
                    ["short-term (0-6 mo)", "medium-term (1-3 yrs)", "long-term (3-5+ yrs)"]
                )
                self.assertGreater(len(m.projected_improvement), 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
