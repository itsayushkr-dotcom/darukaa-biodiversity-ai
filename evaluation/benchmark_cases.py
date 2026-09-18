"""
Darukaa.Earth AI Biodiversity Intelligence Benchmark & Evaluation Suite.
Validates the system against all 5 evaluation criteria, constraints, and edge cases
defined in the Hackathon Specification:

1. Depth of Reasoning (30%): Non-obvious recommendations, >= 3 variables coupled together, causal graph.
2. Scientific Grounding (25%): Credible sources (FAO, IPCC, IPBES, UNEP), quantified delta estimates, explainable mechanisms.
3. Knowledge System Design (20%): Vector RAG retrieval, cosine scoring, structured corpus coverage.
4. Conversational Intelligence (15%): Incomplete input detection, targeted clarifying questions, multi-turn memory accumulation.
5. Output Clarity (10%): Structured Pydantic schema, actionable guidance, time horizons, calibrated confidence scores.

Constraints Verified:
- No generic LLM-only solutions (100% offline deterministic resilience).
- No shallow or obvious recommendations.
- Minimum 3 coupled environmental variables per intervention.

Edge Cases Verified:
- Extreme soil & climate values (SOC 0.05%, rainfall 150mm, pH 8.8).
- Colloquial & varied input syntax.
- Multi-tenant session isolation with SQLite persistence.
- Geo-spatial bounding box matching and global graceful fallback.
"""

import sys
import os
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.intelligence_agent import BiodiversityIntelligenceAgent
from core.schemas import StructuredEnvironmentalInput, GeoCoordinates, IntelligenceResponse
from knowledge_base.vector_store import EnvironmentalVectorStore
from core.geo_spatial import GeoSpatialResolver
from core.parameter_extractor import ParameterExtractor
from core.clarifying_engine import ClarifyingEngine
from core.session_store import UserSessionDB


class TestDarukaaBiodiversitySystem(unittest.TestCase):

    def setUp(self):
        self.vector_store = EnvironmentalVectorStore()
        self.agent = BiodiversityIntelligenceAgent(vector_store=self.vector_store)
        self.geo_resolver = GeoSpatialResolver()
        self.session_db = UserSessionDB()

    # =========================================================================
    # 1. DEPTH OF REASONING (30%)
    # =========================================================================

    def test_criterion_1_depth_of_reasoning_coupling_at_least_3_variables(self):
        """
        Constraint from Challenge:
        'Must handle at least 3 environmental variables together.
         This is the core differentiator—no single-variable answers.'
        """
        query = "Soil organic carbon: 0.3%, Rainfall: low, Crop: monoculture wheat, Region: semi-arid"
        res = self.agent.chat(query)

        self.assertEqual(res.status, "RECOMMENDATION_READY")
        self.assertGreaterEqual(len(res.interventions), 2)

        for it in res.interventions:
            coupled = it.environmental_variables_coupled
            self.assertGreaterEqual(
                len(coupled), 3,
                f"Intervention '{it.title}' must couple >= 3 environmental variables, found {len(coupled)}"
            )

    def test_criterion_1_depth_of_reasoning_non_obvious_biological_mechanisms(self):
        """
        Evaluation Criteria 1 (30%):
        'Are recommendations non-obvious? Do they combine multiple environmental variables?'
        Rejects shallow advice; requires concrete biophysical and ecological mechanisms.
        """
        query = "Soil organic carbon: 0.3%, annual rainfall: 350mm, Crop: monoculture wheat"
        res = self.agent.chat(query)

        self.assertEqual(res.status, "RECOMMENDATION_READY")
        all_why_it_works = " ".join([it.why_it_works.lower() for it in res.interventions])

        # Verify presence of non-obvious biological and agroecological mechanisms
        non_obvious_mechanisms = [
            "hydraulic lift", "hydraulic redistribution", "reverse-phenology",
            "microbial biomass", "glomalin", "trophic", "necromass", "mycorrhizal", "c:n ratio"
        ]
        has_deep_mechanism = any(m in all_why_it_works for m in non_obvious_mechanisms)
        self.assertTrue(
            has_deep_mechanism,
            "Interventions must detail advanced biological mechanisms (hydraulic lift, glomalin, mycorrhizal, etc.)"
        )

    def test_criterion_1_depth_of_reasoning_multi_variable_causal_chain(self):
        """
        Evaluation Criteria 1 (30%):
        Verifies system synthesizes an explicit multi-variable causal chain diagram/string
        connecting Soil <-> Climate <-> Land Use <-> Biodiversity.
        """
        query = "Soil organic carbon: 0.3%, Rainfall: low, Crop: monoculture wheat, Region: semi-arid"
        res = self.agent.chat(query)

        self.assertIsNotNone(res.multi_variable_causal_chain)
        chain = res.multi_variable_causal_chain.lower()
        self.assertTrue("soc" in chain or "carbon" in chain)
        self.assertTrue("rainfall" in chain or "aridity" in chain or "water" in chain)
        self.assertTrue("wheat" in chain or "crop" in chain or "monoculture" in chain)

    # =========================================================================
    # 2. SCIENTIFIC GROUNDING (25%)
    # =========================================================================

    def test_criterion_2_scientific_grounding_verified_citations(self):
        """
        Evaluation Criteria 2 (25%):
        'Are claims backed by credible sources? Is reasoning accurate and explainable?'
        Requires peer-reviewed / institutional consensus sources (FAO, IPCC, IPBES, UNEP, ICAR).
        """
        query = "Soil organic carbon: 0.3%, Rainfall: low, Crop: monoculture wheat"
        res = self.agent.chat(query)

        all_citations = [c for it in res.interventions for c in it.scientific_citations]
        self.assertGreaterEqual(len(all_citations), 2, "Must provide academic citations")

        sources_str = " ".join([c.source for c in all_citations])
        self.assertTrue("FAO" in sources_str, "Must cite FAO consensus reports")
        self.assertTrue("IPCC" in sources_str, "Must cite IPCC consensus reports")

        # Verify chapter/section grounding
        for c in all_citations:
            self.assertIsNotNone(c.year)
            self.assertGreaterEqual(c.year, 2015, "Citations must be contemporary peer-reviewed reports")

    def test_criterion_2_scientific_grounding_quantified_delta_estimates(self):
        """
        Evaluation Criteria 2 (25%):
        Requires measurable, quantified percentage improvement estimates rather than vague claims.
        """
        query = "Soil organic carbon: 0.3%, annual rainfall: 380mm, crop: wheat monoculture"
        res = self.agent.chat(query)

        all_improvements = []
        for it in res.interventions:
            for m in it.impacted_metrics:
                all_improvements.append(m.projected_improvement)

        self.assertGreater(len(all_improvements), 0)
        has_percentage_range = any("%" in imp for imp in all_improvements)
        self.assertTrue(has_percentage_range, "Impacted metrics must include quantified % improvement estimates")

    # =========================================================================
    # 3. KNOWLEDGE SYSTEM DESIGN (20%)
    # =========================================================================

    def test_criterion_3_knowledge_system_rag_retrieval(self):
        """
        Evaluation Criteria 3 (20%):
        'Use of RAG / vector DB / structured datasets; clarity of knowledge retrieval pipeline.'
        """
        # Test vector store indexing
        self.assertGreaterEqual(len(self.vector_store.documents), 4)
        self.assertGreaterEqual(len(self.vector_store.vocabulary), 100)

        # Test cosine semantic hybrid retrieval
        results = self.vector_store.search("soil organic carbon agroforestry semi-arid", top_k=3)
        self.assertGreaterEqual(len(results), 1)
        top_match = results[0]
        self.assertIn("relevance_score", top_match)
        self.assertGreater(top_match["relevance_score"], 0.0)
        self.assertIn("scientific_mechanism", top_match)
        self.assertIn("source", top_match)
        self.assertIn("document_id", top_match)

    def test_criterion_3_knowledge_system_domain_filtering(self):
        """
        Evaluation Criteria 3 (20%):
        Verifies vector store correctly handles domain filtering across agricultural sub-disciplines.
        """
        soil_docs = self.vector_store.search("organic carbon sequestration", domain_filter="soil_health", top_k=3)
        for doc in soil_docs:
            self.assertEqual(doc["domain"], "soil_health")

        pollinator_docs = self.vector_store.search("wild bees floral corridor", domain_filter="biodiversity", top_k=3)
        for doc in pollinator_docs:
            self.assertEqual(doc["domain"], "biodiversity")

    # =========================================================================
    # 4. CONVERSATIONAL INTELLIGENCE (15%)
    # =========================================================================

    def test_criterion_4_conversational_intelligence_incomplete_input_clarification(self):
        """
        Evaluation Criteria 4 (15%):
        'Context awareness, Follow-up questioning, Memory handling.'
        When query lacks sufficient variables (< 3), system MUST NOT guess or give generic advice.
        It must ask targeted clarifying questions.
        """
        agent = BiodiversityIntelligenceAgent(vector_store=self.vector_store)
        res = agent.chat("Biodiversity is declining on my land")

        self.assertEqual(res.status, "NEEDS_CLARIFICATION")
        self.assertIsNotNone(res.clarifying_question)
        self.assertGreaterEqual(len(res.clarifying_question.missing_variables), 1)
        self.assertEqual(len(res.interventions), 0, "Must not return shallow interventions for incomplete input")

        q_lower = res.clarifying_question.question.lower()
        self.assertTrue(
            "soil" in q_lower or "rainfall" in q_lower or "crop" in q_lower,
            "Clarifying question must prompt for specific missing environmental variables"
        )

    def test_criterion_4_conversational_intelligence_multi_turn_accumulation(self):
        """
        Evaluation Criteria 4 (15%):
        Verifies that conversational memory correctly accumulates parameters across multiple turns:
        Turn 1: Incomplete (1 parameter) -> Asks clarifying question
        Turn 2: Follow-up provides remaining parameters -> Synthesizes full recommendation.
        """
        agent = BiodiversityIntelligenceAgent(vector_store=self.vector_store)

        # Turn 1
        t1 = agent.chat("Our land suffers from severe pollinator decline")
        self.assertEqual(t1.status, "NEEDS_CLARIFICATION")

        # Turn 2: Provide remaining coupled variables
        t2 = agent.chat("The soil organic carbon is 0.3%, rainfall is low, and we have been growing monoculture wheat")
        self.assertEqual(t2.status, "RECOMMENDATION_READY")
        self.assertGreaterEqual(len(t2.interventions), 2)

        # Confirm accumulated state
        self.assertEqual(agent.accumulated_params.get("soil_organic_carbon_pct"), 0.3)
        self.assertEqual(agent.accumulated_params.get("crop"), "monoculture wheat")
        self.assertEqual(len(agent.conversation_memory), 4)

    def test_criterion_4_conversational_intelligence_parameter_override(self):
        """
        Evaluation Criteria 4 (15%):
        Verifies that if a user updates an environmental parameter in a later turn,
        the system correctly overrides the old value.
        """
        agent = BiodiversityIntelligenceAgent(vector_store=self.vector_store)
        agent.chat("Soil carbon is 0.3%, rainfall is low, crop is wheat")
        self.assertEqual(agent.accumulated_params.get("crop"), "wheat")

        # User updates crop to cotton in next turn
        agent.chat("Actually, we changed the crop to monoculture cotton with chemical pesticide runoff")
        self.assertEqual(agent.accumulated_params.get("crop"), "monoculture cotton")

    # =========================================================================
    # 5. OUTPUT CLARITY (10%)
    # =========================================================================

    def test_criterion_5_output_clarity_pydantic_schema_compliance(self):
        """
        Evaluation Criteria 5 (10%):
        'Structured, readable, and actionable responses.'
        Ensures strict Pydantic model validation and complete field population.
        """
        agent = BiodiversityIntelligenceAgent(vector_store=self.vector_store)
        res = agent.chat("Soil organic carbon: 0.3%, Rainfall: low, Crop: monoculture wheat")

        self.assertIsInstance(res, IntelligenceResponse)
        self.assertIn(res.status, ["RECOMMENDATION_READY", "NEEDS_CLARIFICATION"])
        self.assertIsInstance(res.extracted_variables, dict)

        for it in res.interventions:
            self.assertTrue(len(it.title) > 5)
            self.assertTrue(len(it.what_to_do) > 20)
            self.assertTrue(len(it.why_it_works) > 20)
            self.assertGreaterEqual(it.confidence_score, 0.70)
            self.assertLessEqual(it.confidence_score, 1.0)
            for m in it.impacted_metrics:
                self.assertIn(
                    m.time_horizon,
                    ["short-term (0-6 mo)", "medium-term (1-3 yrs)", "long-term (3-5+ yrs)"]
                )

    # =========================================================================
    # 6. CONSTRAINTS VERIFICATION
    # =========================================================================

    def test_constraint_no_generic_shallow_llm_recommendations(self):
        """
        Constraint from Challenge:
        'No generic LLM-only solutions. No shallow or obvious recommendations.'
        Prohibits generic platitudes like 'water regularly', 'use good soil', 'be sustainable'.
        """
        agent = BiodiversityIntelligenceAgent(vector_store=self.vector_store)
        res = agent.chat("Soil carbon: 0.3%, rainfall: low, crop: monoculture wheat")

        for it in res.interventions:
            text = (it.title + " " + it.what_to_do).lower()
            self.assertNotIn("water your plants regularly", text)
            self.assertNotIn("use sustainable practices", text)
            self.assertNotIn("add some good fertilizer", text)

    def test_constraint_offline_deterministic_execution_without_api_key(self):
        """
        Constraint:
        System must be 100% resilient and execute complete scientific reasoning
        offline even when no external Gemini API key is configured.
        """
        offline_agent = BiodiversityIntelligenceAgent(gemini_api_key=None, vector_store=self.vector_store)
        res = offline_agent.chat("Soil organic carbon: 0.35%, Rainfall: low, Crop: monoculture wheat")

        self.assertEqual(res.status, "RECOMMENDATION_READY")
        self.assertGreaterEqual(len(res.interventions), 2)
        self.assertIsNotNone(res.overall_scientific_summary)

    # =========================================================================
    # 7. EDGE CASES & REVIEWER SCENARIOS
    # =========================================================================

    def test_edge_case_extreme_soil_and_climate_values(self):
        """
        Edge Case:
        Extreme degraded values: SOC = 0.05%, rainfall = 150mm (hyper-arid), pH = 8.8 (highly alkaline).
        System must handle extreme degradation without crashing and formulate resilient interventions.
        """
        query = "Soil organic carbon: 0.05%, annual rainfall: 150mm, soil pH: 8.8, crop: monoculture wheat"
        res = self.agent.chat(query)

        self.assertEqual(res.status, "RECOMMENDATION_READY")
        self.assertGreaterEqual(len(res.interventions), 2)

    def test_edge_case_colloquial_and_unstructured_text_parsing(self):
        """
        Edge Case:
        Reviewers entering conversational colloquial phrasing without formal keywords:
        e.g., 'we get barely 250mm rain, carbon is around 0.35%, wheat monoculture'
        """
        query = "My soil carbon is around 0.35%, we get barely 250mm rain, and it is all wheat monoculture"
        extracted = ParameterExtractor.extract_from_text(query)

        self.assertEqual(extracted.get("soil_organic_carbon_pct"), 0.35)
        self.assertEqual(extracted.get("annual_rainfall_mm"), 250.0)
        self.assertEqual(extracted.get("rainfall_pattern"), "low")
        self.assertIn("wheat", extracted.get("crop", ""))

    def test_edge_case_chemical_pollution_and_runoff_scenario(self):
        """
        Edge Case:
        High agrochemical contamination and pesticide runoff.
        Must activate UNEP/FAO riparian buffer and biochar sorption modeling.
        """
        query = "Soil carbon: 0.4%, rainfall: low, crop: cotton, heavy synthetic fertilizer and pesticide runoff"
        res = self.agent.chat(query)

        self.assertEqual(res.status, "RECOMMENDATION_READY")
        has_biofilter = any("riparian" in it.title.lower() or "biochar" in it.title.lower() for it in res.interventions)
        self.assertTrue(has_biofilter, "Must propose riparian bio-filtration for chemical runoff")

    def test_edge_case_geo_spatial_bounding_box_and_fallback(self):
        """
        Edge Case:
        Valid GPS coordinates in semi-arid zone vs coordinates in deep ocean/unmapped regions.
        """
        # Valid Jaipur, Rajasthan coordinates
        res_valid = self.geo_resolver.resolve_coordinates(26.9124, 75.7873)
        self.assertTrue(res_valid["matched"])
        self.assertEqual(res_valid["zone_id"], "AEZ_SEMI_ARID")
        self.assertIn("flagship_restoration_species", res_valid)

        # Global ocean coordinate (0, 0)
        res_ocean = self.geo_resolver.resolve_coordinates(0.0, 0.0)
        self.assertFalse(res_ocean["matched"])
        self.assertEqual(res_ocean["zone_id"], "AEZ_GENERAL_APPROXIMATION")
        self.assertIn("flagship_restoration_species", res_ocean)

    def test_edge_case_strict_per_user_session_isolation(self):
        """
        Edge Case:
        Privacy & Data Isolation: User A's consultation history must never be visible to User B.
        """
        user_a_data = {
            "session_1": {
                "title": "User A Farm Consultation",
                "chat_history": [{"role": "user", "content": "I have 0.2% carbon soil"}],
                "agent": None
            }
        }
        self.session_db.save_user_consultations("user_alpha_123", user_a_data, "session_1")

        # Load as User A
        loaded_a, active_a = self.session_db.load_user_consultations("user_alpha_123", self.vector_store)
        self.assertIsNotNone(loaded_a)
        self.assertEqual(active_a, "session_1")

        # Load as User B (must be None, zero leakage)
        loaded_b, active_b = self.session_db.load_user_consultations("user_beta_456", self.vector_store)
        self.assertIsNone(loaded_b)
        self.assertIsNone(active_b)

        # Cleanup test entry
        self.session_db.delete_user_consultations("user_alpha_123")


if __name__ == "__main__":
    unittest.main(verbosity=2)
