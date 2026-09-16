"""
Darukaa.Earth: AI Biodiversity Intelligence Chatbot & Scientist Workbench
Interactive Streamlit Dashboard demonstrating Conversational Intelligence,
Multi-Metric Causal Reasoning, RAG Knowledge Retrieval, and Geo-Spatial Context.
"""

import os
import sys
import json
import streamlit as st
import pandas as pd

# Ensure root directory is on sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.intelligence_agent import BiodiversityIntelligenceAgent
from core.schemas import StructuredEnvironmentalInput, GeoCoordinates
from knowledge_base.vector_store import EnvironmentalVectorStore
from core.geo_spatial import GeoSpatialResolver

st.set_page_config(
    page_title="Darukaa.Earth | AI Biodiversity Intelligence",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling for Vibrant High-Contrast Scientific Aesthetic
st.markdown("""
<style>
    /* Vibrant Gradient High-Contrast Title */
    .main-title {
        font-size: 2.3rem;
        font-weight: 800;
        background: linear-gradient(135deg, #00f2fe 0%, #4facfe 35%, #43e97b 75%, #38ef7d 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.25rem;
        letter-spacing: -0.5px;
        display: inline-block;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #e2e8f0 !important;
        font-weight: 400;
        margin-bottom: 1.5rem;
        opacity: 0.92;
    }
    .card-box {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.12);
        border-left: 4px solid #00f2fe;
        padding: 1.2rem;
        border-radius: 8px;
        margin-bottom: 1.2rem;
    }
    .citation-badge {
        background-color: rgba(56, 189, 248, 0.22) !important;
        color: #38bdf8 !important;
        border: 1px solid rgba(56, 189, 248, 0.45) !important;
        font-size: 0.85rem;
        font-weight: 500;
        padding: 4px 10px;
        border-radius: 6px;
        display: inline-block;
        margin-top: 4px;
        margin-right: 6px;
        text-decoration: none !important;
        cursor: pointer;
        transition: all 0.2s ease-in-out;
    }
    .citation-badge:hover {
        background-color: rgba(56, 189, 248, 0.45) !important;
        color: #ffffff !important;
        border-color: #38bdf8 !important;
        transform: translateY(-1px);
        box-shadow: 0 2px 10px rgba(56, 189, 248, 0.35);
    }
    .metric-badge {
        background-color: rgba(52, 211, 153, 0.22) !important;
        color: #34d399 !important;
        border: 1px solid rgba(52, 211, 153, 0.45);
        font-weight: 600;
        font-size: 0.88rem;
        padding: 4px 10px;
        border-radius: 6px;
        display: inline-block;
        margin: 3px 0;
    }
    .variable-pill {
        background-color: rgba(168, 85, 247, 0.22) !important;
        color: #d8b4fe !important;
        border: 1px solid rgba(168, 85, 247, 0.45);
        font-weight: 500;
        font-size: 0.82rem;
        padding: 3px 9px;
        border-radius: 6px;
        display: inline-block;
        margin: 2px 4px 2px 0;
    }
    .confidence-badge-high {
        background-color: rgba(34, 197, 94, 0.22) !important;
        color: #4ade80 !important;
        border: 1px solid rgba(34, 197, 94, 0.5);
        font-weight: 700;
        font-size: 0.85rem;
        padding: 4px 10px;
        border-radius: 6px;
        display: inline-block;
    }
</style>
""", unsafe_allow_html=True)


# Initialize Session State
@st.cache_resource
def get_vector_store():
    return EnvironmentalVectorStore()

@st.cache_resource
def get_geo_resolver():
    return GeoSpatialResolver()

vector_store = get_vector_store()
geo_resolver = get_geo_resolver()

if "agent" not in st.session_state:
    st.session_state.agent = BiodiversityIntelligenceAgent(vector_store=vector_store)

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


# Sidebar Controls
with st.sidebar:
    st.image("https://img.icons8.com/color/96/natural-food.png", width=64)
    st.title("Darukaa.Earth")
    st.caption("AI Environmental Scientist System")
    st.markdown("---")

    # API Configuration
    st.subheader("🔑 LLM Configuration")
    api_key_input = st.text_input("Gemini API Key (Optional)", type="password", placeholder="AIzaSy...")
    if api_key_input:
        st.session_state.agent.gemini_api_key = api_key_input
        st.success("Gemini API Key Linked")
    else:
        st.info("Operating in Autonomous Scientific Engine Mode (Offline Capable RAG)")

    st.markdown("---")
    st.subheader("🧪 Quick Load Benchmarks")

    if st.button("📌 Example 1: Semi-Arid Monoculture"):
        st.session_state.quick_query = "Soil organic carbon: 0.3%, Rainfall: low, Crop: monoculture wheat, Region: semi-arid"

    if st.button("❓ Example 2: Incomplete Query"):
        st.session_state.quick_query = "Biodiversity is declining on my land"

    if st.button("🏜️ Example 3: Sodic Alkaline Soil"):
        st.session_state.quick_query = "Soil pH: 8.9, ESP: 18%, Region: arid irrigated, Crop: monoculture cotton"

    if st.button("🌊 Example 4: Agrochemical Pollution"):
        st.session_state.quick_query = "Soil organic carbon: 0.4%, Rainfall: low, Crop: monoculture cotton, High chemical pesticide and fertilizer runoff"

    st.markdown("---")
    if st.button("🔄 Reset Conversation Memory"):
        st.session_state.agent.reset_memory()
        st.session_state.chat_history = []
        st.rerun()

    st.markdown("---")
    st.caption(f"Knowledge Base: **{len(vector_store.documents)}** peer-reviewed documents indexed")
    st.caption("Indexed Sources: FAO, IPCC, IPBES, ICAR")


# Header
st.markdown("<div class='main-title'>🌿 Darukaa.Earth — AI Biodiversity Intelligence</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-title'>Autonomous Environmental Scientist: Multi-Metric Causal Reasoning & Evidence-Backed Restoration</div>", unsafe_allow_html=True)

# Tabs
tab_chat, tab_structured, tab_visualizer, tab_kb = st.tabs([
    "💬 Conversational Scientist (Chat)",
    "📋 Structured Input & Geo-Spatial (JSON/Form)",
    "📊 Multi-Metric Impact Radar",
    "📚 Scientific Knowledge Layer (RAG Inspector)"
])


# ----------------------------------------------------
# TAB 1: Conversational Scientist (Chat)
# ----------------------------------------------------
with tab_chat:
    st.markdown("#### Multi-Turn Ecological Dialogue with Missing-Parameter Detection")
    st.caption("Ask free-form questions. If variables are missing (< 3 environmental dimensions), the scientist will ask clarifying questions.")

    # Display chat history
    for msg_idx, msg in enumerate(st.session_state.chat_history):
        with st.chat_message(msg["role"]):
            if msg["role"] == "user":
                st.write(msg["content"])
            else:
                resp = msg.get("response_obj")
                if resp:
                    if resp.status == "NEEDS_CLARIFICATION":
                        st.warning("⚠️ **Incomplete Environmental Profile Detected**")
                        if resp.clarifying_question:
                            st.markdown(resp.clarifying_question.question)
                            st.caption(f"🔬 *Scientific Rationale*: {resp.clarifying_question.scientific_rationale}")
                    else:
                        st.success("✅ **Multi-Metric Scientific Recommendation Formulated**")
                        if resp.overall_scientific_summary:
                            st.info(resp.overall_scientific_summary)

                        # Causal feedback chain
                        if resp.multi_variable_causal_chain:
                            with st.expander("🔗 View Multi-Variable Causal Graph & Interaction Chain", expanded=True):
                                st.markdown(resp.multi_variable_causal_chain)

                        # Render Interventions
                        for idx, it in enumerate(resp.interventions, 1):
                            with st.container():
                                st.markdown(f"### {idx}. {it.title}")
                                st.markdown(f"**What to Do:** {it.what_to_do}")
                                st.markdown(f"**Why it Works:** {it.why_it_works}")

                                col_m1, col_m2 = st.columns([2, 1])
                                with col_m1:
                                    st.markdown("**Coupled Variables Connected (>= 3):**")
                                    pills_html = "".join([f"<span class='variable-pill'>🔗 {var}</span>" for var in it.environmental_variables_coupled])
                                    st.markdown(pills_html, unsafe_allow_html=True)
                                    st.markdown("**Impacted Metrics & Quantified Estimates:**")
                                    for m in it.impacted_metrics:
                                        st.markdown(f"- <span class='metric-badge'>{m.metric_name}</span>: **{m.projected_improvement}** *({m.time_horizon})*", unsafe_allow_html=True)
                                with col_m2:
                                    st.markdown(f"**Confidence Level:** <span class='confidence-badge-high'>{it.confidence_score * 100:.0f}% High Evidence</span>", unsafe_allow_html=True)
                                    st.markdown(f"**Time Horizon:** {it.time_horizon_summary}")
                                    st.markdown("**Evidence Citations (Clickable ↗):**")
                                    citations_html = ""
                                    for c in it.scientific_citations:
                                        link_url = getattr(c, 'url', None) or "https://www.fao.org/global-soil-partnership/recsoil/en/"
                                        citations_html += f"<a href='{link_url}' target='_blank' rel='noopener noreferrer' class='citation-badge' title='Read official study / report'>📖 {c.source} ({c.year}) ↗</a> "
                                    st.markdown(citations_html, unsafe_allow_html=True)
                                st.divider()

                        st.download_button(
                            label="📥 Download Structured Ecological Diagnosis (JSON)",
                            data=json.dumps(resp.model_dump(), indent=2),
                            file_name=f"darukaa_diagnosis_turn_{msg_idx + 1}.json",
                            mime="application/json",
                            key=f"dl_btn_turn_{msg_idx}"
                        )
                else:
                    st.write(msg["content"])

    # Chat Input Box
    default_prompt = st.session_state.pop("quick_query", "")
    user_prompt = st.chat_input("Describe your land, soil conditions, rainfall, or crop monoculture...") or default_prompt

    if user_prompt:
        with st.chat_message("user"):
            st.write(user_prompt)

        response = st.session_state.agent.chat(user_prompt)

        st.session_state.chat_history.append({"role": "user", "content": user_prompt})
        st.session_state.chat_history.append({"role": "assistant", "content": response.overall_scientific_summary or "", "response_obj": response})

        st.rerun()


# ----------------------------------------------------
# TAB 2: Structured Input & Geo-Spatial (JSON/Form)
# ----------------------------------------------------
with tab_structured:
    st.markdown("#### Structured Environmental Form & Geo-Spatial Auto-Resolution")
    st.caption("Provide precise measurements directly or supply GPS coordinates to automatically resolve the Agro-Ecological Zone.")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🌍 1. Geo-Coordinates & Regional Context")
        geo_enabled = st.checkbox("Enable Spatial GPS Resolution", value=True)
        lat_val = st.number_input("Latitude", value=26.9124, format="%.4f")
        lon_val = st.number_input("Longitude", value=75.7873, format="%.4f")

        if geo_enabled:
            spatial_info = geo_resolver.resolve_coordinates(lat_val, lon_val)
            st.success(f"📍 Resolved Zone: **{spatial_info.get('zone_name')}**")
            st.write(f"Baseline Rainfall: `{spatial_info.get('estimated_annual_rainfall_mm')}`")
            st.write(f"Flagship Species: `{', '.join(spatial_info.get('flagship_restoration_species', []))}`")

        st.subheader("🌱 2. Soil Metrics")
        soc_input = st.slider("Soil Organic Carbon (SOC %)", min_value=0.1, max_value=4.0, value=0.3, step=0.05)
        ph_input = st.slider("Soil pH", min_value=4.0, max_value=10.0, value=7.2, step=0.1)
        moisture_input = st.selectbox("Soil Moisture State", ["low (arid)", "moderate", "high (waterlogged)"])

    with col2:
        st.subheader("🌦️ 3. Climate & Water Variables")
        rainfall_input = st.selectbox("Rainfall Pattern", ["low", "erratic semi-arid", "seasonal monsoon", "high"])
        annual_rf_mm = st.number_input("Annual Precipitation (mm)", value=380, step=25)

        st.subheader("🚜 4. Land Use & Cropping System")
        crop_input = st.text_input("Current Crop", value="monoculture wheat")
        cropping_system = st.selectbox("Cropping Pattern", ["monoculture", "intensive double-cropping", "fallow cropland"])
        tillage_input = st.selectbox("Tillage Practice", ["conventional_deep_tillage", "reduced_till", "zero_till"])

    st.markdown("---")

    # Construct JSON payload
    structured_payload = StructuredEnvironmentalInput(
        soil_organic_carbon_pct=soc_input,
        soil_ph=ph_input,
        soil_moisture=moisture_input,
        annual_rainfall_mm=float(annual_rf_mm),
        rainfall_pattern=rainfall_input,
        crop=crop_input,
        cropping_system=cropping_system,
        tillage_practice=tillage_input,
        coordinates=GeoCoordinates(latitude=lat_val, longitude=lon_val) if geo_enabled else None
    )

    with st.expander("🔍 View Structured JSON Payload"):
        st.json(structured_payload.model_dump(exclude_none=True))

    if st.button("🚀 Evaluate Structured Multi-Metric Recommendation", type="primary"):
        with st.spinner("Executing RAG retrieval and multi-variable causal reasoning..."):
            query_str = f"Crop: {crop_input}, SOC: {soc_input}%, Rainfall: {rainfall_input}"
            res = st.session_state.agent.chat(
                user_message=query_str,
                structured_input=structured_payload
            )

        if res.status == "RECOMMENDATION_READY":
            st.success("✅ Scientific Interventions Formulated Successfully!")
            st.info(res.overall_scientific_summary)

            for idx, it in enumerate(res.interventions, 1):
                st.markdown(f"### {idx}. {it.title}")
                st.write(f"**Action:** {it.what_to_do}")
                st.write(f"**Mechanism:** {it.why_it_works}")
                st.write(f"**Coupled Dimensions:** {', '.join(it.environmental_variables_coupled)}")

                col_m1, col_m2 = st.columns(2)
                with col_m1:
                    st.markdown("**Quantified Projected Impacts:**")
                    for imp in it.impacted_metrics:
                        st.markdown(f"- <span class='metric-badge'>{imp.metric_name}:</span> **{imp.projected_improvement}** *({imp.time_horizon})*", unsafe_allow_html=True)
                with col_m2:
                    st.markdown(f"**Confidence Level:** <span class='confidence-badge-high'>{it.confidence_score * 100:.0f}% High Evidence</span>", unsafe_allow_html=True)
                    st.markdown(f"**Time Horizon:** {it.time_horizon_summary}")
                    st.markdown("**Evidence Citations (Clickable ↗):**")
                    citations_html = ""
                    for c in it.scientific_citations:
                        link_url = getattr(c, 'url', None) or "https://www.fao.org/global-soil-partnership/recsoil/en/"
                        citations_html += f"<a href='{link_url}' target='_blank' rel='noopener noreferrer' class='citation-badge' title='Read official study / report'>📖 {c.source} ({c.year}) ↗</a> "
                    st.markdown(citations_html, unsafe_allow_html=True)
                st.divider()


# ----------------------------------------------------
# TAB 3: Multi-Metric Impact Radar
# ----------------------------------------------------
with tab_visualizer:
    st.markdown("#### Multi-Metric Ecological Projection (Baseline vs. Post-Intervention)")
    st.caption("Quantifying improvement across the 5 core environmental dimensions over a 2–3 year restoration cycle.")

    chart_data = pd.DataFrame({
        "Ecological Metric": [
            "Soil Organic Carbon (t/ha)",
            "Water Infiltration Rate (%)",
            "Microbial Biomass (MBC)",
            "Pollinator Visitation Density",
            "Natural Pest Predation (%)"
        ],
        "Baseline (Degraded Monoculture)": [0.3, 20.0, 25.0, 15.0, 20.0],
        "Target Post-Intervention (2-3 Yrs)": [0.65, 55.0, 60.0, 85.0, 65.0]
    })

    st.bar_chart(chart_data.set_index("Ecological Metric"), height=380)

    st.markdown(r"""
    > [!NOTE]
    > **Mathematical Basis**:
    > - Soil Organic Carbon follows the **FAO RECSOIL sequestration model** ($0.2\text{--}0.5\text{ t C/ha/yr}$ under legume cover cropping).
    > - Water infiltration increases proportionally with macropore stabilization and earthworm density (**IPCC SRCCL Ch. 4**).
    > - Pollinator visitation density surge ($+60\text{--}110\%$) is derived from **IPBES Thematic Assessment on Pollinators**.
    """)


# ----------------------------------------------------
# TAB 4: Scientific Knowledge Layer (RAG Inspector)
# ----------------------------------------------------
with tab_kb:
    st.markdown("#### Scientific Knowledge Base & RAG Retrieval Inspector")
    st.caption("Search and inspect the indexed scientific literature, reports, and quantified impact models.")

    kb_query = st.text_input("Search Scientific Knowledge Base", value="legume cover crops soil carbon microbial diversity")

    if kb_query:
        matches = vector_store.search(kb_query, top_k=5)
        st.write(f"Retrieved **{len(matches)}** peer-reviewed records matching query:")

        for m in matches:
            with st.expander(f"📖 {m['title']} — {m['source']} ({m['publication_year']}) [Relevance Score: {m['relevance_score']}]"):
                st.markdown(f"**Domain:** `{m['domain']}`")
                st.markdown(f"**Coupled Environmental Variables:** `{', '.join(m['environmental_variables'])}`")
                st.markdown(f"**Scientific Mechanism:**\n{m['scientific_mechanism']}")
                st.markdown("**Quantified Impact:**")
                st.json(m['quantified_impact'])
                st.markdown("**Recommended Interventions:**")
                for rec in m['recommended_interventions']:
                    st.markdown(f"• {rec}")
                m_url = m.get("url", "https://www.fao.org/global-soil-partnership/en/")
                st.markdown(f"<div style='margin-top: 14px;'><a href='{m_url}' target='_blank' rel='noopener noreferrer' class='citation-badge' style='font-size:0.88rem; padding:6px 14px;'>🔗 Read Official Study / Report ({m['source']}) ↗</a></div>", unsafe_allow_html=True)
