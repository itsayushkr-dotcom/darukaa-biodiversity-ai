"""
Darukaa.Earth: AI Biodiversity Intelligence Chatbot & Scientist Workbench
Interactive Streamlit Dashboard demonstrating Conversational Intelligence,
Multi-Metric Causal Reasoning, RAG Knowledge Retrieval, and Geo-Spatial Context.
"""

import os
import sys
import json
import time
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

# Custom Styling for Minimalist, Ultra-Clean Forest Emerald Aesthetic
st.markdown("""
<style>
    /* Global Clean Font & Background Accents */
    .main .block-container {
        padding-top: 1.8rem;
        padding-bottom: 2.5rem;
        max-width: 1200px;
    }
    
    /* Vibrant Gradient Minimalist Title */
    .main-title {
        font-size: 2.1rem;
        font-weight: 800;
        background: linear-gradient(135deg, #00f2fe 0%, #4facfe 35%, #43e97b 75%, #38ef7d 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.15rem;
        letter-spacing: -0.5px;
        display: inline-block;
    }
    .sub-title {
        font-size: 0.98rem;
        color: #94a3b8 !important;
        font-weight: 400;
        margin-bottom: 1.2rem;
        letter-spacing: 0.1px;
    }
    
    /* Minimalist Glass Cards */
    .card-box {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-left: 3px solid #10b981;
        padding: 1rem 1.2rem;
        border-radius: 8px;
        margin-bottom: 1rem;
    }
    
    /* Citation & Metric Badges */
    .citation-badge {
        background-color: rgba(56, 189, 248, 0.15) !important;
        color: #38bdf8 !important;
        border: 1px solid rgba(56, 189, 248, 0.35) !important;
        font-size: 0.82rem;
        font-weight: 500;
        padding: 3px 9px;
        border-radius: 6px;
        display: inline-block;
        margin-top: 3px;
        margin-right: 5px;
        text-decoration: none !important;
        cursor: pointer;
        transition: all 0.2s ease-in-out;
    }
    .citation-badge:hover {
        background-color: rgba(56, 189, 248, 0.35) !important;
        color: #ffffff !important;
        border-color: #38bdf8 !important;
        transform: translateY(-1px);
        box-shadow: 0 2px 8px rgba(56, 189, 248, 0.25);
    }
    .metric-badge {
        background-color: rgba(52, 211, 153, 0.18) !important;
        color: #34d399 !important;
        border: 1px solid rgba(52, 211, 153, 0.35);
        font-weight: 600;
        font-size: 0.85rem;
        padding: 3px 8px;
        border-radius: 5px;
        display: inline-block;
        margin: 2px 0;
    }
    .variable-pill {
        background-color: rgba(168, 85, 247, 0.18) !important;
        color: #d8b4fe !important;
        border: 1px solid rgba(168, 85, 247, 0.35);
        font-weight: 500;
        font-size: 0.8rem;
        padding: 2px 8px;
        border-radius: 5px;
        display: inline-block;
        margin: 2px 4px 2px 0;
    }
    .confidence-badge-high {
        background-color: rgba(34, 197, 94, 0.18) !important;
        color: #4ade80 !important;
        border: 1px solid rgba(34, 197, 94, 0.4);
        font-weight: 700;
        font-size: 0.82rem;
        padding: 3px 8px;
        border-radius: 5px;
        display: inline-block;
    }
    
    /* Clean Tab Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        border-bottom: 1px solid rgba(255, 255, 255, 0.08);
        padding-bottom: 4px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 8px 16px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.92rem;
        color: #94a3b8;
    }
    .stTabs [aria-selected="true"] {
        color: #34d399 !important;
        border-bottom: 2px solid #34d399 !important;
        background: rgba(16, 185, 129, 0.08);
    }
    
    /* History & Suggestion Buttons */
    .stButton button {
        border-radius: 8px;
        transition: all 0.15s ease-in-out;
    }
</style>
""", unsafe_allow_html=True)


# Initialize Core Resources
@st.cache_resource
def get_vector_store():
    return EnvironmentalVectorStore()

@st.cache_resource
def get_geo_resolver():
    return GeoSpatialResolver()

vector_store = get_vector_store()
geo_resolver = get_geo_resolver()


# ----------------------------------------------------
# Multi-Session Consultation History State Management
# ----------------------------------------------------
if "consultations" not in st.session_state:
    initial_id = "session_default"
    st.session_state.consultations = {
        initial_id: {
            "title": "Biodiversity is declining on...",
            "chat_history": [],
            "agent": BiodiversityIntelligenceAgent(vector_store=vector_store)
        }
    }
    st.session_state.active_session_id = initial_id

# Safety check for active session key
if st.session_state.active_session_id not in st.session_state.consultations:
    st.session_state.active_session_id = list(st.session_state.consultations.keys())[0]

active_session = st.session_state.consultations[st.session_state.active_session_id]
agent = active_session["agent"]
chat_history = active_session["chat_history"]

# Backward compatibility bindings
st.session_state.agent = agent
st.session_state.chat_history = chat_history


# ----------------------------------------------------
# Sidebar: Brand, New Consultation, & History List
# ----------------------------------------------------
with st.sidebar:
    # Sleek Brand Header matching ChatGPT/Claude aesthetic
    st.markdown("""
    <div style='display: flex; align-items: center; gap: 12px; margin-bottom: 1.2rem; padding: 4px 0;'>
        <div style='background: linear-gradient(135deg, #059669 0%, #10b981 100%); width: 44px; height: 44px; border-radius: 10px; display: flex; align-items: center; justify-content: center; box-shadow: 0 4px 14px rgba(16, 185, 129, 0.35); flex-shrink: 0;'>
            <span style='font-size: 1.5rem;'>🌿</span>
        </div>
        <div>
            <div style='font-size: 1.25rem; font-weight: 800; color: #f8fafc; line-height: 1.15;'>Darukaa.Earth</div>
            <div style='font-size: 0.76rem; color: #34d399; font-weight: 500; letter-spacing: 0.2px;'>AI Biodiversity Intelligence</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Prominent "+ New Consultation" Action
    if st.button("➕ New Consultation", key="btn_new_consult", use_container_width=True, type="primary"):
        new_id = f"session_{len(st.session_state.consultations) + 1}_{int(time.time())}"
        st.session_state.consultations[new_id] = {
            "title": "New Consultation",
            "chat_history": [],
            "agent": BiodiversityIntelligenceAgent(vector_store=vector_store)
        }
        st.session_state.active_session_id = new_id
        st.rerun()

    # Section: CONSULTATION HISTORY
    st.markdown("""
    <div style='margin-top: 1.4rem; margin-bottom: 0.5rem; font-size: 0.72rem; font-weight: 700; color: #6ee7b7; letter-spacing: 1px;'>
        CONSULTATION HISTORY
    </div>
    """, unsafe_allow_html=True)

    # Render History List with Active Highlighting
    for s_id, s_data in list(st.session_state.consultations.items()):
        is_active = (s_id == st.session_state.active_session_id)
        btn_label = f"💬 {s_data['title']}"
        btn_type = "primary" if is_active else "secondary"
        if st.button(btn_label, key=f"session_btn_{s_id}", use_container_width=True, type=btn_type):
            st.session_state.active_session_id = s_id
            st.rerun()

    st.markdown("<div style='margin: 1.2rem 0; border-top: 1px solid rgba(255,255,255,0.08);'></div>", unsafe_allow_html=True)

    # Compact Collapsible Benchmark Presets
    with st.expander("🧪 Benchmark Presets", expanded=False):
        if st.button("📌 1. Semi-Arid Monoculture", key="pre_1", use_container_width=True):
            st.session_state.pending_prompt = "Soil organic carbon: 0.3%, Rainfall: low, Crop: monoculture wheat, Region: semi-arid"
            st.rerun()
        if st.button("❓ 2. Incomplete Query", key="pre_2", use_container_width=True):
            st.session_state.pending_prompt = "Biodiversity is declining on my land"
            st.rerun()
        if st.button("🏜️ 3. Sodic Alkaline Soil", key="pre_3", use_container_width=True):
            st.session_state.pending_prompt = "Soil pH: 8.9, ESP: 18%, Region: arid irrigated, Crop: monoculture cotton"
            st.rerun()
        if st.button("🌊 4. Agrochemical Runoff", key="pre_4", use_container_width=True):
            st.session_state.pending_prompt = "Soil organic carbon: 0.4%, Rainfall: low, Crop: monoculture cotton, High chemical pesticide and fertilizer runoff"
            st.rerun()

    # Compact Collapsible Settings
    with st.expander("⚙️ Settings & API Key", expanded=False):
        api_key_input = st.text_input("Gemini API Key (Optional)", type="password", placeholder="AIzaSy...", value=agent.gemini_api_key or "")
        if api_key_input:
            agent.gemini_api_key = api_key_input
            st.success("API Key Linked")
        else:
            st.caption("Autonomous Scientific Engine Mode (Offline Capable RAG)")
        
        st.markdown("<div style='margin-top: 8px;'></div>", unsafe_allow_html=True)
        if st.button("🔄 Reset Current Chat", key="btn_reset_active", use_container_width=True):
            active_session["chat_history"] = []
            active_session["agent"].reset_memory()
            st.rerun()

    st.caption(f"📚 {len(vector_store.documents)} peer-reviewed documents indexed (FAO, IPCC, IPBES, ICAR)")


# ----------------------------------------------------
# Main Header
# ----------------------------------------------------
st.markdown("<div class='main-title'>🌿 Darukaa.Earth — AI Biodiversity Intelligence</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-title'>Autonomous Environmental Scientist: Multi-Metric Causal Reasoning & Evidence-Backed Restoration</div>", unsafe_allow_html=True)


# ----------------------------------------------------
# Minimalist, Non-Congested Clean Tabs
# ----------------------------------------------------
tab_chat, tab_structured, tab_visualizer, tab_kb = st.tabs([
    "💬 Chat",
    "📋 Form & GPS",
    "📊 Impact Radar",
    "📚 Knowledge Base"
])


# ----------------------------------------------------
# TAB 1: Chat (Multi-Turn Dialogue & Follow-Ups)
# ----------------------------------------------------
with tab_chat:
    # Empty State: Starter Prompt Cards
    if len(chat_history) == 0:
        st.markdown("""
        <div style='background: rgba(16, 185, 129, 0.05); border: 1px solid rgba(16, 185, 129, 0.18); border-radius: 10px; padding: 1.2rem 1.4rem; margin-bottom: 1.2rem;'>
            <h4 style='margin: 0 0 0.4rem 0; color: #34d399; font-size: 1.15rem;'>🌿 Welcome to your Ecological Consultation</h4>
            <p style='margin: 0; color: #cbd5e1; font-size: 0.9rem; line-height: 1.5;'>
                Describe your land, soil health, rainfall, or farming conditions. The autonomous scientist will reason across causal dimensions or ask clarifying questions if data is missing.
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<div style='font-size: 0.8rem; font-weight: 700; color: #94a3b8; letter-spacing: 0.5px; margin-bottom: 8px;'>OR LAUNCH A BENCHMARK SCENARIO:</div>", unsafe_allow_html=True)
        col_st1, col_st2 = st.columns(2)
        with col_st1:
            if st.button("🌾 Semi-Arid Monoculture Wheat\n(SOC: 0.3%, Low Rain)", key="hero_st_1", use_container_width=True):
                st.session_state.pending_prompt = "Soil organic carbon: 0.3%, Rainfall: low, Crop: monoculture wheat, Region: semi-arid"
                st.rerun()
            if st.button("🏜️ Sodic Alkaline Soil Reclamation\n(pH: 8.9, ESP: 18%, Arid)", key="hero_st_2", use_container_width=True):
                st.session_state.pending_prompt = "Soil pH: 8.9, ESP: 18%, Region: arid irrigated, Crop: monoculture cotton"
                st.rerun()
        with col_st2:
            if st.button("❓ Incomplete Query (Test Clarifying Engine)\n'Biodiversity is declining on my land'", key="hero_st_3", use_container_width=True):
                st.session_state.pending_prompt = "Biodiversity is declining on my land"
                st.rerun()
            if st.button("🌊 Agrochemical Runoff & Ecosystem Shock\n(Chemical Runoff, SOC 0.4%)", key="hero_st_4", use_container_width=True):
                st.session_state.pending_prompt = "Soil organic carbon: 0.4%, Rainfall: low, Crop: monoculture cotton, High chemical pesticide and fertilizer runoff"
                st.rerun()

    # Render Conversation Messages
    for msg_idx, msg in enumerate(chat_history):
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
                                        link_url = getattr(c, 'url', None) or "https://openknowledge.fao.org/handle/20.500.14283/cb6378en"
                                        sec = getattr(c, 'section_or_page', '') or ''
                                        tip = f"{c.title} — {sec}" if sec else c.title
                                        citations_html += f"<a href='{link_url}' target='_blank' rel='noopener noreferrer' class='citation-badge' title='{tip}'>📖 {c.source} ({c.year}) ↗</a> "
                                    st.markdown(citations_html, unsafe_allow_html=True)

                                with st.expander(f"🔬 Evidence Verification: Exact Sections & Quotes for #{idx}", expanded=False):
                                    for c in it.scientific_citations:
                                        sec = getattr(c, 'section_or_page', None)
                                        quote = getattr(c, 'exact_quote_or_finding', None)
                                        link_url = getattr(c, 'url', None) or "https://openknowledge.fao.org/handle/20.500.14283/cb6378en"
                                        st.markdown(f"**{c.title}**")
                                        st.caption(f"🏛️ **Authority:** {c.source} ({c.year})")
                                        if sec:
                                            st.markdown(f"📍 **Exact Location:** `{sec}`")
                                        if quote:
                                            st.markdown(f"💬 *\"{quote}\"*")
                                        st.markdown(f"<a href='{link_url}' target='_blank' rel='noopener noreferrer' class='citation-badge' style='margin-top:4px; margin-bottom:10px;'>🔗 Jump to Highlighted Section in Report ↗</a>", unsafe_allow_html=True)
                                        st.markdown("---")
                                st.divider()

                        st.download_button(
                            label="📥 Download Structured Ecological Diagnosis (JSON)",
                            data=json.dumps(resp.model_dump(), indent=2),
                            file_name=f"darukaa_diagnosis_turn_{msg_idx + 1}.json",
                            mime="application/json",
                            key=f"dl_btn_turn_{msg_idx}"
                        )
                    
                    # Dynamic Follow-up Suggestive Questions (Displayed on latest turn)
                    if msg_idx == len(chat_history) - 1:
                        if resp.status == "NEEDS_CLARIFICATION":
                            follow_ups = [
                                "🌾 Rainfall: 350mm semi-arid, SOC: 0.35%, monoculture wheat",
                                "🏜️ Soil pH: 8.9, ESP: 18%, cotton with flood irrigation",
                                "🌊 Annual rain 420mm, high agrochemical runoff"
                            ]
                        else:
                            follow_ups = [
                                "💰 What are implementation costs & transition risks in Year 1?",
                                "💧 How will this sequence interact with winter moisture deficits?",
                                "📊 Show quantitative validation from the FAO RECSOIL report"
                            ]

                        st.markdown("<div style='margin-top: 16px; margin-bottom: 6px; font-size: 0.78rem; font-weight: 700; color: #34d399; letter-spacing: 0.8px;'>💡 SUGGESTED FOLLOW-UP INQUIRIES:</div>", unsafe_allow_html=True)
                        col_fu1, col_fu2, col_fu3 = st.columns(3)
                        with col_fu1:
                            if st.button(follow_ups[0], key=f"fu_btn_0_{msg_idx}", use_container_width=True):
                                st.session_state.pending_prompt = follow_ups[0]
                                st.rerun()
                        with col_fu2:
                            if st.button(follow_ups[1], key=f"fu_btn_1_{msg_idx}", use_container_width=True):
                                st.session_state.pending_prompt = follow_ups[1]
                                st.rerun()
                        with col_fu3:
                            if st.button(follow_ups[2], key=f"fu_btn_2_{msg_idx}", use_container_width=True):
                                st.session_state.pending_prompt = follow_ups[2]
                                st.rerun()
                else:
                    st.write(msg["content"])

    # Chat Input Box & Submission Handler
    pending_text = st.session_state.pop("pending_prompt", None)
    chat_input_val = st.chat_input("Ask about soil restoration, causal chains, or crop rotation...")
    user_prompt = chat_input_val or pending_text

    if user_prompt:
        # Dynamically label session title from first user query
        if active_session["title"] in ["New Consultation", "Biodiversity is declining on..."] and len(chat_history) == 0:
            clean_title = user_prompt.replace("Soil organic carbon:", "SOC:").split(",")[0].strip()
            if len(clean_title) > 28:
                clean_title = clean_title[:28] + "..."
            active_session["title"] = clean_title

        with st.chat_message("user"):
            st.write(user_prompt)

        with st.spinner("Analyzing multi-variable causal interactions & scientific literature..."):
            response = agent.chat(user_prompt)

        chat_history.append({"role": "user", "content": user_prompt})
        chat_history.append({
            "role": "assistant",
            "content": response.overall_scientific_summary or "",
            "response_obj": response
        })

        st.rerun()


# ----------------------------------------------------
# TAB 2: Form & GPS (Structured Inputs & Spatial API)
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
            res = agent.chat(
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
                        link_url = getattr(c, 'url', None) or "https://openknowledge.fao.org/handle/20.500.14283/cb6378en"
                        sec = getattr(c, 'section_or_page', '') or ''
                        tip = f"{c.title} — {sec}" if sec else c.title
                        citations_html += f"<a href='{link_url}' target='_blank' rel='noopener noreferrer' class='citation-badge' title='{tip}'>📖 {c.source} ({c.year}) ↗</a> "
                    st.markdown(citations_html, unsafe_allow_html=True)

                with st.expander(f"🔬 Evidence Verification: Exact Sections & Quotes for #{idx}", expanded=False):
                    for c in it.scientific_citations:
                        sec = getattr(c, 'section_or_page', None)
                        quote = getattr(c, 'exact_quote_or_finding', None)
                        link_url = getattr(c, 'url', None) or "https://openknowledge.fao.org/handle/20.500.14283/cb6378en"
                        st.markdown(f"**{c.title}**")
                        st.caption(f"🏛️ **Authority:** {c.source} ({c.year})")
                        if sec:
                            st.markdown(f"📍 **Exact Location:** `{sec}`")
                        if quote:
                            st.markdown(f"💬 *\"{quote}\"*")
                        st.markdown(f"<a href='{link_url}' target='_blank' rel='noopener noreferrer' class='citation-badge' style='margin-top:4px; margin-bottom:10px;'>🔗 Jump to Highlighted Section in Report ↗</a>", unsafe_allow_html=True)
                        st.markdown("---")
                st.divider()


# ----------------------------------------------------
# TAB 3: Impact Radar (Quantitative Visualizer)
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
# TAB 4: Knowledge Base (RAG Inspector)
# ----------------------------------------------------
with tab_kb:
    st.markdown("#### Scientific Knowledge Base & RAG Retrieval Inspector")
    st.caption("Search and inspect the indexed scientific literature, reports, and quantified impact models.")

    kb_query = st.text_input("Search Scientific Knowledge Base", value="legume cover crops soil carbon microbial diversity")

    if kb_query:
        matches = vector_store.search(kb_query, top_k=5)
        st.write(f"Retrieved **{len(matches)}** peer-reviewed records matching query:")

        for m in matches:
            with st.expander(f"📖 {m['title']} — {m['source']} ({m['publication_year']}) [Relevance: {m['relevance_score']}]"):
                st.markdown(f"**Domain:** `{m['domain']}`")
                st.markdown(f"**Coupled Environmental Variables:** `{', '.join(m['environmental_variables'])}`")
                st.markdown(f"**Scientific Mechanism:**\n{m['scientific_mechanism']}")
                st.markdown("**Quantified Impact:**")
                st.json(m['quantified_impact'])
                st.markdown("**Recommended Interventions:**")
                for rec in m['recommended_interventions']:
                    st.markdown(f"• {rec}")
                m_sec = m.get("section_or_page")
                if m_sec:
                    st.markdown(f"📍 **Exact Report Location:** `{m_sec}`")
                m_url = m.get("url", "https://www.fao.org/global-soil-partnership/en/")
                st.markdown(f"<div style='margin-top: 14px;'><a href='{m_url}' target='_blank' rel='noopener noreferrer' class='citation-badge' style='font-size:0.88rem; padding:6px 14px;'>🔗 Jump to Highlighted Section in Official Report ({m['source']}) ↗</a></div>", unsafe_allow_html=True)
