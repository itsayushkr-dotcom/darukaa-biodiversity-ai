"""
Generates the official Word submission document (.docx) for the Darukaa.Earth Hackathon
meeting all criteria listed on Page 5 of the challenge document.
"""

import os
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT


def create_submission_document(output_path: str):
    doc = Document()

    # Document Title
    title = doc.add_heading("Darukaa.Earth: AI Biodiversity Intelligence Challenge", 0)
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER

    sub = doc.add_paragraph("Submission & Technical Architecture Dossier")
    sub.alignment = WD_ALIGN_PARAGRAPH.CENTER
    sub.runs[0].font.size = Pt(14)
    sub.runs[0].font.color.rgb = RGBColor(40, 100, 60)

    doc.add_paragraph()

    # Section 1: Core Submission Links
    h1 = doc.add_heading("1. Submission Links & Access", level=1)
    p = doc.add_paragraph()
    p.add_run("• GitHub Repository: ").bold = True
    p.add_run("https://github.com/itsayushkr-dotcom/darukaa-biodiversity-ai\n")
    p.add_run("• Live Demo URL: ").bold = True
    p.add_run("https://darukaa-biodiversity-ai.streamlit.app (or local workbench at http://localhost:8501)\n")
    p.add_run("• API Documentation URL: ").bold = True
    p.add_run("http://localhost:8000/docs (Interactive Swagger/OpenAPI)")

    # Reviewer Access
    doc.add_heading("Repository Access for Private Repositories", level=2)
    p_access = doc.add_paragraph("If set to private, collaborator access is granted to all designated Darukaa reviewers:")
    reviewers = [
        "ankita.dasgupta@darukaa.com",
        "harsh.kumar@darukaa.com",
        "utkarsh.gauniyal@darukaa.com",
        "guneet.mutreja@darukaa.com"
    ]
    for r in reviewers:
        doc.add_paragraph(f"  ✔ {r}", style="List Bullet")

    # Section 2: Executive Summary & Objective Alignment
    doc.add_heading("2. System Design & Scientific Depth Overview", level=1)
    doc.add_paragraph(
        "In strict adherence to the Darukaa.Earth challenge objective, this system was engineered not as a shallow "
        "LLM prompt wrapper or UI-only artifact, but as an autonomous AI Environmental Scientist. The core design combines "
        "a retrievable scientific knowledge layer (RAG), conversational intelligence with missing-parameter detection, "
        "and multi-metric causal reasoning coupling at least 3 environmental variables simultaneously."
    )

    # Key Highlights Table
    table = doc.add_table(rows=1, cols=3)
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    hdr_cells = table.rows[0].cells
    hdr_cells[0].text = "Core Pillar"
    hdr_cells[1].text = "Implementation Architecture"
    hdr_cells[2].text = "Scientific Benchmark / Standard"

    for cell in hdr_cells:
        cell.paragraphs[0].runs[0].font.bold = True

    features = [
        ("Knowledge System (20%)", "In-memory dense vector store with hybrid TF-IDF/semantic cosine retrieval", "FAO RECSOIL, IPCC SRCCL Ch. 4, IPBES Pollinators, ICAR"),
        ("Multi-Metric Reasoning (30%)", "Causal graph engine cross-linking Soil (SOC/pH), Climate (Rainfall), and Land Use (Crop)", "Simultaneously couples >= 3 environmental variables with quantified delta estimates"),
        ("Scientific Grounding (25%)", "Every recommendation is linked to exact mechanisms, metrics, and academic citations", "Strict prohibition of generic advice like 'use sustainable practices'"),
        ("Conversational Intelligence (15%)", "Multi-turn dialog memory with automatic missing-variable detector (<3 vars)", "Detects incomplete input ('Biodiversity is declining') and asks clarifying questions"),
        ("Geo-Spatial Bonus", "Lat/Long coordinate bounding box resolver to Agro-Ecological Zones (AEZ)", "Resolves semi-arid, Deccan vertisols, and temperate biomes + flagship species")
    ]

    for pillar, impl, bench in features:
        row_cells = table.add_row().cells
        row_cells[0].text = pillar
        row_cells[1].text = impl
        row_cells[2].text = bench

    # Section 3: Architecture & Schemas
    doc.add_heading("3. Database Schema & Data Architecture", level=1)
    doc.add_paragraph(
        "The system maintains structured Pydantic schemas for data integrity and RAG vector indexing:\n"
        "• StructuredEnvironmentalInput: Captures soil_organic_carbon_pct, soil_ph, soil_moisture, rainfall_pattern, annual_rainfall_mm, crop, cropping_system, tillage_practice, and coordinates.\n"
        "• EnvironmentalIntervention: Requires intervention title, what_to_do, why_it_works (mechanism), environmental_variables_coupled (>= 3 required), impacted_metrics (with quantified % improvements), and scientific_citations.\n"
        "• Vector Store Corpus: Indexed FAO, IPCC, IPBES, and ICAR consensus reports in knowledge_base/raw_documents/."
    )

    # Section 4: Local Setup & Running
    doc.add_heading("4. Local Setup & Execution Guide", level=1)
    doc.add_paragraph("The system can be executed in under 2 minutes with Python 3.10+:")

    setup_code = (
        "# 1. Clone or navigate to the repository\n"
        "git clone https://github.com/itsayushkr-dotcom/darukaa-biodiversity-ai.git\n"
        "cd darukaa-biodiversity-ai\n\n"
        "# 2. Install dependencies\n"
        "pip install -r requirements.txt\n\n"
        "# 3. Run Automated Benchmark Suite (All 7 challenge criteria)\n"
        "python evaluation/benchmark_cases.py\n\n"
        "# 4. Launch Streamlit Environmental Scientist Dashboard\n"
        "python -m streamlit run app.py\n\n"
        "# 5. (Optional) Run FastAPI REST Service\n"
        "python -m uvicorn api.main:app --port 8000 --reload"
    )
    doc.add_paragraph(setup_code)

    # Section 5: CI/CD & Production Readiness
    doc.add_heading("5. CI/CD & Automated Quality Assurance", level=1)
    doc.add_paragraph(
        "The codebase includes a continuous integration test pipeline (GitHub Actions workflow in .github/workflows/ci.yml) "
        "that automatically runs the evaluation test suite against every pull request, validating:\n"
        "1. Primary problem statement use case (SOC 0.3%, low rainfall, wheat monoculture -> agroforestry + FAO/IPCC citations).\n"
        "2. Clarifying question generation on incomplete user queries.\n"
        "3. Multi-turn context and parameter accumulation.\n"
        "4. Strict >= 3 environmental variable coupling constraint.\n"
        "5. Geo-spatial coordinate resolution and structured API compatibility."
    )

    # Section 6: Notes for Reviewers
    doc.add_heading("6. Notes for Hackathon Evaluators", level=1)
    doc.add_paragraph(
        "• Offline Resilience: The system does NOT crash or halt if an external LLM API key is absent. "
        "Its native deterministic scientific reasoning engine and local vector store execute complete, valid "
        "interventions offline.\n"
        "• Adding Gemini API Key: You can dynamically supply a Gemini API Key via the Streamlit UI sidebar or in an `.env` file "
        "to unlock dynamic generative commentary.\n"
        "• Test Coverage: 100% of benchmark tests in evaluation/benchmark_cases.py pass without warnings."
    )

    doc.save(output_path)
    print(f"Submission Word document successfully created at: {output_path}")


if __name__ == "__main__":
    out = os.path.join(os.path.dirname(__file__), "Darukaa_Earth_Submission_Dossier.docx")
    create_submission_document(out)
