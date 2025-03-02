import streamlit as st
import pandas as pd
import json
import time
import logging
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
from io import StringIO

# Import all phases
from phase1 import PhenotypeExtractor
from phase2 import PhenotypeDiseaseMatcher
from phase3 import DiagnosisConfidenceAnalyzer
from phase4 import PhenotypeRefinementEngine
from phase5 import NovelDiseaseDetector

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create necessary directories
Path("data").mkdir(exist_ok=True)
Path("results").mkdir(exist_ok=True)
Path("uploads").mkdir(exist_ok=True)

# App state management
if 'phenotypes' not in st.session_state:
    st.session_state.phenotypes = None
if 'disease_matches' not in st.session_state:
    st.session_state.disease_matches = None
if 'diagnosis_assessment' not in st.session_state:
    st.session_state.diagnosis_assessment = None
if 'novelty_analysis' not in st.session_state:
    st.session_state.novelty_analysis = None
if 'session_id' not in st.session_state:
    st.session_state.session_id = None
if 'refinement_state' not in st.session_state:
    st.session_state.refinement_state = None
if 'current_question' not in st.session_state:
    st.session_state.current_question = 0
if 'interactive_mode' not in st.session_state:
    st.session_state.interactive_mode = False
if 'file_content' not in st.session_state:
    st.session_state.file_content = None

# Initialize the system components


@st.cache_resource
def load_system_components():
    try:
        extractor = PhenotypeExtractor()
        matcher = PhenotypeDiseaseMatcher()
        analyzer = DiagnosisConfidenceAnalyzer()
        refiner = PhenotypeRefinementEngine(
            mim_titles_path="data/mimTitles.csv")
        detector = NovelDiseaseDetector(mim_titles_path="data/mimTitles.csv")

        return {
            "extractor": extractor,
            "matcher": matcher,
            "analyzer": analyzer,
            "refiner": refiner,
            "detector": detector
        }
    except Exception as e:
        logger.error(f"Error loading system components: {e}")
        st.error(f"Failed to initialize PhenoGenisis components: {e}")
        return None


# App header
st.subheader("Rare Disease Diagnostic Tool")

# Sidebar for navigation and options
with st.sidebar:
    st.markdown("<h1 style='text-align: center; font-size: 36px;'><span style='color: skyblue;'>Pheno</span><span style='color: hotpink;'>Genesis</span></h1>",
                unsafe_allow_html=True)
    st.image("/Users/chaitanyavarma/Developer/NORD/dna.png")

    st.header("Navigation")
    app_mode = st.selectbox(
        "Diagnostic Mode",
        ["Upload Clinical Notes", "Interactive Diagnosis",]
    )

    st.header("Settings")
    confidence_threshold = st.slider(
        "Confidence Threshold Adjustment",
        min_value=0.0,
        max_value=1.0,
        value=0.65,
        step=0.05,
        help="Adjust the confidence threshold for disease matching"
    )

    display_settings = st.expander("Display Settings")
    with display_settings:
        max_phenotypes = st.slider("Max Phenotypes to Display", 5, 50, 20)
        max_diseases = st.slider("Max Diseases to Display", 3, 20, 5)
        show_advanced = st.checkbox("Show Advanced Details", value=False)
        show_ic_values = st.checkbox(
            "Show Information Content Values", value=False)

    export_options = st.expander("Export Options")
    with export_options:
        export_format = st.selectbox(
            "Export Format",
            ["JSON", "CSV"],
            index=0
        )

# Main app content based on selected mode


if app_mode == "Upload Clinical Notes":
    # Load system components
    components = load_system_components()

    # File upload or text input
    # File upload for clinical notes
    uploaded_file = st.file_uploader(
        "Upload Clinical Notes", type=["txt", "pdf", "docx"])

    clinical_text = None
    file_path = None

    if uploaded_file:
        # Save the uploaded file
        file_path = f"uploads/{uploaded_file.name}"
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Display file preview
        try:
            with open(file_path, "r") as f:
                file_content = f.read()
                st.session_state.file_content = file_content
        except UnicodeDecodeError:
            st.warning("Cannot display preview for non-text files")
            st.session_state.file_content = "Binary content (cannot display preview)"

    # Process Button
    if st.button("Process Clinical Notes", type="primary"):
        if uploaded_file is not None or clinical_text:
            with st.spinner("Processing Clinical Notes..."):
                try:
                    # Phase 1: Extract Phenotypes
                    st.info("Phase 1: Extracting phenotypes...")
                    phenotypes = components["extractor"].extract_phenotypes(
                        file_path)
                    st.session_state.phenotypes = phenotypes

                    st.subheader("Validated Phenotypes")
                    phenotype_data = []
                    for p in st.session_state.phenotypes:
                        phenotype_data.append({
                            "Term ID": p.get("term_id", ""),
                            "Term Name": p.get("term_name", ""),
                            "Status": "Present" if p.get("present", True) else "Absent",
                            "Confidence": f"{p.get('confidence', 0)*100:.1f}%",
                            "Source": p.get("source", "extracted")
                        })

                    phenotype_df = pd.DataFrame(phenotype_data)
                    st.dataframe(phenotype_df, use_container_width=True)

                    # Visualization
                    st.subheader("Phenotype Visualization")
                    present_count = sum(
                        1 for p in st.session_state.phenotypes if p.get("present", True))
                    absent_count = len(
                        st.session_state.phenotypes) - present_count

                    fig = px.bar(
                        x=["Present", "Absent"],
                        y=[present_count, absent_count],
                        labels={"x": "Status", "y": "Count"},
                        title="Phenotype Status Distribution"
                    )

                    fig.update_traces(marker_color=["#0068c9", "#ff5252"])
                    st.plotly_chart(fig, use_container_width=True)

                    # Phase 2: Match Diseases
                    st.info("Phase 2: Matching diseases...")
                    phase2_results = components["matcher"].match_diseases(
                        file_path)
                    disease_matches = phase2_results.get("disease_matches", [])
                    st.session_state.disease_matches = disease_matches

                    st.subheader("Disease Matches")

                    if st.session_state.disease_matches:
                        # Convert to DataFrame for easier display
                        disease_data = []
                        for i, d in enumerate(st.session_state.disease_matches[:max_diseases]):
                            # Get proper disease name if available
                            disease_name = d.get(
                                "disease_name", d.get("disease_id", "Unknown"))
                            if d.get("disease_id", "").startswith("OMIM:"):
                                try:
                                    mim_id = d["disease_id"].split(":")[1]
                                    if hasattr(components["refiner"], "mim_titles") and components["refiner"].mim_titles and mim_id in components["refiner"].mim_titles:
                                        disease_name = components["refiner"].mim_titles[mim_id]
                                except:
                                    pass

                            disease_data.append({
                                "Rank": i+1,
                                "Disease": disease_name,
                                "ID": d.get("disease_id", ""),
                                "Match Score": f"{d.get('match_score', 0)*100:.1f}%",
                                "Matched Phenotypes": d.get("matched_phenotypes", ""),
                                "Associated Genes": ", ".join(d.get("associated_genes", []))
                            })

                        disease_df = pd.DataFrame(disease_data)
                        st.dataframe(disease_df, use_container_width=True)

                        # Visualization
                        st.subheader("Top Disease Matches")

                        if len(st.session_state.disease_matches) > 0:
                            top_n = min(
                                10, len(st.session_state.disease_matches))
                            fig = px.bar(
                                x=[d.get("disease_name", d.get("disease_id", ""))[:30]
                                    for d in st.session_state.disease_matches[:top_n]],
                                y=[d.get("match_score", 0)
                                    for d in st.session_state.disease_matches[:top_n]],
                                labels={"x": "Disease", "y": "Match Score"},
                                title="Top Disease Matches by Score"
                            )
                            st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.warning("No disease matches found.")

                    # Phase 3: Analyze Confidence
                    st.info("Phase 3: Analyzing diagnostic confidence...")
                    phase3_results = components["analyzer"].analyze_clinical_notes(
                        file_path)
                    st.session_state.diagnosis_assessment = phase3_results.get(
                        "diagnosis_assessment", {})

                    if st.session_state.phenotypes:
                        st.header("Diagnostic Results")

                    # Disease Matches Tab

                    # Diagnostic Assessment Tab
                        if st.session_state.diagnosis_assessment:
                            assessment = st.session_state.diagnosis_assessment

                            # Confidence level badge
                            confidence_level = assessment.get(
                                "confidence_level", "unknown")
                            confidence_color = {
                                "definitive": "green",
                                "probable": "orange",
                                "possible": "blue",
                                "novel": "purple",
                                "unknown": "gray"
                            }.get(confidence_level, "gray")

                            st.markdown(f"""
                            <div style="background-color: {confidence_color}; padding: 10px; border-radius: 5px; color: white; font-weight: bold; text-align: center; margin-bottom: 20px;">
                                Confidence Level: {confidence_level.upper()}
                            </div>
                            """, unsafe_allow_html=True)

                            col1, col2 = st.columns(2)
                            with col1:
                                st.subheader("Diagnostic Assessment")
                                st.write(
                                    f"**Explanation:** {assessment.get('explanation', 'Not available')}")
                                st.write(
                                    f"**Recommendation:** {assessment.get('recommendation', 'Not available')}")
                                st.write(
                                    f"**Threshold Used:** {assessment.get('threshold_used', 0):.2f}")
                                st.write(
                                    f"**Phenotype Count:** {assessment.get('phenotype_count', 0)}")

                            with col2:
                                if assessment.get("top_match"):
                                    st.subheader("Top Disease Match")
                                    top_match = assessment.get("top_match")

                                    # Get proper disease name if available
                                    disease_name = top_match.get(
                                        "disease_name", top_match.get("disease_id", "Unknown"))
                                    if top_match.get("disease_id", "").startswith("OMIM:"):
                                        try:
                                            mim_id = top_match["disease_id"].split(":")[
                                                1]
                                            if hasattr(components["refiner"], "mim_titles") and components["refiner"].mim_titles and mim_id in components["refiner"].mim_titles:
                                                disease_name = components["refiner"].mim_titles[mim_id]
                                        except:
                                            pass

                                    st.write(f"**Disease:** {disease_name}")
                                    st.write(
                                        f"**ID:** {top_match.get('disease_id', 'Unknown')}")
                                    st.write(
                                        f"**Match Score:** {top_match.get('match_score', 0)*100:.1f}%")
                                    st.write(
                                        f"**Matched Phenotypes:** {top_match.get('matched_phenotypes', 'Unknown')}")
                                    if top_match.get("associated_genes"):
                                        st.write(
                                            f"**Associated Genes:** {', '.join(top_match.get('associated_genes', []))}")

                            # Threshold visualization
                            st.subheader("Confidence Threshold Analysis")

                            # Create gauge chart for confidence
                            if assessment.get("top_match") and assessment.get("threshold_used"):
                                top_score = assessment["top_match"].get(
                                    "match_score", 0)
                                threshold = assessment["threshold_used"]

                                fig = go.Figure(go.Indicator(
                                    mode="gauge+number",
                                    value=top_score,
                                    domain={"x": [0, 1], "y": [0, 1]},
                                    title={"text": "Match Score vs Threshold"},
                                    gauge={
                                        "axis": {"range": [0, 1]},
                                        "bar": {"color": "darkblue"},
                                        "steps": [
                                            {"range": [0, threshold],
                                                "color": "lightgray"},
                                            {"range": [threshold, 1],
                                                "color": "lightgreen"}
                                        ],
                                        "threshold": {
                                            "line": {"color": "red", "width": 4},
                                            "thickness": 0.75,
                                            "value": threshold
                                        }
                                    }
                                ))

                                st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.warning("No diagnostic assessment available.")

                    # Initialize Phase 4 session
                    st.info("Phase 4: Preparing refinement session...")
                    refinement_state = components["refiner"].start_refinement_session(
                        file_path)
                    st.session_state.refinement_state = refinement_state

                    st.subheader("Interactive Phenotype Refinement")

                    if st.session_state.refinement_state:
                        if st.session_state.refinement_state.get("status") == "complete" and "message" in st.session_state.refinement_state:
                            st.info(
                                st.session_state.refinement_state["message"])

                            if "results" in st.session_state.refinement_state:
                                final_diagnosis = components["refiner"].get_final_diagnosis(
                                    st.session_state.refinement_state)

                                if "error" not in final_diagnosis:
                                    st.subheader("Final Diagnosis")
                                    st.write(
                                        f"**Confidence Level:** {final_diagnosis.get('confidence_level', 'Unknown')}")
                                    st.write(
                                        f"**Explanation:** {final_diagnosis.get('explanation', 'Not available')}")
                                    st.write(
                                        f"**Recommendation:** {final_diagnosis.get('recommendation', 'Not available')}")

                                    if "top_match" in final_diagnosis and final_diagnosis["top_match"]:
                                        top_match = final_diagnosis["top_match"]

                                        # Get proper disease name
                                        disease_name = components["refiner"]._get_disease_name(
                                            top_match['disease_id'])

                                        st.write(
                                            f"**Disease:** {disease_name}")
                                        st.write(
                                            f"**Match Score:** {top_match.get('match_score', 0)*100:.1f}%")

                                        if "associated_genes" in top_match and top_match["associated_genes"]:
                                            st.write(
                                                f"**Associated Genes:** {', '.join(top_match['associated_genes'])}")

                        elif st.session_state.refinement_state.get("status") == "in_progress":
                            st.write(
                                "This case could benefit from interactive refinement to improve diagnostic accuracy.")

                            if st.button("Start Interactive Refinement", key="start_refinement"):
                                st.session_state.interactive_mode = True

                            if st.session_state.interactive_mode:
                                questions = st.session_state.refinement_state.get(
                                    "questions", [])

                                if questions:
                                    st.subheader("Phenotype Questions")
                                    st.write(
                                        "Please answer the following questions to refine the diagnosis:")

                                    # Get the current question
                                    current_q_idx = st.session_state.current_question

                                    if current_q_idx < len(questions):
                                        current_question = questions[current_q_idx]

                                        st.write(
                                            f"**Question {current_q_idx + 1}/{len(questions)}:** {current_question['question']}")

                                        col1, col2 = st.columns(2)
                                        with col1:
                                            if st.button("Yes", key=f"yes_{current_q_idx}"):
                                                # Process this answer
                                                answers = ["y"]
                                                new_state = components["refiner"].process_answers(
                                                    st.session_state.refinement_state,
                                                    answers
                                                )

                                                if "error" not in new_state:
                                                    st.session_state.refinement_state = new_state
                                                    st.session_state.current_question = 0
                                                    st.rerun()
                                                else:
                                                    st.error(
                                                        f"Error: {new_state['error']}")

                                        with col2:
                                            if st.button("No", key=f"no_{current_q_idx}"):
                                                # Process this answer
                                                answers = ["n"]
                                                new_state = components["refiner"].process_answers(
                                                    st.session_state.refinement_state,
                                                    answers
                                                )

                                                if "error" not in new_state:
                                                    st.session_state.refinement_state = new_state
                                                    st.session_state.current_question = 0
                                                    st.rerun()
                                                else:
                                                    st.error(
                                                        f"Error: {new_state['error']}")

                                        # Navigation buttons
                                        nav_col1, nav_col2 = st.columns(2)
                                        with nav_col1:
                                            if current_q_idx > 0 and st.button("Previous Question"):
                                                st.session_state.current_question -= 1
                                                st.rerun()

                                        with nav_col2:
                                            if current_q_idx < len(questions) - 1 and st.button("Next Question"):
                                                st.session_state.current_question += 1
                                                st.rerun()

                                    else:
                                        st.success("All questions answered!")

                                        if st.session_state.refinement_state.get("status") == "complete":
                                            final_diagnosis = components["refiner"].get_final_diagnosis(
                                                st.session_state.refinement_state)

                                            if "error" not in final_diagnosis:
                                                st.subheader("Final Diagnosis")
                                                st.write(
                                                    f"**Confidence Level:** {final_diagnosis.get('confidence_level', 'Unknown')}")
                                                st.write(
                                                    f"**Explanation:** {final_diagnosis.get('explanation', 'Not available')}")
                                                st.write(
                                                    f"**Recommendation:** {final_diagnosis.get('recommendation', 'Not available')}")

                                                if "top_match" in final_diagnosis and final_diagnosis["top_match"]:
                                                    top_match = final_diagnosis["top_match"]

                                                    # Get proper disease name
                                                    disease_name = components["refiner"]._get_disease_name(
                                                        top_match['disease_id'])

                                                    st.write(
                                                        f"**Disease:** {disease_name}")
                                                    st.write(
                                                        f"**Match Score:** {top_match.get('match_score', 0)*100:.1f}%")
                                else:
                                    st.info("No more questions available.")
                                    st.session_state.interactive_mode = False
                    else:
                        st.warning(
                            "No refinement state available. Please process clinical notes first.")

                    # Phase 5: Analyze Novelty
                    st.info("Phase 5: Analyzing novelty...")
                    try:
                        novelty_analysis = components["detector"].analyze_novelty(
                            refinement_state) if hasattr(components["detector"], "analyze_novelty") else None
                        st.session_state.novelty_analysis = novelty_analysis
                    except Exception as e:
                        st.warning(f"Novelty analysis not available: {e}")
                        st.session_state.novelty_analysis = {"error": str(e)}

                    # Generate a unique session ID
                    st.session_state.session_id = int(time.time())

                    st.success("Processing complete! View the results below.")

                except Exception as e:
                    st.error(f"Error processing clinical notes: {e}")
                    logger.error(f"Processing error: {e}")
        else:
            st.warning(
                "Please upload a file or enter clinical notes to process.")
        st.subheader("Novelty Analysis")

        # Check if novelty analysis is available
        if "novelty_analysis" in st.session_state and st.session_state.novelty_analysis:
            novelty = st.session_state.novelty_analysis

            if "error" in novelty:
                st.error(f"Error in novelty analysis: {novelty['error']}")
            elif not hasattr(components["detector"], "analyze_novelty"):
                st.info(
                    "Novelty analysis functionality is not yet available in Phase 5.")
            else:
                # Display actual novelty results
                col1, col2 = st.columns(2)

                with col1:
                    st.metric(
                        label="Novelty Score",
                        value=f"{novelty.get('novelty_score', 0)*100:.1f}%",
                        help="Higher scores indicate greater likelihood of a novel disease or phenotypic expansion"
                    )

                with col2:
                    if "interpretation" in novelty:
                        interpretation = novelty["interpretation"]
                        st.write(
                            f"**Level:** {interpretation.get('level', 'unknown').upper()}")
                        st.write(
                            f"**Explanation:** {interpretation.get('explanation', 'Not available')}")

                # Novelty gauge chart if score is available
                if "novelty_score" in novelty:
                    fig = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=novelty.get('novelty_score', 0),
                        domain={"x": [0, 1], "y": [0, 1]},
                        title={"text": "Novelty Score"},
                        gauge={
                            "axis": {"range": [0, 1]},
                            "bar": {"color": "darkblue"},
                            "steps": [
                                {"range": [0, 0.4], "color": "green"},
                                {"range": [0.4, 0.6], "color": "yellow"},
                                {"range": [0.6, 0.8], "color": "orange"},
                                {"range": [0.8, 1], "color": "red"}
                            ],
                            "threshold": {
                                "line": {"color": "black", "width": 4},
                                "thickness": 0.75,
                                "value": novelty.get('novelty_score', 0)
                            }
                        }
                    ))

                    st.plotly_chart(fig, use_container_width=True)

                # Show unique phenotypes if available
                if "unique_phenotypes" in novelty and novelty["unique_phenotypes"]:
                    st.subheader("Unique Phenotypes")
                    st.write(
                        "These phenotypes are uncommon in known disease entities and may indicate a novel syndrome:")

                    unique_data = []
                    for i, p in enumerate(novelty["unique_phenotypes"][:10]):
                        unique_data.append({
                            "Rank": i+1,
                            "Term": p.get("term_name", ""),
                            "ID": p.get("term_id", ""),
                            "Uniqueness": f"{p.get('uniqueness_score', 0)*100:.1f}%",
                            "Information Content": f"{p.get('information_content', 0):.2f}"
                        })

                    unique_df = pd.DataFrame(unique_data)
                    st.dataframe(unique_df, use_container_width=True)

                # Display recommendations if available
                if "recommendation" in novelty:
                    st.subheader("Recommendations")
                    st.info(novelty["recommendation"])
            st.subheader("Export Results")

            col1, col2 = st.columns(2)

            with col1:
                if st.button("Export to JSON"):
                    # Prepare export data
                    export_data = {
                        "session_id": st.session_state.session_id,
                        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "file_content": st.session_state.file_content,
                        "validated_phenotypes": st.session_state.phenotypes,
                        "disease_matches": st.session_state.disease_matches,
                        "diagnosis_assessment": st.session_state.diagnosis_assessment,
                        "novelty_analysis": st.session_state.novelty_analysis
                    }

                    # Convert to JSON string
                    json_str = json.dumps(export_data, indent=2)

                    # Create a download link
                    st.download_button(
                        label="Download JSON",
                        data=json_str,
                        file_name=f"PhenoGenisis_results_{st.session_state.session_id}.json",
                        mime="application/json"
                    )

                with col2:
                    if st.button("Export to CSV"):
                        # Prepare phenotypes CSV
                        phenotype_data = []
                        for p in st.session_state.phenotypes:
                            phenotype_data.append({
                                "Term ID": p.get("term_id", ""),
                                "Term Name": p.get("term_name", ""),
                                "Status": "Present" if p.get("present", True) else "Absent",
                                "Confidence": p.get("confidence", 0),
                                "Source": p.get("source", "extracted")
                            })

                        phenotype_df = pd.DataFrame(phenotype_data)

                        # Prepare disease matches CSV
                        disease_data = []
                        for i, d in enumerate(st.session_state.disease_matches):
                            disease_data.append({
                                "Rank": i+1,
                                "Disease": d.get("disease_name", ""),
                                "ID": d.get("disease_id", ""),
                                "Match Score": d.get("match_score", 0),
                                "Matched Phenotypes": d.get("matched_phenotypes", ""),
                                "Associated Genes": "|".join(d.get("associated_genes", []))
                            })

                        disease_df = pd.DataFrame(disease_data)

                        # Combine into one CSV string
                        csv_buffer = StringIO()
                        csv_buffer.write("PhenoGenisis RESULTS\n\n")

                        csv_buffer.write("PHENOTYPES\n")
                        phenotype_df.to_csv(csv_buffer, index=False)

                        csv_buffer.write("\n\nDISEASE MATCHES\n")
                        disease_df.to_csv(csv_buffer, index=False)

                        # Add diagnostic assessment
                        csv_buffer.write("\n\nDIAGNOSTIC ASSESSMENT\n")
                        if st.session_state.diagnosis_assessment:
                            assessment = st.session_state.diagnosis_assessment
                            csv_buffer.write(
                                f"Confidence Level,{assessment.get('confidence_level', '')}\n")
                            csv_buffer.write(
                                f"Explanation,{assessment.get('explanation', '')}\n")
                            csv_buffer.write(
                                f"Recommendation,{assessment.get('recommendation', '')}\n")

                        # Add novelty analysis
                        csv_buffer.write("\n\nNOVELTY ANALYSIS\n")
                        if st.session_state.novelty_analysis:
                            novelty = st.session_state.novelty_analysis
                            csv_buffer.write(
                                f"Novelty Score,{novelty.get('novelty_score', '')}\n")
                            if "interpretation" in novelty:
                                csv_buffer.write(
                                    f"Level,{novelty['interpretation'].get('level', '')}\n")
                                csv_buffer.write(
                                    f"Explanation,{novelty['interpretation'].get('explanation', '')}\n")

                            # Create a download link

        else:
            st.warning(
                "No novelty analysis available. Please process clinical notes first.")


elif app_mode == "Interactive Diagnosis":
    st.header("Interactive Diagnosis")

    # Load system components
    components = load_system_components()

    st.write("""
    In this mode, you can interact directly with the PhenoGenisis system by selecting phenotypes
    and answering clarifying questions to reach a diagnosis.
    """)

    phenotype_search = st.text_input("Search for a phenotype term", "")

    if phenotype_search and len(phenotype_search) >= 3:
        # Search for matching HPO terms
        matching_terms = []

        for term_id, term_name in components["refiner"].diagnosis_analyzer.phenotype_matcher.phenotype_extractor.hpo_terms.items():
            if phenotype_search.lower() in term_name.lower():
                matching_terms.append(
                    {"term_id": term_id, "term_name": term_name})

        # Display matching terms
        if matching_terms:
            st.subheader(f"Found {len(matching_terms)} matching terms")

            selected_terms = []
            for i, term in enumerate(matching_terms[:20]):
                col1, col2 = st.columns([4, 1])
                with col1:
                    st.write(f"{term['term_name']} ({term['term_id']})")
                with col2:
                    if st.button("Add", key=f"add_{i}"):
                        if "manual_phenotypes" not in st.session_state:
                            st.session_state.manual_phenotypes = []

                        # Check if term is already added
                        if not any(p["term_id"] == term["term_id"] for p in st.session_state.manual_phenotypes):
                            st.session_state.manual_phenotypes.append({
                                "term_id": term["term_id"],
                                "term_name": term["term_name"],
                                "present": True,
                                "confidence": 1.0,
                                "source": "manual"
                            })
                            st.success(f"Added {term['term_name']}")
                            st.rerun()
        else:
            st.info("No matching terms found. Try a different search term.")

    # Display selected phenotypes
    if "manual_phenotypes" in st.session_state and st.session_state.manual_phenotypes:
        st.subheader("Selected Phenotypes")

        for i, p in enumerate(st.session_state.manual_phenotypes):
            col1, col2, col3 = st.columns([3, 1, 1])
            with col1:
                st.write(f"{p['term_name']} ({p['term_id']})")
            with col2:
                status = st.selectbox(
                    "Status",
                    ["Present", "Absent"],
                    index=0 if p.get("present", True) else 1,
                    key=f"status_{i}"
                )
                st.session_state.manual_phenotypes[i]["present"] = (
                    status == "Present")
            with col3:
                if st.button("Remove", key=f"remove_{i}"):
                    st.session_state.manual_phenotypes.pop(i)
                    st.rerun()

        # Match diseases button
        if st.button("Match Diseases", type="primary"):
            with st.spinner("Matching diseases..."):
                try:
                    # Convert manual phenotypes to format expected by matcher
                    phenotypes = st.session_state.manual_phenotypes

                    # Create a dummy file with the phenotypes
                    dummy_file = "uploads/manual_phenotypes.json"
                    with open(dummy_file, "w") as f:
                        json.dump(phenotypes, f)

                    # Phase 2: Match diseases directly with phenotypes
                    phase2_results = components["matcher"]._match_phenotypes_to_diseases(
                        phenotypes)
                    st.session_state.disease_matches = phase2_results

                    # Phase 3: Analyze confidence
                    present_phenotype_count = sum(
                        1 for p in phenotypes if p.get("present", True))
                    phase3_results = components["analyzer"]._analyze_top_matches(
                        phase2_results, present_phenotype_count)
                    st.session_state.diagnosis_assessment = phase3_results

                    # Create a dummy session state for refinement
                    refinement_state = {
                        "validated_phenotypes": phenotypes,
                        "disease_matches": phase2_results,
                        "confidence_assessment": phase3_results,
                        "status": "in_progress",
                        "iteration": 1,
                        "file_path": dummy_file,
                        "questions": [],
                        "previous_answers": []
                    }

                    # Generate questions
                    if phase2_results:
                        top_diseases = phase2_results[:3] if len(
                            phase2_results) >= 3 else phase2_results
                        discriminative_phenotypes = components["refiner"]._identify_discriminative_phenotypes(
                            top_diseases, phenotypes)
                        questions = components["refiner"]._generate_questions(
                            discriminative_phenotypes)
                        refinement_state["questions"] = questions

                    st.session_state.refinement_state = refinement_state

                    # Phase 5: Analyze novelty
                    try:
                        if hasattr(components["detector"], "analyze_novelty"):
                            novelty_analysis = components["detector"].analyze_novelty(
                                session_state=refinement_state)
                            st.session_state.novelty_analysis = novelty_analysis
                    except Exception as e:
                        st.warning(f"Novelty analysis not available: {e}")

                    st.success("Disease matching complete!")

                except Exception as e:
                    st.error(f"Error matching diseases: {e}")
                    logger.error(f"Disease matching error: {e}")

        # Display disease matches if available
        if "disease_matches" in st.session_state and st.session_state.disease_matches:
            st.subheader("Disease Matches")

            for i, d in enumerate(st.session_state.disease_matches[:5]):
                with st.expander(f"{i+1}. {d.get('disease_name', d.get('disease_id', 'Unknown'))} - Score: {d.get('match_score', 0)*100:.1f}%"):
                    # Get proper disease name if available
                    disease_name = d.get(
                        "disease_name", d.get("disease_id", "Unknown"))
                    if d.get("disease_id", "").startswith("OMIM:"):
                        try:
                            mim_id = d["disease_id"].split(":")[1]
                            if hasattr(components["refiner"], "mim_titles") and components["refiner"].mim_titles and mim_id in components["refiner"].mim_titles:
                                disease_name = components["refiner"].mim_titles[mim_id]
                        except:
                            pass

                    st.write(f"**Disease:** {disease_name}")
                    st.write(f"**ID:** {d.get('disease_id', 'Unknown')}")
                    st.write(
                        f"**Match Score:** {d.get('match_score', 0)*100:.1f}%")
                    st.write(
                        f"**Matched Phenotypes:** {d.get('matched_phenotypes', 'Unknown')}")
                    if d.get("associated_genes"):
                        st.write(
                            f"**Associated Genes:** {', '.join(d.get('associated_genes', []))}")

            # Display refinement questions if available
            if "refinement_state" in st.session_state and st.session_state.refinement_state:
                refinement_state = st.session_state.refinement_state

                if refinement_state.get("questions"):
                    st.subheader("Refinement Questions")
                    st.write("Answer these questions to refine your diagnosis:")

                    for i, q in enumerate(refinement_state["questions"]):
                        with st.expander(f"Question {i+1}: {q['question']}"):
                            col1, col2 = st.columns(2)
                            with col1:
                                if st.button("Yes", key=f"refine_yes_{i}"):
                                    # Process this answer
                                    answers = ["y"]
                                    new_state = components["refiner"].process_answers(
                                        st.session_state.refinement_state,
                                        answers
                                    )

                                    if "error" not in new_state:
                                        st.session_state.refinement_state = new_state

                                        # Update disease matches
                                        st.session_state.disease_matches = new_state["disease_matches"]

                                        # Update diagnosis assessment
                                        st.session_state.diagnosis_assessment = new_state[
                                            "confidence_assessment"]

                                        # Update novelty analysis
                                        try:
                                            if hasattr(components["detector"], "analyze_novelty"):
                                                novelty_analysis = components["detector"].analyze_novelty(
                                                    session_state=new_state)
                                                st.session_state.novelty_analysis = novelty_analysis
                                        except Exception as e:
                                            st.warning(
                                                f"Novelty analysis not available: {e}")

                                        st.success("Diagnosis refined!")
                                        st.rerun()
                                    else:
                                        st.error(
                                            f"Error: {new_state['error']}")

                            with col2:
                                if st.button("No", key=f"refine_no_{i}"):
                                    # Process this answer
                                    answers = ["n"]
                                    new_state = components["refiner"].process_answers(
                                        st.session_state.refinement_state,
                                        answers
                                    )

                                    if "error" not in new_state:
                                        st.session_state.refinement_state = new_state

                                        # Update disease matches
                                        st.session_state.disease_matches = new_state["disease_matches"]

                                        # Update diagnosis assessment
                                        st.session_state.diagnosis_assessment = new_state[
                                            "confidence_assessment"]

                                        # Update novelty analysis
                                        try:
                                            if hasattr(components["detector"], "analyze_novelty"):
                                                novelty_analysis = components["detector"].analyze_novelty(
                                                    session_state=new_state)
                                                st.session_state.novelty_analysis = novelty_analysis
                                        except Exception as e:
                                            st.warning(
                                                f"Novelty analysis not available: {e}")

                                        st.success("Diagnosis refined!")
                                        st.rerun()
                                    else:
                                        st.error(
                                            f"Error: {new_state['error']}")

            # Display diagnosis assessment if available
            if "diagnosis_assessment" in st.session_state and st.session_state.diagnosis_assessment:
                st.subheader("Diagnostic Assessment")

                assessment = st.session_state.diagnosis_assessment

                # Confidence level
                confidence_level = assessment.get(
                    "confidence_level", "unknown")
                st.write(f"**Confidence Level:** {confidence_level.upper()}")
                st.write(
                    f"**Explanation:** {assessment.get('explanation', 'Not available')}")
                st.write(
                    f"**Recommendation:** {assessment.get('recommendation', 'Not available')}")

            # Display novelty analysis if available
            if "novelty_analysis" in st.session_state and st.session_state.novelty_analysis:
                st.subheader("Novelty Analysis")

                novelty = st.session_state.novelty_analysis

                if "error" not in novelty:
                    st.write(
                        f"**Novelty Score:** {novelty.get('novelty_score', 0)*100:.1f}%")

                    if "interpretation" in novelty:
                        interpretation = novelty["interpretation"]
                        st.write(
                            f"**Level:** {interpretation.get('level', 'unknown').upper()}")
                        st.write(
                            f"**Explanation:** {interpretation.get('explanation', 'Not available')}")

                    if "recommendation" in novelty:
                        st.write(
                            f"**Recommendation:** {novelty['recommendation']}")
    else:
        st.info("Search and add phenotypes to begin the diagnostic process.")
