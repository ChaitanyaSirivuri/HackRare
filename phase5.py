"""
Phase 5: Novel Disease Flagging

Purpose: Identify cases that may represent novel diseases or presentations.
Calculate novelty scores based on the relationship between patient phenotypes and known diseases.

Author: Clinical Research Specialist
Date: March 2, 2025
"""

import logging
import os
import json
import numpy as np
from pathlib import Path
from phase4 import PhenotypeRefinementEngine


class NovelDiseaseDetector:
    def __init__(self,
                 hpo_path="data/hp.obo",
                 hpo_synonyms_file="data/hpo_synonyms.txt",
                 hpo_gene_disease_path="data/phenotype_to_genes.txt",
                 mim_titles_path="data/mimTitles.csv"):
        """
        Initialize the novel disease detector.

        Args:
            hpo_path (str): Path to HPO ontology file
            hpo_synonyms_file (str): Path to HPO synonyms file
            hpo_gene_disease_path (str): Path to phenotype-gene-disease mapping file
            mim_titles_path (str): Path to MIM titles file
        """
        self.logger = self._setup_logger()

        # Initialize refinement engine from Phase 4 (which includes Phases 1-3)
        self.refinement_engine = PhenotypeRefinementEngine(
            hpo_path=hpo_path,
            hpo_synonyms_file=hpo_synonyms_file,
            hpo_gene_disease_path=hpo_gene_disease_path,
            mim_titles_path=mim_titles_path
        )

        # Novelty scoring thresholds
        self.novelty_thresholds = {
            "high": 0.8,  # High likelihood of novel disease
            "moderate": 0.6,  # Moderate likelihood of novel disease
            "low": 0.4  # Low likelihood of novel disease
        }

    def _setup_logger(self):
        """Set up logging system for novel disease detection."""
        logger = logging.getLogger("novel_disease_detector")
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        return logger

    def _calculate_novelty_score(self, phenotypes, top_disease_matches):
        """
        Calculate novelty score based on information content of phenotypes.

        Args:
            phenotypes (list): Patient phenotypes
            top_disease_matches (list): Top disease matches

        Returns:
            float: Novelty score between 0 and 1
        """
        self.logger.info("Calculating novelty score...")

        # Extract present phenotype term IDs
        patient_phenotypes = [p['term_id']
                              for p in phenotypes if p.get('present', True)]

        if not patient_phenotypes:
            return 0.0

        # Extract term information content mapping from Phase 2
        term_ic = self.refinement_engine.diagnosis_analyzer.phenotype_matcher.term_information_content

        # If no disease matches, novelty is high
        if not top_disease_matches:
            return 1.0

        # Get top disease match and its phenotypes
        top_disease = top_disease_matches[0]
        top_disease_id = top_disease['disease_id']

        # Get phenotypes for top disease
        disease_phenotypes = self.refinement_engine.diagnosis_analyzer.phenotype_matcher.disease_to_hpo.get(
            top_disease_id, [])

        # Find shared phenotypes
        shared_phenotypes = set(patient_phenotypes).intersection(
            set(disease_phenotypes))

        # Calculate information content sums
        shared_ic_sum = sum(term_ic.get(term, 1.0)
                            for term in shared_phenotypes)
        patient_ic_sum = sum(term_ic.get(term, 1.0)
                             for term in patient_phenotypes)

        # Avoid division by zero
        if patient_ic_sum == 0:
            return 1.0

        # Calculate novelty score: 1 - (IC of shared terms / IC of all patient terms)
        novelty_score = 1.0 - (shared_ic_sum / patient_ic_sum)

        return max(0.0, min(1.0, novelty_score))

    def _interpret_novelty_score(self, novelty_score):
        """
        Interpret novelty score with explanation.

        Args:
            novelty_score (float): Calculated novelty score

        Returns:
            dict: Interpretation with level and explanation
        """
        if novelty_score >= self.novelty_thresholds["high"]:
            return {
                "level": "high",
                "explanation": "High likelihood of novel disease. This phenotype profile shows significant uniqueness compared to known disease entities."
            }
        elif novelty_score >= self.novelty_thresholds["moderate"]:
            return {
                "level": "moderate",
                "explanation": "Moderate likelihood of novel disease or atypical presentation of known disease. Consider broader genetic testing and expert consultation."
            }
        elif novelty_score >= self.novelty_thresholds["low"]:
            return {
                "level": "low",
                "explanation": "Low likelihood of novel disease. This likely represents an atypical presentation of a known disease or phenotypic expansion."
            }
        else:
            return {
                "level": "very_low",
                "explanation": "Very low likelihood of novel disease. This presentation is consistent with known disease patterns."
            }

    def _identify_unique_phenotypes(self, phenotypes, disease_matches):
        """
        Identify phenotypes unique to this patient compared to known diseases.

        Args:
            phenotypes (list): Patient phenotypes
            disease_matches (list): Top disease matches

        Returns:
            list: Unique phenotype terms with uniqueness scores
        """
        # Extract present phenotype terms
        patient_phenotypes = [
            {'term_id': p['term_id'], 'term_name': p['term_name']}
            for p in phenotypes if p.get('present', True)
        ]

        if not patient_phenotypes or not disease_matches:
            return patient_phenotypes

        # Get phenotypes for top diseases
        disease_phenotype_sets = []
        for disease in disease_matches[:5]:  # Consider top 5 matches
            disease_id = disease['disease_id']
            disease_phenotypes = self.refinement_engine.diagnosis_analyzer.phenotype_matcher.disease_to_hpo.get(
                disease_id, [])
            disease_phenotype_sets.append(set(disease_phenotypes))

        # Calculate uniqueness for each phenotype
        unique_phenotypes = []
        for phenotype in patient_phenotypes:
            term_id = phenotype['term_id']

            # Count how many top diseases have this phenotype
            disease_count = sum(
                1 for disease_set in disease_phenotype_sets if term_id in disease_set)

            # Calculate uniqueness score (1 = completely unique, 0 = in all top diseases)
            uniqueness_score = 1.0 - \
                (disease_count / len(disease_phenotype_sets)
                 ) if disease_phenotype_sets else 1.0

            # Add information content for weighting
            term_ic = self.refinement_engine.diagnosis_analyzer.phenotype_matcher.term_information_content.get(
                term_id, 1.0)

            # Add to results if reasonably unique
            if uniqueness_score > 0.5:
                unique_phenotypes.append({
                    'term_id': term_id,
                    'term_name': phenotype['term_name'],
                    'uniqueness_score': uniqueness_score,
                    'information_content': term_ic
                })

        # Sort by combined score of uniqueness and information content
        unique_phenotypes.sort(
            key=lambda x: x['uniqueness_score'] * x['information_content'],
            reverse=True
        )

        return unique_phenotypes

    def _generate_recommendation(self, novelty_score, confidence_level):
        """
        Generate recommendations based on novelty score and confidence level.

        Args:
            novelty_score (float): Calculated novelty score
            confidence_level (str): Confidence level from Phase 3/4

        Returns:
            str: Recommendation
        """
        if novelty_score >= self.novelty_thresholds["high"]:
            if confidence_level in ["definitive", "probable"]:
                return "Consider reporting phenotypic expansion of the identified disease. Recommend comprehensive genomic analysis to identify potential molecular basis."
            else:
                return "High likelihood of novel disease. Recommend comprehensive genomic analysis (WES/WGS), detailed phenotyping, and expert consultation."

        elif novelty_score >= self.novelty_thresholds["moderate"]:
            if confidence_level in ["definitive", "probable"]:
                return "Potential phenotypic expansion. Consider targeted panel testing for the identified disease plus comprehensive genomic analysis."
            else:
                return "Moderate likelihood of novel disease or atypical presentation. Recommend detailed phenotyping, comprehensive genomic analysis, and functional studies."

        else:
            if confidence_level == "definitive":
                return "Presentation consistent with known disease. Recommend standard management for the identified condition."
            elif confidence_level == "probable":
                return "Likely typical presentation of known disease. Recommend targeted testing for top candidate diseases."
            else:
                return "Insufficient evidence for novelty assessment. Recommend additional phenotyping and targeted testing for candidate diseases."

    def analyze_novelty(self, session_state=None, file_path=None):
        """
        Analyze potential novelty of a disease presentation using Phase 4 results.

        Args:
            session_state (dict): Session state from Phase 4
            file_path (str): Path to clinical notes file (if session state not provided)

        Returns:
            dict: Novelty analysis results
        """
        self.logger.info("Analyzing disease novelty...")

        try:
            # Get diagnostic results from Phase 4 if not provided
            if session_state is None:
                if file_path is None:
                    return {"error": "Either session state or file path must be provided"}

                # Start a refinement session
                session_state = self.refinement_engine.start_refinement_session(
                    file_path)

                if "error" in session_state:
                    return {"error": session_state["error"]}

            # Handle the case where refinement wasn't needed (definitive diagnosis)
            if session_state.get("status") == "complete" and "results" in session_state:
                results = session_state["results"]
                validated_phenotypes = results.get("validated_phenotypes", [])
                disease_matches = results.get("disease_matches", [])
                confidence_assessment = results.get("diagnosis_assessment", {})
            else:
                # Extract validated phenotypes and disease matches from session state
                validated_phenotypes = session_state.get(
                    "validated_phenotypes", [])
                disease_matches = session_state.get("disease_matches", [])
                confidence_assessment = session_state.get(
                    "confidence_assessment", {})

            # Calculate novelty score
            novelty_score = self._calculate_novelty_score(
                validated_phenotypes, disease_matches)

            # Interpret novelty score
            novelty_interpretation = self._interpret_novelty_score(
                novelty_score)

            # Identify unique phenotypes
            unique_phenotypes = self._identify_unique_phenotypes(
                validated_phenotypes, disease_matches)

            # Generate recommendation based on novelty and confidence
            confidence_level = confidence_assessment.get(
                "confidence_level", "unknown")
            recommendation = self._generate_recommendation(
                novelty_score, confidence_level)

            # Compile novelty analysis results
            novelty_analysis = {
                "novelty_score": novelty_score,
                "interpretation": novelty_interpretation,
                "unique_phenotypes": unique_phenotypes,
                "recommendation": recommendation
            }

            return novelty_analysis

        except Exception as e:
            self.logger.error(f"Error analyzing novelty: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return {"error": str(e)}

    def process_case(self, file_path, output_path=None):
        """
        Process a case through all phases and perform novelty analysis.

        Args:
            file_path (str): Path to clinical notes file
            output_path (str, optional): Path to save results

        Returns:
            dict: Complete case analysis including novelty assessment
        """
        self.logger.info(f"Processing complete case analysis for {file_path}")

        try:
            # Start with refinement session (Phase 4)
            refinement_state = self.refinement_engine.start_refinement_session(
                file_path)

            if "error" in refinement_state:
                return {"error": refinement_state["error"]}

            # For definitive diagnoses, we can just use the results directly
            if refinement_state.get("status") == "complete" and "results" in refinement_state:
                # Get diagnostic assessment from Phase 3 results
                phase3_results = refinement_state["results"]
                final_diagnosis = self.refinement_engine.get_final_diagnosis(
                    refinement_state)
            else:
                # For cases needing refinement, get final diagnosis after refinement
                if refinement_state.get("status") == "in_progress":
                    # Auto-answer all questions with "yes" for automated processing
                    # In a real implementation, this would be interactive
                    while refinement_state.get("status") == "in_progress":
                        questions = refinement_state.get("questions", [])
                        if not questions:
                            break

                        # Auto-generate "yes" answers
                        answers = ["y"] * len(questions)

                        # Process answers
                        refinement_state = self.refinement_engine.process_answers(
                            refinement_state, answers)

                        if "error" in refinement_state:
                            return {"error": refinement_state["error"]}

                # Get final diagnosis
                final_diagnosis = self.refinement_engine.get_final_diagnosis(
                    refinement_state)

                if "error" in final_diagnosis:
                    return {"error": final_diagnosis["error"]}

                # Use the updated diagnostic results
                phase3_results = {
                    "validated_phenotypes": final_diagnosis["validated_phenotypes"],
                    "disease_matches": final_diagnosis["disease_matches"]
                }

            # Run novelty analysis
            novelty_analysis = self.analyze_novelty(
                session_state=refinement_state)

            # Compile comprehensive results
            complete_results = {
                "validated_phenotypes": final_diagnosis.get("validated_phenotypes", []),
                "disease_matches": final_diagnosis.get("disease_matches", []),
                "confidence_level": final_diagnosis.get("confidence_level", "unknown"),
                "explanation": final_diagnosis.get("explanation", ""),
                "recommendation": final_diagnosis.get("recommendation", ""),
                "top_match": final_diagnosis.get("top_match", {}),
                "refinement_iterations": final_diagnosis.get("refinement_iterations", 0),
                "total_phenotypes": final_diagnosis.get("total_phenotypes", 0),
                "present_phenotypes": final_diagnosis.get("present_phenotypes", 0),
                "novelty_analysis": novelty_analysis
            }

            # Save results if output path provided
            if output_path:
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                with open(output_path, 'w') as f:
                    json.dump(complete_results, f, indent=2)
                self.logger.info(f"Complete results saved to {output_path}")

            return complete_results

        except Exception as e:
            self.logger.error(f"Error processing case: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return {"error": str(e)}


def run_novelty_analysis(file_path, output_path=None):
    """
    Run novelty analysis on a clinical notes file.

    Args:
        file_path (str): Path to clinical notes file
        output_path (str, optional): Path to save results
    """
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Initialize detector
    detector = NovelDiseaseDetector()

    # Process the case
    results = detector.process_case(file_path, output_path)

    if "error" in results:
        print(f"Error: {results['error']}")
        return

    # Display novelty analysis
    novelty = results["novelty_analysis"]

    print("\n===== NOVELTY ANALYSIS =====")
    print(f"Novelty Score: {novelty.get('novelty_score', 0):.4f}")

    if "interpretation" in novelty:
        interpretation = novelty["interpretation"]
        print(f"Interpretation: {interpretation.get('level', '').upper()}")
        print(f"Explanation: {interpretation.get('explanation', '')}")

    if "recommendation" in novelty:
        print(f"Recommendation: {novelty['recommendation']}")

    if "unique_phenotypes" in novelty and novelty["unique_phenotypes"]:
        print("\n----- UNIQUE PHENOTYPES -----")
        for i, phenotype in enumerate(novelty["unique_phenotypes"][:5]):
            print(
                f"{i+1}. {phenotype['term_name']} - Uniqueness: {phenotype['uniqueness_score']:.2f}")

    # Display diagnostic assessment
    print("\n===== FINAL DIAGNOSIS =====")
    print(f"Confidence Level: {results['confidence_level']}")
    print(f"Explanation: {results['explanation']}")
    print(f"Recommendation: {results['recommendation']}")

    print(
        f"\nRefinement completed after {results['refinement_iterations']} iterations.")
    print(
        f"Total phenotypes: {results['total_phenotypes']}, Present: {results['present_phenotypes']}")

    print("\n===== TOP DISEASE MATCH =====")
    top_match = results["top_match"]
    if top_match:
        # Get proper disease name
        disease_name = detector.refinement_engine._get_disease_name(
            top_match['disease_id'])
        print(f"Disease: {disease_name}")
        print(f"Match Score: {top_match['match_score']:.4f}")

        if "associated_genes" in top_match and top_match["associated_genes"]:
            print(
                f"Associated Genes: {', '.join(top_match['associated_genes'])}")
    else:
        print("No disease matches found.")


# Example usage
if __name__ == "__main__":
    # Create data directory if it doesn't exist
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)

    # Process a sample clinical note
    file_path = "clinical_notes.txt"
    output_path = "results/complete_analysis.json"

    if not os.path.exists(file_path):
        print(f"Error: File {file_path} not found.")
    else:
        # Run novelty analysis
        run_novelty_analysis(file_path, output_path)
