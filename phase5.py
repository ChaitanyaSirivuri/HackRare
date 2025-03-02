"""
Phase 5: Novel Disease Flagging

Purpose: Identify cases that may represent novel diseases or presentations.
Calculate novelty scores based on the relationship between patient phenotypes and known diseases.

Author: Clinical Research Specialist
Date: March 1, 2025
"""

import logging
import os
import json
import numpy as np
import pandas as pd
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

    def _calculate_phenotype_cluster_distance(self, phenotypes, disease_matches):
        """
        Calculate how the patient's phenotypes cluster compared to known diseases.

        Args:
            phenotypes (list): Patient phenotypes
            disease_matches (list): Top disease matches

        Returns:
            dict: Cluster analysis results
        """
        # Extract present phenotype term IDs
        patient_phenotypes = [p['term_id']
                              for p in phenotypes if p.get('present', True)]

        if not patient_phenotypes or len(disease_matches) < 2:
            return {
                "cluster_variance": 0.0,
                "disease_variance": 0.0,
                "interpretation": "Insufficient data for cluster analysis"
            }

        # Extract term information content mapping
        term_ic = self.refinement_engine.diagnosis_analyzer.phenotype_matcher.term_information_content

        # Calculate pairwise disease distances
        disease_distances = []

        for i, disease1 in enumerate(disease_matches[:10]):
            disease1_id = disease1['disease_id']
            disease1_terms = set(
                self.refinement_engine.diagnosis_analyzer.phenotype_matcher.disease_to_hpo.get(disease1_id, []))

            for j, disease2 in enumerate(disease_matches[:10]):
                if i >= j:
                    continue

                disease2_id = disease2['disease_id']
                disease2_terms = set(
                    self.refinement_engine.diagnosis_analyzer.phenotype_matcher.disease_to_hpo.get(disease2_id, []))

                # Calculate similarity using Jaccard index
                shared_terms = disease1_terms.intersection(disease2_terms)
                all_terms = disease1_terms.union(disease2_terms)

                if not all_terms:
                    continue

                similarity = len(shared_terms) / len(all_terms)
                distance = 1.0 - similarity
                disease_distances.append(distance)

        # Calculate patient's distance to top diseases
        patient_distances = []
        patient_phenotypes_set = set(patient_phenotypes)

        for disease in disease_matches[:10]:
            disease_id = disease['disease_id']
            disease_terms = set(
                self.refinement_engine.diagnosis_analyzer.phenotype_matcher.disease_to_hpo.get(disease_id, []))

            if not disease_terms:
                continue

            shared_terms = patient_phenotypes_set.intersection(disease_terms)
            all_terms = patient_phenotypes_set.union(disease_terms)

            if not all_terms:
                continue

            similarity = len(shared_terms) / len(all_terms)
            distance = 1.0 - similarity
            patient_distances.append(distance)

        # Calculate variance of distances
        disease_variance = np.var(
            disease_distances) if disease_distances else 0.0
        patient_variance = np.var(
            patient_distances) if patient_distances else 0.0

        # Compare patient variance to disease variance
        if disease_variance == 0:
            cluster_ratio = 1.0
        else:
            cluster_ratio = patient_variance / disease_variance

        # Interpret the results
        if cluster_ratio > 2.0:
            interpretation = "Patient phenotypes suggest a novel disease pattern that doesn't cluster with known diseases"
        elif cluster_ratio > 1.2:
            interpretation = "Patient phenotypes show moderate deviation from known disease clusters"
        else:
            interpretation = "Patient phenotypes cluster within established disease patterns"

        return {
            "cluster_ratio": cluster_ratio,
            "patient_variance": patient_variance,
            "disease_variance": disease_variance,
            "interpretation": interpretation
        }

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

            # Extract validated phenotypes and disease matches
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

            # Calculate phenotype clustering
            cluster_analysis = self._calculate_phenotype_cluster_distance(
                validated_phenotypes, disease_matches)

            # Convert OMIM IDs to disease names in unique phenotypes
            for phenotype in unique_phenotypes:
                if 'relevant_diseases' in phenotype:
                    phenotype['relevant_diseases_names'] = [
                        self.refinement_engine._get_disease_name(disease_id)
                        for disease_id in phenotype['relevant_diseases']
                    ]

            # Calculate confidence-novelty relationship
            confidence_level = confidence_assessment.get(
                "confidence_level", "unknown")

            if confidence_level == "definitive" and novelty_score > self.novelty_thresholds["moderate"]:
                confidence_novelty_relationship = "Unusual: High confidence diagnosis with significant novelty suggests phenotypic expansion"
            elif confidence_level == "definitive":
                confidence_novelty_relationship = "Expected: High confidence diagnosis with low novelty"
            elif confidence_level == "probable" and novelty_score > self.novelty_thresholds["moderate"]:
                confidence_novelty_relationship = "Concerning: Moderate confidence with high novelty suggests potentially novel disease"
            elif novelty_score > self.novelty_thresholds["high"]:
                confidence_novelty_relationship = "Critical: Low confidence diagnosis with high novelty strongly suggests novel disease"
            else:
                confidence_novelty_relationship = "Typical: Confidence and novelty levels are consistent with known disease patterns"

            # Compile novelty analysis results
            novelty_analysis = {
                "novelty_score": novelty_score,
                "interpretation": novelty_interpretation,
                "unique_phenotypes": unique_phenotypes,
                "cluster_analysis": cluster_analysis,
                "confidence_novelty_relationship": confidence_novelty_relationship,
                "recommendation": self._generate_recommendation(novelty_score, confidence_level)
            }

            return novelty_analysis

        except Exception as e:
            self.logger.error(f"Error analyzing novelty: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return {"error": str(e)}

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

    def process_final_diagnosis(self, session_state=None, file_path=None, output_path=None):
        """
        Process a complete diagnosis through all phases (1-5).

        Args:
            session_state (dict): Session state from Phase 4
            file_path (str): Path to clinical notes file
            output_path (str): Path to save results

        Returns:
            dict: Complete analysis including novelty assessment
        """
        self.logger.info("Generating complete diagnostic report...")

        try:
            # Get or create a session state
            if session_state is None and file_path is not None:
                # Start a Phase 4 session
                session_state = self.refinement_engine.start_refinement_session(
                    file_path)

                if session_state.get("status") == "in_progress":
                    # Run iterative refinement
                    self.logger.info(
                        "Running interactive refinement session...")

                    iteration = 1
                    max_iterations = 3

                    while session_state.get("status") == "in_progress" and iteration <= max_iterations:
                        questions = session_state.get("questions", [])
                        if not questions:
                            break

                        # Generate automated answers for non-interactive mode
                        self.logger.info(
                            f"Automated refinement iteration {iteration}...")

                        answers = []
                        for question in questions:
                            # Default answer is 'y' (simplistic approach for automated refinement)
                            answers.append('y')

                        # Process the answers
                        session_state = self.refinement_engine.process_answers(
                            session_state, answers)

                        if "error" in session_state:
                            return {"error": session_state["error"]}

                        iteration += 1

                # Get final diagnosis
                if session_state.get("status") == "complete":
                    final_diagnosis = self.refinement_engine.get_final_diagnosis(
                        session_state)

                    # Update session state with final diagnosis
                    session_state.update(final_diagnosis)

            if session_state is None:
                return {"error": "No valid session state or file path provided"}

            # Run novelty analysis
            novelty_analysis = self.analyze_novelty(session_state)

            # Compile complete results
            complete_results = {
                "validated_phenotypes": session_state.get("validated_phenotypes", []),
                "disease_matches": session_state.get("disease_matches", []),
                "confidence_assessment": session_state.get("confidence_assessment", {}),
                "refinement_iterations": session_state.get("iteration", 1) - 1,
                "total_phenotypes": len(session_state.get("validated_phenotypes", [])),
                "present_phenotypes": sum(1 for p in session_state.get("validated_phenotypes", []) if p.get('present', True)),
                "novelty_analysis": novelty_analysis
            }

            # Convert OMIM IDs to disease names in disease matches
            for disease in complete_results["disease_matches"]:
                disease_id = disease.get("disease_id", "")
                if disease_id.startswith("OMIM:"):
                    disease_name = self.refinement_engine._get_disease_name(
                        disease_id)
                    disease["disease_name"] = disease_name

            # Save results if output path provided
            if output_path:
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                with open(output_path, 'w') as f:
                    json.dump(complete_results, f, indent=2)
                self.logger.info(f"Complete results saved to {output_path}")

            return complete_results

        except Exception as e:
            self.logger.error(f"Error processing diagnosis: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return {"error": str(e)}


def run_complete_analysis(file_path, output_path=None, interactive=False):
    """
    Run a complete analysis through all phases (1-5).

    Args:
        file_path (str): Path to clinical notes file
        output_path (str, optional): Path to save results
        interactive (bool): Whether to run Phase 4 interactively
    """
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Initialize novelty detector
    detector = NovelDiseaseDetector()

    if interactive:
        # Run interactive Phase 4 session
        print("\n===== STARTING INTERACTIVE REFINEMENT SESSION =====")
        refinement_engine = detector.refinement_engine
        session_state = refinement_engine.start_refinement_session(file_path)

        if "error" in session_state:
            print(f"Error: {session_state['error']}")
            return

        if session_state["status"] == "complete":
            print(f"Note: {session_state['message']}")
        else:
            # Run interactive session
            session_state = run_interactive_session(
                refinement_engine, file_path, session_state)

        # Get final diagnosis
        final_diagnosis = refinement_engine.get_final_diagnosis(session_state)

        # Run novelty analysis
        novelty_analysis = detector.analyze_novelty(session_state)

        # Create a results dict for consistent handling
        results = {
            "refinement_iterations": session_state["iteration"] - 1,
            "total_phenotypes": len(session_state.get("validated_phenotypes", [])),
            "present_phenotypes": sum(1 for p in session_state.get("validated_phenotypes", []) if p.get('present', True))
        }
    else:
        # Run automated analysis
        results = detector.process_final_diagnosis(
            file_path=file_path, output_path=output_path)

        if "error" in results:
            print(f"Error: {results['error']}")
            return

        # Extract relevant parts for display
        final_diagnosis = {
            "confidence_level": results.get("confidence_assessment", {}).get("confidence_level", "unknown"),
            "explanation": results.get("confidence_assessment", {}).get("explanation", ""),
            "recommendation": results.get("confidence_assessment", {}).get("recommendation", ""),
            "top_match": next((d for d in results.get("disease_matches", []) if d), {})
        }

        novelty_analysis = results.get("novelty_analysis", {})

    # Display novelty analysis
    print("\n===== NOVELTY ANALYSIS =====")
    print(f"Novelty Score: {novelty_analysis.get('novelty_score', 0):.4f}")

    if "interpretation" in novelty_analysis:
        interpretation = novelty_analysis["interpretation"]
        print(f"Interpretation: {interpretation.get('level', '').upper()}")
        print(f"Explanation: {interpretation.get('explanation', '')}")

    if "recommendation" in novelty_analysis:
        print(f"Recommendation: {novelty_analysis['recommendation']}")

    if "unique_phenotypes" in novelty_analysis and novelty_analysis["unique_phenotypes"]:
        print("\n----- UNIQUE PHENOTYPES -----")
        for i, phenotype in enumerate(novelty_analysis["unique_phenotypes"][:5]):
            print(
                f"{i+1}. {phenotype['term_name']} - Uniqueness: {phenotype['uniqueness_score']:.2f}")

    # Display diagnostic assessment
    print("\n===== FINAL DIAGNOSIS =====")
    print(f"Confidence Level: {final_diagnosis['confidence_level']}")
    print(f"Explanation: {final_diagnosis['explanation']}")
    print(f"Recommendation: {final_diagnosis['recommendation']}")

    # Display refinement information
    if isinstance(results, dict):
        print(
            f"\nRefinement completed after {results.get('refinement_iterations', 0)} iterations.")
        print(
            f"Total phenotypes: {results.get('total_phenotypes', 0)}, Present: {results.get('present_phenotypes', 0)}")

    # Display top disease match
    print("\n===== TOP DISEASE MATCH =====")
    top_match = final_diagnosis.get("top_match", {})
    if top_match:
        # Get proper disease name
        disease_id = top_match.get('disease_id', '')
        if disease_id.startswith("OMIM:"):
            disease_name = detector.refinement_engine._get_disease_name(
                disease_id)
            print(f"Disease: {disease_name}")
        else:
            print(f"Disease: {top_match.get('disease_name', disease_id)}")

        print(f"Match Score: {top_match.get('match_score', 0):.4f}")

        if "associated_genes" in top_match and top_match["associated_genes"]:
            print(
                f"Associated Genes: {', '.join(top_match['associated_genes'])}")
    else:
        print("No disease matches found.")


def run_interactive_session(refinement_engine, file_path, session_state=None):
    """
    Run an interactive refinement session and return the final session state.

    Args:
        refinement_engine: PhenotypeRefinementEngine instance
        file_path (str): Path to clinical notes file
        session_state (dict, optional): Existing session state

    Returns:
        dict: Final session state
    """
    if session_state is None:
        session_state = refinement_engine.start_refinement_session(file_path)

    if "error" in session_state:
        print(f"Error: {session_state['error']}")
        return session_state

    if session_state["status"] == "complete":
        print(f"Note: {session_state['message']}")
        return session_state

    # Process iterations
    iteration = 1
    while session_state["status"] == "in_progress":
        print(f"\n----- Refinement Iteration {iteration} -----")

        # Show current top matches
        top_matches = session_state["disease_matches"][:3]
        print("\nCurrent Top Matches:")
        for i, match in enumerate(top_matches):
            disease_name = refinement_engine._get_disease_name(
                match['disease_id'])
            print(f"{i+1}. {disease_name} - Score: {match['match_score']:.4f}")

        # Display questions
        questions = session_state["questions"]
        if not questions:
            print("\nNo more questions available. Completing refinement.")
            break

        print("\nPlease answer the following questions (y/n):")
        answers = []

        for i, question in enumerate(questions):
            while True:
                answer = input(
                    f"{i+1}. {question['question']} (y/n): ").strip().lower()
                if answer in ['y', 'n']:
                    answers.append(answer)
                    break
                else:
                    print("Please enter 'y' for yes or 'n' for no.")

        # Process answers
        session_state = refinement_engine.process_answers(
            session_state, answers)

        if "error" in session_state:
            print(f"Error: {session_state['error']}")
            return session_state

        iteration += 1

    return session_state


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
        # Run complete analysis (set interactive=True for interactive mode)
        run_complete_analysis(file_path, output_path, interactive=True)
