"""
Phase 3: Dynamic Threshold Analysis

Purpose: Determine whether current disease matches are definitive, probable, or require refinement.

Author: Clinical Decision Support Specialist
Date: March 1, 2025
"""

import os
import json
import logging
from pathlib import Path
from phase2 import PhenotypeDiseaseMatcher


class DiagnosisConfidenceAnalyzer:
    """
    Phase 3: Analyzes disease matches from Phase 2 to determine diagnostic confidence
    based on dynamic thresholds that adapt to the quantity of phenotypes provided.
    """

    def __init__(self,
                 hpo_path="data/hp.obo",
                 hpo_synonyms_file="data/hpo_synonyms.txt",
                 hpo_gene_disease_path="data/phenotype_to_genes.txt"):
        """
        Initialize the confidence analysis component.

        Args:
            hpo_path (str): Path to HPO ontology file
            hpo_synonyms_file (str): Path to HPO synonyms file
            hpo_gene_disease_path (str): Path to phenotype-gene-disease mapping file
        """
        self.logger = self._setup_logger()

        # Initialize Phase 1 and Phase 2 components
        self.phenotype_matcher = PhenotypeDiseaseMatcher(
            hpo_path=hpo_path,
            hpo_synonyms_file=hpo_synonyms_file,
            hpo_gene_disease_path=hpo_gene_disease_path
        )

        # Define confidence thresholds based on phenotype count
        self.confidence_thresholds = {
            "few_phenotypes": {  # ≤5 phenotypes
                "threshold": 0.95,
                "max_count": 5
            },
            "moderate_phenotypes": {  # 6-10 phenotypes
                "threshold": 0.85,
                "min_count": 6,
                "max_count": 10
            },
            "many_phenotypes": {  # ≥11 phenotypes
                "threshold": 0.65,
                "min_count": 11
            }
        }

        # Define diagnostic confidence levels
        self.confidence_levels = {
            "definitive": "Definitive diagnosis with high confidence",
            "probable": "Probable diagnosis with moderate confidence",
            "possible": "Possible diagnosis with low confidence, refinement recommended",
            "novel": "No strong match to known diseases, may represent novel phenotype"
        }

    def _setup_logger(self):
        """Set up logging system for the pipeline."""
        logger = logging.getLogger("diagnosis_analyzer")
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        return logger

    def _determine_confidence_threshold(self, phenotype_count):
        """
        Determine the appropriate confidence threshold based on phenotype count.

        Args:
            phenotype_count (int): Number of phenotypes provided

        Returns:
            float: Confidence threshold for this phenotype count
        """
        if phenotype_count <= self.confidence_thresholds["few_phenotypes"]["max_count"]:
            return self.confidence_thresholds["few_phenotypes"]["threshold"]
        elif self.confidence_thresholds["moderate_phenotypes"]["min_count"] <= phenotype_count <= self.confidence_thresholds["moderate_phenotypes"]["max_count"]:
            return self.confidence_thresholds["moderate_phenotypes"]["threshold"]
        # phenotype_count >= self.confidence_thresholds["many_phenotypes"]["min_count"]
        else:
            return self.confidence_thresholds["many_phenotypes"]["threshold"]

    def _analyze_top_matches(self, disease_matches, present_phenotype_count):
        """
        Analyze top disease matches against dynamic thresholds.

        Args:
            disease_matches (list): Ranked disease matches from Phase 2
            present_phenotype_count (int): Number of present phenotypes

        Returns:
            dict: Diagnostic assessment including confidence level and top matches
        """
        self.logger.info(
            f"Analyzing matches for {present_phenotype_count} phenotypes")

        # Determine appropriate threshold based on phenotype count
        confidence_threshold = self._determine_confidence_threshold(
            present_phenotype_count)
        self.logger.info(f"Using confidence threshold: {confidence_threshold}")

        # Extract top matches
        top_matches = disease_matches[:5] if len(
            disease_matches) >= 5 else disease_matches

        # Check if any match meets the threshold
        if not top_matches:
            confidence_level = "novel"
            diagnosis_assessment = {
                "confidence_level": confidence_level,
                "explanation": self.confidence_levels[confidence_level],
                "top_matches": [],
                "threshold_used": confidence_threshold,
                "phenotype_count": present_phenotype_count,
                "recommendation": "Consider genome sequencing and submission to undiagnosed disease programs"
            }
            return diagnosis_assessment

        top_score = top_matches[0]["match_score"]
        second_score = top_matches[1]["match_score"] if len(
            top_matches) > 1 else 0

        # Analyze match distribution
        score_differential = top_score - second_score
        # 15% difference indicates clear leader
        clear_leading_match = score_differential > 0.15

        # Determine confidence level
        if top_score >= confidence_threshold and clear_leading_match:
            confidence_level = "definitive"
            recommendation = "Proceed with targeted genetic testing for the top match"
        elif top_score >= confidence_threshold * 0.9:  # Within 10% of threshold
            confidence_level = "probable"
            recommendation = "Consider targeted panel testing for top 2-3 matches"
        elif top_score >= confidence_threshold * 0.7:  # Within 30% of threshold
            confidence_level = "possible"
            recommendation = "Consider phenotype refinement or broader genetic testing"
        else:
            confidence_level = "novel"
            recommendation = "Consider genome sequencing and submission to undiagnosed disease programs"

        # Detect if multiple similar-scoring diagnoses exist (potential differential diagnoses)
        has_differential = False
        differential_candidates = []

        for i, match in enumerate(top_matches[:3]):  # Check top 3 matches
            if i > 0 and match["match_score"] > top_score * 0.85:  # Within 15% of top score
                has_differential = True
                differential_candidates.append(match)

        # Add differential diagnoses if they exist
        if has_differential:
            differential = [top_matches[0]] + differential_candidates
            differential_info = "Multiple similar-scoring diagnoses detected, consider as differential diagnoses"
        else:
            differential = [top_matches[0]]
            differential_info = "Clear leading diagnosis detected"

        # Assemble final diagnostic assessment
        diagnosis_assessment = {
            "confidence_level": confidence_level,
            "explanation": self.confidence_levels[confidence_level],
            "top_match": top_matches[0],
            "top_matches": top_matches,
            "differential_diagnoses": differential,
            "differential_info": differential_info,
            "threshold_used": confidence_threshold,
            "phenotype_count": present_phenotype_count,
            "recommendation": recommendation
        }

        return diagnosis_assessment

    def _count_present_phenotypes(self, phenotypes):
        """
        Count the number of present (vs. absent) phenotypes.

        Args:
            phenotypes (list): List of phenotype dictionaries

        Returns:
            int: Count of present phenotypes
        """
        return sum(1 for p in phenotypes if p.get('present', True))

    def _analyze_results_distribution(self, disease_matches):
        """
        Analyze the distribution of match scores for additional insights.

        Args:
            disease_matches (list): Ranked disease matches

        Returns:
            dict: Distribution analysis results
        """
        if not disease_matches:
            return {
                "distribution": "empty",
                "score_range": 0,
                "grouped_matches": {}
            }

        # Calculate score statistics
        scores = [d["match_score"] for d in disease_matches]
        max_score = max(scores) if scores else 0
        min_score = min(scores) if scores else 0
        score_range = max_score - min_score

        # Group matches by score brackets
        grouped_matches = {
            "high_confidence": [],    # >90% of max score
            "medium_confidence": [],  # 70-90% of max score
            "low_confidence": []      # <70% of max score
        }

        for match in disease_matches:
            relative_score = match["match_score"] / \
                max_score if max_score > 0 else 0

            if relative_score > 0.9:
                grouped_matches["high_confidence"].append(match)
            elif relative_score > 0.7:
                grouped_matches["medium_confidence"].append(match)
            else:
                grouped_matches["low_confidence"].append(match)

        # Determine distribution type
        if len(grouped_matches["high_confidence"]) == 1:
            distribution = "single_leading_diagnosis"
        elif len(grouped_matches["high_confidence"]) > 1:
            distribution = "multiple_competing_diagnoses"
        elif len(grouped_matches["medium_confidence"]) > 0:
            distribution = "no_clear_diagnosis"
        else:
            distribution = "no_good_matches"

        return {
            "distribution": distribution,
            "score_range": score_range,
            "max_score": max_score,
            "min_score": min_score,
            "grouped_matches": grouped_matches
        }

    def analyze_clinical_notes(self, file_path, output_path=None):
        """
        Process clinical notes through Phases 1-3 to determine diagnostic confidence.

        Args:
            file_path (str): Path to clinical notes file
            output_path (str, optional): Path to save results as JSON

        Returns:
            dict: Complete diagnostic assessment
        """
        try:
            self.logger.info(f"Processing clinical notes from {file_path}")

            # Run Phase 2 which includes Phase 1 internally
            phase2_results = self.phenotype_matcher.match_diseases(file_path)

            if "error" in phase2_results:
                self.logger.error(
                    f"Error in Phase 2: {phase2_results['error']}")
                return {
                    "error": phase2_results["error"],
                    "phase": "phase2",
                    "confidence_level": "error",
                    "validated_phenotypes": [],
                    "disease_matches": []
                }

            validated_phenotypes = phase2_results["validated_phenotypes"]
            disease_matches = phase2_results["disease_matches"]

            # Count present phenotypes for threshold determination
            present_phenotype_count = self._count_present_phenotypes(
                validated_phenotypes)

            # Analyze distribution of match scores
            distribution_analysis = self._analyze_results_distribution(
                disease_matches)

            # Apply dynamic thresholding based on phenotype count
            diagnosis_assessment = self._analyze_top_matches(
                disease_matches, present_phenotype_count)

            # Compile comprehensive results
            results = {
                "validated_phenotypes": validated_phenotypes,
                "disease_matches": disease_matches,
                "diagnosis_assessment": diagnosis_assessment,
                "distribution_analysis": distribution_analysis
            }

            # Save results to JSON if output path provided
            if output_path:
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                with open(output_path, 'w') as f:
                    json.dump(results, f, indent=2)
                self.logger.info(f"Results saved to {output_path}")

            return results

        except Exception as e:
            self.logger.error(f"Diagnosis analysis failed: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return {
                "error": str(e),
                "phase": "phase3",
                "confidence_level": "error",
                "validated_phenotypes": [],
                "disease_matches": []
            }

    def analyze_matches(self, phenotypes, disease_matches):
        """
        Analyze match results directly from phenotypes and disease matches.
        Useful when Phase 1 and 2 have already been run externally.

        Args:
            phenotypes (list): Validated phenotypes from Phase 1
            disease_matches (list): Disease matches from Phase 2

        Returns:
            dict: Diagnostic assessment
        """
        try:
            # Count present phenotypes
            present_phenotype_count = self._count_present_phenotypes(
                phenotypes)

            # Analyze distribution of match scores
            distribution_analysis = self._analyze_results_distribution(
                disease_matches)

            # Apply dynamic thresholding
            diagnosis_assessment = self._analyze_top_matches(
                disease_matches, present_phenotype_count)

            # Compile results
            return {
                "diagnosis_assessment": diagnosis_assessment,
                "distribution_analysis": distribution_analysis
            }

        except Exception as e:
            self.logger.error(f"Direct analysis failed: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return {
                "error": str(e),
                "confidence_level": "error"
            }


# Example usage
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Create data directory if it doesn't exist
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)

    # Initialize the confidence analyzer
    analyzer = DiagnosisConfidenceAnalyzer()

    # Process a sample clinical note
    file_path = "clinical_notes.txt"
    output_path = "results/diagnosis_results.json"

    if not os.path.exists(file_path):
        print(f"Error: File {file_path} not found.")
    else:
        print(f"Processing clinical notes from {file_path}")
        results = analyzer.analyze_clinical_notes(file_path, output_path)

        if "error" in results:
            print(f"Error: {results['error']}")
        else:
            # Display results
            assessment = results["diagnosis_assessment"]
            print("\n===== DIAGNOSTIC ASSESSMENT =====")
            print(
                f"Confidence Level: {assessment['confidence_level'].upper()}")
            print(f"Explanation: {assessment['explanation']}")
            print(f"Recommendation: {assessment['recommendation']}")

            print("\n===== TOP DISEASE MATCH =====")
            top_match = assessment["top_match"]
            print(
                f"Disease: {top_match['disease_name']} ({top_match['disease_id']})")
            print(f"Match Score: {top_match['match_score']:.4f}")
            print(f"Matched Phenotypes: {top_match['matched_phenotypes']}")

            if assessment.get("differential_diagnoses") and len(assessment["differential_diagnoses"]) > 1:
                print("\n===== DIFFERENTIAL DIAGNOSES =====")
                print(assessment["differential_info"])
                for i, match in enumerate(assessment["differential_diagnoses"][1:], start=2):
                    print(
                        f"{i}. {match['disease_name']} - Score: {match['match_score']:.4f}")
