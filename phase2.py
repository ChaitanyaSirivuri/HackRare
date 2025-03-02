"""
Phase 2: Initial Phenotype Matching

Purpose: Match patient phenotypes to diseases using a similarity algorithm.

Author: Clinical Genomics Specialist
Date: March 1, 2025
"""

import os
import logging
import numpy as np
from pathlib import Path
from phase1 import PhenotypeExtractor


class PhenotypeDiseaseMatcher:
    def __init__(self,
                 hpo_gene_disease_path="data/phenotype_to_genes.txt",
                 hpo_path="data/hp.obo",
                 hpo_synonyms_file="data/hpo_synonyms.txt"):
        """
        Initialize phenotype-disease matching components.

        Args:
            hpo_gene_disease_path (str): Path to phenotype-gene-disease mapping file
            hpo_path (str): Path to HPO ontology file
            hpo_synonyms_file (str): Path to HPO synonyms file
        """
        self.logger = self._setup_logger()

        # Load HPO-gene-disease mappings
        self.hpo_gene_disease_path = hpo_gene_disease_path
        mappings = self._load_hpo_gene_disease_data(hpo_gene_disease_path)
        self.hpo_to_disease = mappings['hpo_to_disease']
        self.disease_to_hpo = mappings['disease_to_hpo']
        self.gene_to_disease = mappings['gene_to_disease']
        self.disease_to_gene = mappings['disease_to_gene']
        self.disease_names = mappings['disease_names']

        # Initialize the phenotype extractor from Phase 1
        self.phenotype_extractor = PhenotypeExtractor(
            hpo_path=hpo_path,
            hpo_synonyms_file=hpo_synonyms_file
        )

        # Track term information content for IC-based matching
        self.term_information_content = self._calculate_term_ic()

    def _setup_logger(self):
        """Set up logging system for the pipeline."""
        logger = logging.getLogger("phenotype_matcher")
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        return logger

    def _load_hpo_gene_disease_data(self, mapping_path):
        """
        Load mapping between HPO terms, genes, and diseases from the tabular format.

        Args:
            mapping_path (str): Path to HPO-gene-disease mapping file

        Returns:
            dict: Multiple mappings between HPO, genes, and diseases
        """
        self.logger.info("Loading phenotype to genes direct mapping...")

        hpo_to_disease = {}  # HPO ID -> list of associated disease IDs
        disease_to_hpo = {}  # Disease ID -> list of associated HPO IDs
        gene_to_disease = {}  # Gene symbol -> list of associated disease IDs
        disease_to_gene = {}  # Disease ID -> list of associated gene symbols
        disease_names = {}    # Disease ID -> disease name

        try:
            with open(mapping_path, 'r') as f:
                # Check and skip header if present
                header = f.readline()
                if "hpo_id" in header.lower():
                    # File has a header, continue
                    pass
                else:
                    # No header, reset to the beginning of the file
                    f.seek(0)

                for line in f:
                    parts = line.strip().split('\t')
                    if len(parts) < 5:
                        continue

                    hpo_id = parts[0]
                    hpo_name = parts[1]
                    gene_id = parts[2]
                    gene_symbol = parts[3]
                    disease_id = parts[4]

                    # Extract disease name if available (from OMIM ID format)
                    disease_name = disease_id
                    if len(parts) > 5:
                        disease_name = parts[5]
                    elif "OMIM:" in disease_id:
                        # Try to extract a readable name
                        disease_names[disease_id] = f"OMIM:{disease_id.split(':')[1]}"

                    # Build HPO to disease mapping
                    if hpo_id not in hpo_to_disease:
                        hpo_to_disease[hpo_id] = set()
                    hpo_to_disease[hpo_id].add(disease_id)

                    # Build disease to HPO mapping
                    if disease_id not in disease_to_hpo:
                        disease_to_hpo[disease_id] = set()
                    disease_to_hpo[disease_id].add(hpo_id)

                    # Build gene to disease mapping
                    if gene_symbol not in gene_to_disease:
                        gene_to_disease[gene_symbol] = set()
                    gene_to_disease[gene_symbol].add(disease_id)

                    # Build disease to gene mapping
                    if disease_id not in disease_to_gene:
                        disease_to_gene[disease_id] = set()
                    disease_to_gene[disease_id].add(gene_symbol)

                    # Store disease name
                    if disease_id not in disease_names:
                        disease_names[disease_id] = disease_name

            # Convert sets to lists for better handling
            for hpo_id in hpo_to_disease:
                hpo_to_disease[hpo_id] = list(hpo_to_disease[hpo_id])

            for disease_id in disease_to_hpo:
                disease_to_hpo[disease_id] = list(disease_to_hpo[disease_id])

            for gene_symbol in gene_to_disease:
                gene_to_disease[gene_symbol] = list(
                    gene_to_disease[gene_symbol])

            for disease_id in disease_to_gene:
                disease_to_gene[disease_id] = list(disease_to_gene[disease_id])

            hpo_count = len(hpo_to_disease)
            disease_count = len(disease_to_hpo)
            gene_count = len(gene_to_disease)

            self.logger.info(
                f"Loaded mappings: {hpo_count} HPO terms, {disease_count} diseases, {gene_count} genes")

            return {
                'hpo_to_disease': hpo_to_disease,
                'disease_to_hpo': disease_to_hpo,
                'gene_to_disease': gene_to_disease,
                'disease_to_gene': disease_to_gene,
                'disease_names': disease_names
            }

        except Exception as e:
            self.logger.error(
                f"Failed to load HPO-gene-disease mapping: {str(e)}")
            return {
                'hpo_to_disease': {},
                'disease_to_hpo': {},
                'gene_to_disease': {},
                'disease_to_gene': {},
                'disease_names': {}
            }

    def _calculate_term_ic(self):
        """
        Calculate information content (IC) for each HPO term based on disease annotation frequency.

        Returns:
            dict: Mapping of HPO term IDs to their information content values
        """
        term_frequency = {}
        total_disease_count = len(self.disease_to_hpo)

        # Count how many diseases each term appears in
        for disease_id, term_list in self.disease_to_hpo.items():
            for term_id in term_list:
                if term_id not in term_frequency:
                    term_frequency[term_id] = 0
                term_frequency[term_id] += 1

        # Calculate IC values: -log(frequency/total)
        term_ic = {}
        for term_id, freq in term_frequency.items():
            if total_disease_count > 0:
                # Avoid division by zero and log(0)
                probability = max(1e-10, freq / total_disease_count)
                term_ic[term_id] = -np.log(probability)
            else:
                term_ic[term_id] = 0

        self.logger.info(f"Calculated IC values for {len(term_ic)} HPO terms")
        return term_ic

    def _calculate_phrank_similarity(self, patient_terms, disease_terms):
        """
        Calculate phenotype similarity using Phrank-inspired algorithm.

        Args:
            patient_terms (list): List of patient HPO term IDs
            disease_terms (list): List of disease HPO term IDs

        Returns:
            float: Similarity score between 0 and 1
        """
        if not patient_terms or not disease_terms:
            return 0.0

        # Convert to sets for intersection/union operations
        patient_set = set(patient_terms)
        disease_set = set(disease_terms)

        # Find shared terms
        shared_terms = patient_set.intersection(disease_set)

        # Basic Phrank-inspired scoring with IC weighting
        shared_ic_sum = sum(self.term_information_content.get(
            term, 1.0) for term in shared_terms)
        patient_ic_sum = sum(self.term_information_content.get(
            term, 1.0) for term in patient_set)

        # Avoid division by zero
        if patient_ic_sum == 0:
            return 0.0

        # Normalize to 0-1 range
        similarity_score = shared_ic_sum / patient_ic_sum

        return min(1.0, similarity_score)

    def _match_phenotypes_to_diseases(self, phenotypes):
        """
        Match phenotypes to diseases with improved scoring algorithm.

        Args:
            phenotypes (list): List of extracted phenotypes with term_id and present status

        Returns:
            list: Ranked list of disease matches with scores
        """
        self.logger.info(
            "Matching phenotypes to diseases using direct mapping...")

        # Extract present phenotypes with confidence weighting
        present_terms = []
        for p in phenotypes:
            if p.get('present', True):
                # Add weighted confidence - terms with higher confidence get more weight
                present_terms.append({
                    'term_id': p['term_id'],
                    'weight': p.get('confidence', 0.95)
                })

        # Count matches for each disease with weighted scoring
        disease_matches = {}
        for term in present_terms:
            term_id = term['term_id']
            term_weight = term['weight']

            if term_id in self.hpo_to_disease:
                diseases = self.hpo_to_disease[term_id]
                for disease_id in diseases:
                    if disease_id not in disease_matches:
                        disease_matches[disease_id] = {
                            'matched_terms': [],
                            'score': 0.0
                        }

                    # Only count each term once per disease
                    if term_id not in disease_matches[disease_id]['matched_terms']:
                        disease_matches[disease_id]['matched_terms'].append(
                            term_id)
                        disease_matches[disease_id]['score'] += term_weight

        # Calculate normalized scores with phenotypic relevance
        disease_scores = []
        for disease_id, match_data in disease_matches.items():
            # Get expected phenotypes for this disease
            total_phenotypes = len(self.disease_to_hpo.get(disease_id, []))
            if total_phenotypes == 0:
                continue

            # Get associated genes
            associated_genes = self.disease_to_gene.get(disease_id, [])

            # Get all phenotype terms for this disease
            disease_phenotypes = self.disease_to_hpo.get(disease_id, [])

            # Get patient phenotype term IDs
            patient_phenotypes = [term['term_id'] for term in present_terms]

            # Calculate semantic similarity using Phrank-inspired algorithm
            phrank_score = self._calculate_phrank_similarity(
                patient_phenotypes, disease_phenotypes)

            # Calculate adjusted match score: (matched_count^2) / total_phenotypes
            # This quadratically rewards more matches while considering total phenotypes
            matched_count = len(match_data['matched_terms'])
            if matched_count == 0:
                continue

            # Basic score based on match count and phenotype completeness
            basic_score = (matched_count ** 2) / total_phenotypes

            # Combine scores (give more weight to Phrank score)
            combined_score = 0.7 * phrank_score + 0.3 * basic_score

            # Get disease name
            disease_name = self.disease_names.get(disease_id, disease_id)

            disease_scores.append({
                'disease_id': disease_id,
                'disease_name': disease_name,
                'match_score': combined_score,
                'phrank_score': phrank_score,
                'basic_score': basic_score,
                'matched_phenotypes': f"{matched_count}/{total_phenotypes}",
                'associated_genes': associated_genes
            })

        # Sort by match score in descending order
        ranked_diseases = sorted(
            disease_scores, key=lambda x: x['match_score'], reverse=True)

        self.logger.info(f"Found {len(ranked_diseases)} matching diseases")
        return ranked_diseases

    def match_diseases(self, file_path):
        """
        Process clinical notes to extract phenotypes and match to diseases.

        Args:
            file_path (str): Path to clinical notes file

        Returns:
            dict: Dictionary containing validated phenotypes and disease matches
        """
        try:
            self.logger.info(f"Processing clinical notes from {file_path}")

            # PHASE 1: Extract and validate phenotypes
            validated_phenotypes = self.phenotype_extractor.extract_phenotypes(
                file_path)

            if not validated_phenotypes:
                self.logger.warning(
                    "No valid phenotypes extracted from clinical notes")
                return {
                    "validated_phenotypes": [],
                    "disease_matches": []
                }

            self.logger.info(
                f"Extracted {len(validated_phenotypes)} validated phenotypes")

            # PHASE 2: Match phenotypes to diseases
            disease_matches = self._match_phenotypes_to_diseases(
                validated_phenotypes)

            return {
                "validated_phenotypes": validated_phenotypes,
                "disease_matches": disease_matches
            }

        except Exception as e:
            self.logger.error(f"Disease matching failed: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return {
                "error": str(e),
                "validated_phenotypes": [],
                "disease_matches": []
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

    # Initialize the disease matcher
    matcher = PhenotypeDiseaseMatcher()

    # Process a sample clinical note
    file_path = "clinical_notes.txt"

    if not os.path.exists(file_path):
        print(f"Error: File {file_path} not found.")
    else:
        print(f"Processing clinical notes from {file_path}")
        results = matcher.match_diseases(file_path)

        if "error" in results:
            print(f"Error: {results['error']}")
        else:
            print("\n===== VALIDATED PHENOTYPES =====")
            for term in results["validated_phenotypes"]:
                status = "Present" if term["present"] else "Absent"
                uncertain = " (uncertain)" if term.get(
                    "uncertain", False) else ""
                print(
                    f"- {term['term_name']} ({term['term_id']}): {status}{uncertain} (confidence: {term['confidence']})")

            print("\n===== TOP DISEASE MATCHES =====")
            for i, disease in enumerate(results["disease_matches"][:5]):
                print(
                    f"{i+1}. {disease['disease_name']} - Score: {disease['match_score']:.4f}")
                print(
                    f"   Matched Phenotypes: {disease['matched_phenotypes']}")
                print(
                    f"   Associated Genes: {', '.join(disease['associated_genes'])}")
