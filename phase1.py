"""
Phase 1: Phenotype Ingestion & Validation

Purpose: Extract and validate phenotypes from clinical notes or direct user input.

Author: Clinical NLP Specialist
Date: March 1, 2025
"""

import logging
import Levenshtein as lev
import pronto
import PyPDF2
import docx
import os
import re
from clinphen_src.get_phenotypes import extract_phenotypes


class PhenotypeExtractor:
    def __init__(self,
                 hpo_path="data/hp.obo",
                 hpo_synonyms_file="data/hpo_synonyms.txt"):
        """
        Initialize phenotype extraction components including HPO ontology.
        """
        self.logger = self._setup_logger()

        # Standard initialization
        self.hpo = self._load_hpo_ontology(hpo_path)
        self.hpo_terms = self._preprocess_hpo_terms()
        self.synonym_to_term_id = self._build_synonym_map()
        self.hpo_synonyms_file = hpo_synonyms_file

        # Enhanced negation patterns for better accuracy
        self.negation_patterns = [
            "no", "not", "without", "absence of", "absent", "negative",
            "denies", "denied", "negative for", "neg for", "unremarkable",
            "no evidence of", "no history of", "no sign of", "non-contributory",
            "noncontributory", "never had", "rules out", "ruled out"
        ]

        # Affirming patterns that override negation detection
        self.affirming_patterns = [
            "presents with", "presenting with", "reports", "reporting",
            "has", "have", "had", "experiencing", "experiences", "experienced",
            "complains of", "complaining of", "complained of", "positive for",
            "consistent with", "compatible with", "diagnostic of", "confirmed"
        ]

        # Disease categories that should be treated with caution (high false positive risk)
        self.high_risk_term_categories = [
            "birth", "history", "pain", "frailty", "paresthesia",
            "unilateral", "wide", "behavioral", "psychiatric"
        ]

        # Uncertainty markers for phrases like "X or Y"
        self.uncertainty_markers = [
            " or ", " versus ", " vs ", " possible ", " probable ",
            " suspected ", " likely ", " maybe ", " perhaps "
        ]

        # Fuzzy match threshold
        self.fuzzy_match_threshold = 0.80

    def _setup_logger(self):
        """Set up logging system for the pipeline."""
        logger = logging.getLogger("phenotype_extractor")
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        return logger

    def _load_hpo_ontology(self, hpo_path):
        """Load HPO ontology using pronto from local file with explicit encoding."""
        self.logger.info("Loading HPO ontology...")
        try:
            # Explicitly specify encoding as UTF-8 instead of relying on auto-detection
            return pronto.Ontology(hpo_path)
        except Exception as e:
            # If UTF-8 fails, fall back to ISO-8859-1 as suggested by the warning
            self.logger.warning(
                f"Failed to load with UTF-8 encoding: {str(e)}")
            try:
                return pronto.Ontology(hpo_path)
            except Exception as e2:
                self.logger.error(
                    f"HPO loading failed with all encodings: {str(e2)}")
                raise

    def _build_synonym_map(self):
        """
        Build a comprehensive mapping from synonyms to term IDs.
        This enables synonym lookup during phenotype validation.
        """
        self.logger.info("Building HPO synonym map...")
        synonym_to_term_id = {}

        # Modified to work with newer pronto API
        for term in self.hpo.terms():
            # Add primary name
            name = term.name.lower()
            synonym_to_term_id[name] = term.id

            # Add all synonyms
            if hasattr(term, 'synonyms'):
                for synonym in term.synonyms:
                    syn_text = str(synonym).lower()
                    synonym_to_term_id[syn_text] = term.id

        self.logger.info(
            f"Built synonym map with {len(synonym_to_term_id)} entries")
        return synonym_to_term_id

    def _preprocess_hpo_terms(self):
        """Extract and normalize HPO terms for efficient lookup."""
        return {term.id: term.name for term in self.hpo.terms()}

    def _extract_text(self, file_path):
        """Handle multiple file formats (PDF, DOCX, TXT) for clinical notes."""
        self.logger.info(f"Processing file: {file_path}")
        text = ""

        if file_path.endswith('.pdf'):
            with open(file_path, 'rb') as f:
                reader = PyPDF2.PdfReader(f)
                text = " ".join([page.extract_text() for page in reader.pages])
        elif file_path.endswith('.docx'):
            doc = docx.Document(file_path)
            text = " ".join([para.text for para in doc.paragraphs])
        elif file_path.endswith('.txt'):
            with open(file_path, 'r') as f:
                text = f.read()

        return text.strip()

    def _detect_negation_improved(self, sentence, term):
        """
        Enhanced negation detection for phenotype terms with contextual understanding.
        Args:
            sentence (str): The sentence or context containing the term
            term (str): The phenotype term to check for negation
        Returns:
            bool: True if term is negated, False otherwise
        """
        # Convert to lowercase for case-insensitive matching
        sentence = sentence.lower()
        term = term.lower()

        # Special case for "unremarkable" and similar phrases that indicate absence
        if "unremarkable" in sentence:
            sections = sentence.split("unremarkable")
            # If term is mentioned before "unremarkable", it's likely negated
            if term in sections[0]:
                return True

        # Special case for family history being unremarkable
        if "family history" in sentence and "unremarkable" in sentence:
            return True

        # Check for direct negation patterns
        for pattern in self.negation_patterns:
            pattern_pos = sentence.find(pattern)
            term_pos = sentence.find(term)

            # If negation appears before the term and within reasonable proximity
            if pattern_pos != -1 and term_pos != -1 and pattern_pos < term_pos:
                # Check for distance - within 80 characters is reasonable for most clinical phrases
                if term_pos - (pattern_pos + len(pattern)) < 80:
                    # Check that there's no affirming pattern between the negation and the term
                    text_between = sentence[pattern_pos +
                                            len(pattern):term_pos]
                    affirmed = False
                    for affirm_pattern in self.affirming_patterns:
                        if affirm_pattern in text_between:
                            affirmed = True
                            break

                    if not affirmed:
                        return True  # Term is negated

        # Check for phrases that indicate something was ruled out
        rule_out_phrases = ["ruled out", "rule out",
                            "unlikely", "improbable", "excluded"]
        for phrase in rule_out_phrases:
            if phrase in sentence and abs(sentence.find(phrase) - sentence.find(term)) < 50:
                return True

        return False  # Term is not negated

    def _handle_uncertainty(self, context, term_name):
        """
        Handle uncertainty expressions like 'X or Y' in clinical notes.
        Args:
            context (str): The surrounding text context
            term_name (str): The phenotype term
        Returns:
            tuple: (is_uncertain, confidence_score)
        """
        context_lower = context.lower()
        term_lower = term_name.lower()

        # Find position of the term in context
        term_pos = context_lower.find(term_lower)
        if term_pos == -1:
            return False, 0.95  # Not found, return default confidence

        # Look for uncertainty markers near the term
        for marker in self.uncertainty_markers:
            if marker in context_lower:
                # Check if marker is reasonably close to the term (within 30 chars)
                marker_pos = context_lower.find(marker)
                if abs(marker_pos - term_pos) < 30:
                    # Return lower confidence for uncertain terms
                    return True, 0.7

        return False, 0.95  # No uncertainty detected, return default confidence

    def _extract_context_improved(self, text, term, window_size=200):
        """
        Enhanced context extraction around a term for improved negation detection.
        Args:
            text (str): Full text
            term (str): Term to find context for
            window_size (int): Character window size
        Returns:
            str: Context around the term
        """
        term_pos = text.lower().find(term.lower())
        if term_pos == -1:
            return text  # Term not found, return full text

        # Try to extract a sentence or paragraph containing the term
        text_lower = text.lower()

        # Look for sentence boundaries (., !, ?, new line)
        sent_start = max(
            text_lower.rfind('.', 0, term_pos),
            text_lower.rfind('?', 0, term_pos),
            text_lower.rfind('!', 0, term_pos),
            text_lower.rfind('\n', 0, term_pos)
        )

        if sent_start == -1:
            sent_start = max(0, term_pos - window_size)
        else:
            sent_start += 1  # Move past the punctuation

        # Find the end of the sentence
        sent_end = min(
            text_lower.find('.', term_pos) if text_lower.find(
                '.', term_pos) != -1 else len(text),
            text_lower.find('?', term_pos) if text_lower.find(
                '?', term_pos) != -1 else len(text),
            text_lower.find('!', term_pos) if text_lower.find(
                '!', term_pos) != -1 else len(text),
            text_lower.find('\n', term_pos) if text_lower.find(
                '\n', term_pos) != -1 else len(text)
        )

        if sent_end == -1:
            sent_end = min(len(text), term_pos + window_size)

        # Extract the sentence or context
        return text[sent_start:sent_end].strip()

    def _generate_hpo_synonyms_file(self, output_path=None):
        """
        Generate HPO synonyms file from the loaded HPO ontology.
        Args:
            output_path (str): Where to save the file, defaults to self.hpo_synonyms_file
        Returns:
            str: Path to the generated file
        """
        if output_path is None:
            output_path = self.hpo_synonyms_file

        self.logger.info(f"Generating HPO synonyms file at {output_path}")

        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        try:
            with open(output_path, 'w') as f:
                for term_id, term_name in self.hpo_terms.items():
                    # Write the primary name
                    f.write(f"{term_id}\t{term_name}\n")

                    # Add synonyms if available in the ontology
                    term = self.hpo[term_id]
                    if hasattr(term, 'synonyms'):
                        for synonym in term.synonyms:
                            syn_name = str(synonym)
                            f.write(f"{term_id}\t{syn_name}\n")

            self.logger.info(
                f"Successfully generated HPO synonyms file with {len(self.hpo_terms)} terms")
            return output_path
        except Exception as e:
            self.logger.error(
                f"Failed to generate HPO synonyms file: {str(e)}")
            return None

    def _validate_term_context(self, term_id, term_name, context, clinical_text):
        """
        Perform contextual validation to reduce false positives.
        Args:
            term_id (str): HPO ID of the term
            term_name (str): Name of the term
            context (str): Context where the term was found
            clinical_text (str): Full clinical text
        Returns:
            bool: True if term is valid in this context, False otherwise
        """
        # Check for high-risk terms that are prone to false positives
        term_lower = term_name.lower()

        # Check if this is a high-risk category term
        is_high_risk = any(
            category in term_lower for category in self.high_risk_term_categories)

        if is_high_risk:
            # For high-risk terms, require stronger evidence

            # 1. Term should appear in an affirming context
            has_affirming_context = any(
                pattern in context.lower() for pattern in self.affirming_patterns)

            # 2. For terms like "birth history", check they're not just section headers
            if "history" in term_lower:
                if term_lower in clinical_text.lower() and len(context.strip()) < 30:
                    # Likely just a section header
                    return False

            # 3. For sensory terms like "pain", require specific descriptor
            if term_lower == "pain":
                # Check if there's a specific descriptor for the pain
                pain_descriptors = [
                    "severe", "mild", "moderate", "sharp", "dull", "chronic", "acute"]
                has_descriptor = any(desc in context.lower()
                                     for desc in pain_descriptors)
                if not has_descriptor:
                    return False

            # More stringent requirements for high-risk terms
            if not has_affirming_context:
                return False

        # Exclude family history - if term is in family history context and family history is unremarkable
        if "family history" in context.lower() and ("unremarkable" in context.lower() or "negative" in context.lower()):
            return False

        return True

    def _lookup_term(self, candidate_term):
        """
        Look up a term in the HPO ontology, checking primary names and synonyms.
        Args:
            candidate_term (str): Term to look up
        Returns:
            tuple: (term_id, term_name, exact_match)
        """
        # Check for exact match in synonym map
        term_lower = candidate_term.lower()

        # Standard synonym lookup
        if term_lower in self.synonym_to_term_id:
            term_id = self.synonym_to_term_id[term_lower]
            return term_id, self.hpo_terms.get(term_id, candidate_term), True

        # If no exact match, use Levenshtein for fuzzy matching
        return self._validate_term(candidate_term)

    def _validate_term(self, candidate_term):
        """
        Validate HPO terms using Levenshtein similarity.
        Args:
            candidate_term (str): The phenotype term to validate
        Returns:
            tuple: (matched_term_id, matched_term_name, confidence_score)
        """
        max_similarity = 0
        matched_term_id = None
        matched_term_name = None

        # Regular fuzzy matching with threshold
        for hpo_id, hpo_name in self.hpo_terms.items():
            similarity = lev.ratio(candidate_term.lower(), hpo_name.lower())
            if similarity > max_similarity:
                max_similarity = similarity
                matched_term_id = hpo_id
                matched_term_name = hpo_name

        # Only return matches above threshold
        if max_similarity >= self.fuzzy_match_threshold:
            return matched_term_id, matched_term_name, max_similarity

        return None, None, 0

    def _extract_multiword_phrases(self, text, max_words=4):
        """
        Extract multi-word phrases for term detection.
        Args:
            text (str): Text to extract phrases from
            max_words (int): Maximum words in a phrase
        Returns:
            list: List of phrases found in text
        """
        words = re.findall(r'\b\w+\b', text.lower())
        unique_phrases = set()

        # Extract original phrases
        for phrase_len in range(1, min(max_words + 1, len(words) + 1)):
            for i in range(len(words) - phrase_len + 1):
                phrase = ' '.join(words[i:i+phrase_len])
                if len(phrase) > 3:  # Skip very short phrases
                    unique_phrases.add(phrase)

        return list(unique_phrases)

    def extract_phenotypes(self, file_path, existing_hpo=None):
        """
        Extract and validate phenotypes from clinical notes.

        Args:
            file_path (str): Path to clinical note (PDF/TXT/DOCX)
            existing_hpo (list): Optional list of existing HPO terms

        Returns:
            list: Validated phenotype terms with metadata
        """
        try:
            self.logger.info(f"Processing file: {file_path}")

            # Step 1: Text extraction
            clinical_text = self._extract_text(file_path)
            self.logger.info(
                f"Extracted text from {file_path} (length: {len(clinical_text)})")

            # Check if text is too short for meaningful analysis
            if len(clinical_text) < 10:
                return []

            # Check if HPO synonyms file exists, generate if not
            if not os.path.exists(self.hpo_synonyms_file):
                self.logger.warning(
                    f"HPO synonyms file not found at {self.hpo_synonyms_file}")
                self.hpo_synonyms_file = self._generate_hpo_synonyms_file()
                if not self.hpo_synonyms_file:
                    raise ValueError("Could not generate HPO synonyms file")

            # Track which term IDs we've already added to prevent duplicates
            processed_term_ids = set()
            candidate_terms = []

            # Step 2: Direct term extraction with enhanced phrase detection
            self.logger.info("Performing enhanced term extraction...")
            phrases = self._extract_multiword_phrases(clinical_text)

            for phrase in phrases:
                # Try to look up the term
                term_id, term_name, is_exact = self._lookup_term(phrase)

                # Skip if no term found or already processed this term ID
                if not term_id or term_id in processed_term_ids:
                    continue

                # Mark as processed to prevent duplicates
                processed_term_ids.add(term_id)

                # Extract broader context (increased window size)
                context = self._extract_context_improved(clinical_text, phrase)

                # Check if this is a valid phenotype mention (not a section header, etc.)
                if self._validate_term_context(term_id, term_name, context, clinical_text):
                    # Detect negation using improved algorithm
                    is_negated = self._detect_negation_improved(
                        context, phrase)

                    # Check for uncertainty (e.g., "asthma or fever")
                    is_uncertain, confidence = self._handle_uncertainty(
                        context, term_name)

                    candidate_terms.append({
                        'term_id': term_id,
                        'term_name': term_name,
                        'confidence': confidence,
                        'uncertain': is_uncertain,
                        'present': not is_negated,
                        'context': context
                    })

                    status = "present" if not is_negated else "absent"
                    uncertain_flag = " (uncertain)" if is_uncertain else ""
                    self.logger.info(
                        f"Direct match: {term_name} ({term_id}) - {status}{uncertain_flag}")

            # Step 3: ClinPhen phenotype extraction with improved validation
            self.logger.info(
                "Using ClinPhen for phenotype extraction with enhanced validation...")
            names_map = self.hpo_terms

            # Load HPO synonyms and extract phenotypes
            phenotype_raw_results = extract_phenotypes(
                clinical_text, names_map, self.hpo_synonyms_file)
            self.logger.info("Extracted phenotypes from clinical text")

            # Parse the results string into structured data
            phenotype_lines = phenotype_raw_results.strip().split('\n')
            if len(phenotype_lines) > 0:
                header = phenotype_lines[0]  # Skip header
                for line in phenotype_lines[1:]:
                    parts = line.split('\t')
                    if len(parts) >= 5:  # HPO ID, name, occurrences, earliness, example
                        hpo_id = parts[0]
                        term_name = parts[1]
                        occurrences = int(parts[2])
                        example_sentence = parts[4]

                        # Skip if already found through direct extraction
                        if hpo_id in processed_term_ids:
                            continue

                        # Mark as processed to prevent duplicates
                        processed_term_ids.add(hpo_id)

                        # Extract broader context for the term
                        context = self._extract_context_improved(
                            clinical_text, term_name)

                        # Validate term context to reduce false positives
                        if self._validate_term_context(hpo_id, term_name, context, clinical_text):
                            # Detect negation using improved algorithm
                            is_negated = self._detect_negation_improved(
                                context, term_name)

                            # Check for uncertainty
                            is_uncertain, confidence = self._handle_uncertainty(
                                context, term_name)

                            # Add validation for high-frequency, low-information terms
                            high_risk_terms = [
                                "pain", "birth", "history", "unilateral", "bilateral"]
                            if any(risk_term in term_name.lower() for risk_term in high_risk_terms):
                                # For high-risk terms, we need additional evidence
                                if occurrences < 2 and not any(pattern in context.lower() for pattern in self.affirming_patterns):
                                    # Skip this term, not enough evidence
                                    self.logger.info(
                                        f"Skipping high-risk term with insufficient context: {term_name} ({hpo_id})")
                                    continue

                            candidate_terms.append({
                                'term_id': hpo_id,
                                'term_name': term_name,
                                'confidence': confidence,
                                'uncertain': is_uncertain,
                                'present': not is_negated,
                                'occurrences': occurrences,
                                'example': example_sentence,
                                'context': context
                            })

                            status = "present" if not is_negated else "absent"
                            uncertain_flag = " (uncertain)" if is_uncertain else ""
                            self.logger.info(
                                f"Validated term: {term_name} ({hpo_id}) - {status}{uncertain_flag}")

            # Merge with existing HPO terms if provided
            if existing_hpo:
                for term in existing_hpo:
                    if isinstance(term, str):  # If just term ID is provided
                        if term in processed_term_ids:
                            continue

                        processed_term_ids.add(term)
                        term_name = self.hpo_terms.get(term, term)

                        candidate_terms.append({
                            'term_id': term,
                            'term_name': term_name,
                            'confidence': 1.0,
                            'present': True
                        })

                        self.logger.info(
                            f"Added existing term: {term_name} ({term})")
                    elif isinstance(term, dict):  # If term with metadata is provided
                        if term.get('term_id') in processed_term_ids:
                            continue

                        processed_term_ids.add(term.get('term_id'))
                        candidate_terms.append(term)

                        self.logger.info(
                            f"Added existing term with metadata: {term}")

            # Final validation and filtering
            validated_terms = []
            for term in candidate_terms:
                # Skip terms that are in family history context
                if "family history" in term.get('context', '').lower() and "unremarkable" in term.get('context', '').lower():
                    continue

                # For birth history and similar categories that often represent section headers
                if term['term_name'].lower() == "birth history" or term['term_name'].lower() == "family history":
                    # Only include if there's specific affirming evidence
                    if 'context' in term and len(term['context']) > 50 and any(pattern in term['context'].lower() for pattern in self.affirming_patterns):
                        validated_terms.append(term)
                else:
                    validated_terms.append(term)

            self.logger.info(
                f"Extracted {len(validated_terms)} validated phenotypes")

            return validated_terms

        except Exception as e:
            self.logger.error(f"Phenotype extraction failed: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return []


# Main module execution
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # Initialize the extractor
    extractor = PhenotypeExtractor()

    # Process a sample clinical note
    file_path = "clinical_notes.txt"
    if not os.path.exists(file_path):
        print(f"Error: File {file_path} not found.")
    else:
        print(f"Processing clinical notes from {file_path}")
        validated_phenotypes = extractor.extract_phenotypes(file_path)

        print("\n===== VALIDATED PHENOTYPES =====")
        for term in validated_phenotypes:
            status = "Present" if term["present"] else "Absent"
            uncertain = " (uncertain)" if term.get("uncertain", False) else ""
            print(
                f"- {term['term_name']} ({term['term_id']}): {status}{uncertain} (confidence: {term['confidence']})")
