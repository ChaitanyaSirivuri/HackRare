"""
Phase 4: Iterative Phenotype Refinement

Purpose: Suggest additional phenotypes to refine diagnosis if confidence is low.
Ask interactive questions and update disease matching based on responses.

Author: Clinical Decision Support Specialist
Date: March 1, 2025
"""

import logging
import os
import numpy as np
import re
from phase3 import DiagnosisConfidenceAnalyzer
import pandas as pd


class PhenotypeRefinementEngine:
    def __init__(self,
                 hpo_path="data/hp.obo",
                 hpo_synonyms_file="data/hpo_synonyms.txt",
                 hpo_gene_disease_path="data/phenotype_to_genes.txt",
                 mim_titles_path="data/mimTitles.txt"):
        """
        Initialize the phenotype refinement engine.

        Args:
            hpo_path (str): Path to HPO ontology file
            hpo_synonyms_file (str): Path to HPO synonyms file
            hpo_gene_disease_path (str): Path to phenotype-gene-disease mapping file
            mim_titles_path (str): Path to MIM titles file
        """
        self.logger = self._setup_logger()
        np.random.seed(42)
        # Initialize Phase 3 components
        self.diagnosis_analyzer = DiagnosisConfidenceAnalyzer(
            hpo_path=hpo_path,
            hpo_synonyms_file=hpo_synonyms_file,
            hpo_gene_disease_path=hpo_gene_disease_path
        )

        # Load MIM titles if file exists
        self.mim_titles = self._load_mim_titles(mim_titles_path)

        # Maximum number of questions to ask per refinement iteration
        self.max_questions = 5

        # Tracking previous answers to avoid asking the same question twice
        self.asked_phenotypes = set()

        # Mapping for human-friendly question formats
        self.question_templates = [
            "Do you experience {}?",
            "Have you noticed {}?",
            "Has there been any {}?",
            "Do you have {}?",
            "Have you been diagnosed with {}?"
        ]

    def _setup_logger(self):
        """Set up logging system for refinement engine."""
        logger = logging.getLogger("refinement_engine")
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        return logger

    def _load_mim_titles(self, mim_titles_path):
        """
        Load MIM titles from the mimTitles.csv file.

        Args:
            mim_titles_path (str): Path to the mimTitles file 

        Returns:
            dict: Mapping of MIM numbers to disease names
        """
        mim_titles = {}

        try:
            # Modify path to use CSV instead of txt
            csv_path = mim_titles_path

            if not os.path.exists(csv_path):
                self.logger.warning(
                    f"MIM titles CSV file not found at {csv_path}")
                return {}

            self.logger.info(f"Loading MIM titles from {csv_path}")

            # Load the CSV file using pandas
            df = pd.read_csv(csv_path)

            if 'MIM Number' in df.columns and 'Preferred Title' in df.columns:
                df['Main Title'] = df.apply(
                    lambda row: f"{row['Preferred Title']} ({row['symbol']})" if 'symbol' in df.columns and pd.notna(
                        row['symbol']) and row['symbol'] else row['Preferred Title'],
                    axis=1
                )
                mim_titles = dict(
                    zip(df['MIM Number'].astype(str), df['Main Title']))

            self.logger.info(f"Loaded {len(mim_titles)} MIM titles")
            return mim_titles

        except Exception as e:
            self.logger.error(f"Error loading MIM titles: {str(e)}")
            return {}

    def _get_disease_name(self, disease_id):
        """
        Get the disease name from the disease ID.

        Args:
            disease_id (str): Disease ID (e.g., "OMIM:251280")

        Returns:
            str: Disease name or original ID if not found
        """
        if not disease_id.startswith("OMIM:"):
            return disease_id

        mim_number = disease_id.split(":")[1]

        if mim_number in self.mim_titles:
            return self.mim_titles[mim_number]
        else:
            return disease_id

    def _identify_discriminative_phenotypes(self, top_diseases, current_phenotypes):
        """
        Identify phenotypes that can discriminate between top diseases.

        Args:
            top_diseases (list): Top disease matches
            current_phenotypes (list): Currently validated phenotypes

        Returns:
            list: Discriminative phenotypes with metadata
        """
        self.logger.info("Identifying discriminative phenotypes...")

        # Extract current phenotype IDs for quick lookup
        current_phenotype_ids = {p['term_id'] for p in current_phenotypes}

        # Track disease-specific phenotypes
        disease_specific_phenotypes = {}
        all_phenotypes = set()

        # For each top disease, get associated phenotypes not in current set
        for disease in top_diseases:
            disease_id = disease['disease_id']
            disease_name = disease['disease_name']

            # Get all phenotypes for this disease
            if disease_id in self.diagnosis_analyzer.phenotype_matcher.disease_to_hpo:
                disease_phenotypes = self.diagnosis_analyzer.phenotype_matcher.disease_to_hpo[
                    disease_id]

                # Filter out phenotypes already reported and previously asked
                novel_phenotypes = [
                    p for p in disease_phenotypes
                    if p not in current_phenotype_ids and
                    p not in self.asked_phenotypes
                ]

                # Store for this disease
                disease_specific_phenotypes[disease_id] = {
                    'disease_name': disease_name,
                    'phenotypes': novel_phenotypes
                }

                # Add to all phenotypes
                all_phenotypes.update(novel_phenotypes)

        # Calculate information content for discriminative power
        phenotype_scores = {}

        for phenotype in all_phenotypes:
            # Skip if we've already asked about this phenotype
            if phenotype in self.asked_phenotypes:
                continue

            # Check how many diseases have this phenotype
            disease_count = sum(
                1 for disease_id in disease_specific_phenotypes
                if phenotype in disease_specific_phenotypes[disease_id]['phenotypes']
            )

            # Calculate a discrimination score
            # Most valuable phenotypes appear in some but not all diseases
            if disease_count == 0:
                continue

            # Appears in only one disease (highly specific)
            if disease_count == 1:
                specificity = 1.0
            else:  # Appears in multiple diseases
                specificity = 1.0 / disease_count

            # Use information content as a factor if available
            ic_value = self.diagnosis_analyzer.phenotype_matcher.term_information_content.get(
                phenotype, 1.0)

            # Final score combines specificity and information content
            phenotype_scores[phenotype] = specificity * ic_value

        # Sort phenotypes by score
        sorted_phenotypes = sorted(
            phenotype_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )

        # Convert to list of dictionaries with metadata
        discriminative_phenotypes = []

        for phenotype_id, score in sorted_phenotypes:
            # Get term name
            term_name = self.diagnosis_analyzer.phenotype_matcher.phenotype_extractor.hpo_terms.get(
                phenotype_id, phenotype_id)

            # Get diseases that have this phenotype
            relevant_diseases = [
                disease_id for disease_id in disease_specific_phenotypes
                if phenotype_id in disease_specific_phenotypes[disease_id]['phenotypes']
            ]

            discriminative_phenotypes.append({
                'term_id': phenotype_id,
                'term_name': term_name,
                'score': score,
                'relevant_diseases': relevant_diseases
            })

        return discriminative_phenotypes

    def _generate_questions(self, discriminative_phenotypes, max_questions=5):
        """
        Generate patient-friendly questions for discriminative phenotypes.

        Args:
            discriminative_phenotypes (list): Discriminative phenotypes with metadata
            max_questions (int): Maximum number of questions to generate

        Returns:
            list: Questions with metadata
        """
        self.logger.info("Generating patient-friendly questions...")

        questions = []

        for phenotype in discriminative_phenotypes[:max_questions]:
            term_id = phenotype['term_id']
            term_name = phenotype['term_name']

            # Make sure we haven't asked this before
            if term_id in self.asked_phenotypes:
                continue

            # Add to tracking set
            self.asked_phenotypes.add(term_id)

            # Format a patient-friendly question
            question_text = np.random.choice(self.question_templates).format(
                self._make_term_patient_friendly(term_name)
            )

            questions.append({
                'question_id': len(questions) + 1,
                'question': question_text,
                'term_id': term_id,
                'term_name': term_name,
                'relevant_diseases': phenotype['relevant_diseases']
            })

            if len(questions) >= max_questions:
                break

        return questions

    def _make_term_patient_friendly(self, term):
        """
        Convert HPO term to patient-friendly language.

        Args:
            term (str): HPO term name

        Returns:
            str: Patient-friendly version of the term
        """
        # Remove HP: prefix if present
        term = re.sub(r'^HP:', '', term)

        # Replace underscores with spaces
        term = term.replace('_', ' ')

        # Make first letter lowercase for natural flow in questions
        if term and len(term) > 0:
            term = term[0].lower() + term[1:]

        # Remove brackets and IDs
        term = re.sub(r'\s*\([^)]*\)', '', term)

        # Specific term substitutions for medical jargon
        replacements = {
            "abnormal": "unusual",
            "defect": "problem with",
            "hypoplasia": "underdevelopment",
            "aplasia": "absence",
            "hyperplasia": "overgrowth",
            "dysplasia": "abnormal growth",
            "atrophy": "wasting",
            "hypertrophy": "enlargement"
        }

        for medical, simple in replacements.items():
            term = re.sub(r'\b' + medical + r'\b', simple,
                          term, flags=re.IGNORECASE)

        return term

    def _update_phenotype_profile(self, current_phenotypes, answers):
        """
        Update the phenotype profile with new answers.

        Args:
            current_phenotypes (list): Current phenotype profile
            answers (list): User answers to questions

        Returns:
            list: Updated phenotype profile
        """
        self.logger.info("Updating phenotype profile with new answers...")

        # Create a copy of the current profile
        updated_phenotypes = current_phenotypes.copy()

        # Process each answer
        for answer in answers:
            term_id = answer['term_id']
            term_name = answer['term_name']
            is_present = answer['is_present']

            # Add the new phenotype to the profile
            updated_phenotypes.append({
                'term_id': term_id,
                'term_name': term_name,
                'present': is_present,
                'confidence': 1.0,  # High confidence for direct patient reports
                'source': 'patient_response'
            })

            status = "Present" if is_present else "Absent"
            self.logger.info(
                f"Added phenotype from response: {term_name} ({term_id}): {status}")

        return updated_phenotypes

    def _update_disease_scores(self, disease_matches, answers):
        """
        Update disease match scores based on new phenotype answers.

        Args:
            disease_matches (list): Current disease matches
            answers (list): User answers to questions

        Returns:
            list: Updated disease matches
        """
        self.logger.info("Updating disease match scores...")

        # Create a copy of the current disease matches
        updated_matches = disease_matches.copy()

        # Process each answer to adjust scores
        for answer in answers:
            term_id = answer['term_id']
            is_present = answer['is_present']
            relevant_diseases = answer['relevant_diseases']

            for disease in updated_matches:
                disease_id = disease['disease_id']

                # If this disease expects this phenotype
                if disease_id in relevant_diseases:
                    # Calculate score adjustment
                    if is_present:
                        # Positive match - increase score
                        # The adjustment is proportional to the term's information content
                        ic_value = self.diagnosis_analyzer.phenotype_matcher.term_information_content.get(
                            term_id, 1.0)
                        adjustment = 0.1 * ic_value  # 10% of IC value
                    else:
                        # Negative match - decrease score
                        adjustment = -0.05  # 5% penalty

                    # Apply adjustment with boundary checks
                    disease['match_score'] = max(
                        0.0, min(1.0, disease['match_score'] + adjustment))

                    # Update matched phenotypes count if positive
                    if is_present:
                        current = disease['matched_phenotypes']
                        if "/" in current:
                            matched, total = current.split("/")
                            matched = int(matched) + 1
                            disease['matched_phenotypes'] = f"{matched}/{total}"

        # Re-sort based on updated scores
        updated_matches.sort(key=lambda x: x['match_score'], reverse=True)

        return updated_matches

    def start_refinement_session(self, file_path, existing_results=None):
        """
        Start an interactive refinement session from clinical notes.

        Args:
            file_path (str): Path to clinical notes file
            existing_results (dict, optional): Existing results from previous phases

        Returns:
            dict: Session state for continuing the interaction
        """
        self.logger.info(f"Starting refinement session for {file_path}")

        try:
            # Either use existing results or process from scratch
            if existing_results is None:
                results = self.diagnosis_analyzer.analyze_clinical_notes(
                    file_path)
            else:
                results = existing_results

            if "error" in results:
                self.logger.error(f"Error in processing: {results['error']}")
                return {"error": results["error"]}

            # Extract key information
            validated_phenotypes = results["validated_phenotypes"]
            disease_matches = results["disease_matches"]
            confidence_assessment = results["diagnosis_assessment"]

            # If confidence is already high, no refinement needed
            if confidence_assessment["confidence_level"] == "definitive":
                self.logger.info(
                    "Diagnosis is already definitive. No refinement needed.")
                return {
                    "status": "complete",
                    "message": "Diagnosis is already definitive. No refinement needed.",
                    "results": results
                }

            # Reset asked phenotypes for new session
            self.asked_phenotypes = set()

            # Get top diseases for discrimination (up to 3)
            top_diseases = disease_matches[:3] if len(
                disease_matches) >= 3 else disease_matches

            # Identify discriminative phenotypes
            discriminative_phenotypes = self._identify_discriminative_phenotypes(
                top_diseases, validated_phenotypes
            )

            # Generate questions
            questions = self._generate_questions(
                discriminative_phenotypes, self.max_questions
            )

            # Create session state
            session_state = {
                "status": "in_progress",
                "iteration": 1,
                "file_path": file_path,
                "validated_phenotypes": validated_phenotypes,
                "disease_matches": disease_matches,
                "confidence_assessment": confidence_assessment,
                "questions": questions,
                "previous_answers": []
            }

            return session_state

        except Exception as e:
            self.logger.error(f"Error starting refinement session: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return {"error": str(e)}

    def process_answers(self, session_state, answers):
        """
        Process user answers and update the session state.

        Args:
            session_state (dict): Current session state
            answers (list): List of answers (y/n) to questions

        Returns:
            dict: Updated session state
        """
        self.logger.info(
            f"Processing answers for iteration {session_state['iteration']}")

        try:
            # Validate session state
            if "status" not in session_state or session_state["status"] != "in_progress":
                return {"error": "Invalid session state"}

            # Extract key information
            validated_phenotypes = session_state["validated_phenotypes"]
            disease_matches = session_state["disease_matches"]
            questions = session_state["questions"]
            previous_answers = session_state["previous_answers"]

            # Process the answers
            processed_answers = []

            for i, answer in enumerate(answers):
                if i >= len(questions):
                    break

                question = questions[i]
                is_present = (answer.lower() == 'y')

                processed_answers.append({
                    'term_id': question['term_id'],
                    'term_name': question['term_name'],
                    'is_present': is_present,
                    'relevant_diseases': question['relevant_diseases']
                })

            # Update the phenotype profile
            updated_phenotypes = self._update_phenotype_profile(
                validated_phenotypes, processed_answers
            )

            # Update disease scores
            updated_matches = self._update_disease_scores(
                disease_matches, processed_answers
            )

            # Re-assess confidence with dynamic threshold
            confidence_assessment = self.diagnosis_analyzer._analyze_top_matches(
                updated_matches,
                sum(1 for p in updated_phenotypes if p.get('present', True))
            )

            # Combine previous and current answers
            all_answers = previous_answers + processed_answers

            # Check if refinement should continue
            max_iterations = 3  # Limit to 3 iterations
            current_iteration = session_state["iteration"]

            if (confidence_assessment["confidence_level"] == "definitive" or
                    current_iteration >= max_iterations):
                # Refinement complete
                status = "complete"
                new_questions = []
            else:
                # Continue refinement
                status = "in_progress"

                # Identify new discriminative phenotypes
                discriminative_phenotypes = self._identify_discriminative_phenotypes(
                    updated_matches[:3], updated_phenotypes
                )

                # Generate new questions
                new_questions = self._generate_questions(
                    discriminative_phenotypes, self.max_questions
                )

                # If no new questions can be generated, end the session
                if not new_questions:
                    status = "complete"

            # Update session state
            updated_state = {
                "status": status,
                "iteration": current_iteration + 1,
                "file_path": session_state["file_path"],
                "validated_phenotypes": updated_phenotypes,
                "disease_matches": updated_matches,
                "confidence_assessment": confidence_assessment,
                "questions": new_questions,
                "previous_answers": all_answers
            }

            return updated_state

        except Exception as e:
            self.logger.error(f"Error processing answers: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return {"error": str(e)}

    def get_final_diagnosis(self, session_state):
        """
        Get the final diagnosis after refinement is complete.

        Args:
            session_state (dict): Final session state

        Returns:
            dict: Final diagnostic assessment
        """
        self.logger.info("Generating final diagnosis...")

        try:
            # Validate session state
            if "status" not in session_state:
                return {"error": "Invalid session state"}

            # Extract final results
            validated_phenotypes = session_state["validated_phenotypes"]
            disease_matches = session_state["disease_matches"]
            confidence_assessment = session_state["confidence_assessment"]

            # Get top match
            top_match = disease_matches[0] if disease_matches else None

            # Format the final diagnosis
            final_diagnosis = {
                "status": "complete",
                "confidence_level": confidence_assessment["confidence_level"],
                "explanation": confidence_assessment["explanation"],
                "recommendation": confidence_assessment["recommendation"],
                "validated_phenotypes": validated_phenotypes,
                "top_match": top_match,
                "disease_matches": disease_matches[:5],  # Top 5 matches
                "refinement_iterations": session_state["iteration"] - 1,
                "total_phenotypes": len(validated_phenotypes),
                "present_phenotypes": len([p for p in validated_phenotypes if p.get('present', True)])
            }

            return final_diagnosis

        except Exception as e:
            self.logger.error(f"Error generating final diagnosis: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return {"error": str(e)}


def run_interactive_session(refinement_engine, file_path):
    """
    Run an interactive command-line refinement session with enhanced output formatting.

    Args:
        refinement_engine: PhenotypeRefinementEngine instance
        file_path (str): Path to clinical notes file
    """
    print("\n===== STARTING PHENOTYPE REFINEMENT SESSION =====")
    print(f"Processing clinical notes from: {file_path}")
    print("This interactive session will help refine the diagnosis.")

    # Start the session
    session_state = refinement_engine.start_refinement_session(file_path)

    if "error" in session_state:
        print(f"Error: {session_state['error']}")
        return

    if session_state["status"] == "complete":
        print(f"Note: {session_state['message']}")
        return

    # Process iterations
    iteration = 1
    while session_state["status"] == "in_progress":
        print(f"\n----- Refinement Iteration {iteration} -----")

        # Show current top matches
        top_matches = session_state["disease_matches"][:3]
        print("\nCurrent Top Matches:")
        for i, match in enumerate(top_matches):
            print(
                f"{i+1}. {match['disease_name']} - Score: {match['match_score']:.4f}")

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
            return

        iteration += 1

    # Generate final diagnosis
    final_diagnosis = refinement_engine.get_final_diagnosis(session_state)

    if "error" in final_diagnosis:
        print(f"Error: {final_diagnosis['error']}")
        return

    # Display final diagnosis in the requested format
    print("\n===== FINAL DIAGNOSIS =====")
    print(f"Confidence Level: {final_diagnosis['confidence_level']}")
    print(f"Explanation: {final_diagnosis['explanation']}")
    print(f"Recommendation: {final_diagnosis['recommendation']}")

    print(
        f"\nRefinement completed after {final_diagnosis['refinement_iterations']} iterations.")
    print(
        f"Total phenotypes: {final_diagnosis['total_phenotypes']}, Present: {final_diagnosis['present_phenotypes']}")

    print("\n===== TOP DISEASE MATCH =====")
    top_match = final_diagnosis["top_match"]
    if top_match:
        # Get disease name from MIM ID if available
        disease_name = refinement_engine._get_disease_name(
            top_match['disease_id'])
        print(f"Disease: {disease_name} ({top_match['disease_id']})")
        print(f"Match Score: {top_match['match_score']:.4f}")

        if "associated_genes" in top_match and top_match["associated_genes"]:
            print(
                f"Associated Genes: {', '.join(top_match['associated_genes'])}")
    else:
        print("No disease matches found.")


# Example usage
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Initialize refinement engine
    refinement_engine = PhenotypeRefinementEngine(
        mim_titles_path="data/mimTitles.csv")

    # Process a sample clinical note
    file_path = "clinical_notes.txt"

    if not os.path.exists(file_path):
        print(f"Error: File {file_path} not found.")
    else:
        # Run interactive session
        run_interactive_session(refinement_engine, file_path)
