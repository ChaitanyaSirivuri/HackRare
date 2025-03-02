## PhenoGenisis

## Overview

PhenoGenisis (PHENotype-driven Intelligent eXpert system) is a comprehensive diagnostic tool for rare diseases that processes clinical notes through an integrated pipeline of five specialized phases. By leveraging phenotypic information extracted from clinical documentation and matching it to disease databases, PhenoGenisis assists clinicians in diagnosing rare genetic disorders with increased accuracy and efficiency.

## Key Features

- **Automated Phenotype Extraction**: Identifies Human Phenotype Ontology (HPO) terms from unstructured clinical notes
- **Semantic Similarity Matching**: Connects patient phenotypes to potential diseases using information content-based similarity
- **Dynamic Confidence Thresholding**: Adapts diagnostic confidence assessment based on phenotype quantity and quality
- **Interactive Refinement**: Improves diagnosis through targeted questions about discriminative phenotypes
- **Novel Disease Detection**: Identifies potential new syndromes or phenotypic expansions of known disorders


## System Architecture

PhenoGenisis consists of five integrated phases:

1. **Phase 1: Phenotype Ingestion \& Validation**
    - Extracts HPO-coded phenotypes from clinical notes
    - Validates terms and resolves synonyms and hierarchical relationships
2. **Phase 2: Initial Phenotype Matching**
    - Matches phenotypes to diseases from OMIM and other databases
    - Calculates similarity scores using semantic relationships
3. **Phase 3: Dynamic Threshold Analysis**
    - Determines diagnostic confidence based on adaptive thresholds
    - Provides confidence levels: definitive, probable, possible, or novel
4. **Phase 4: Iterative Phenotype Refinement**
    - Improves diagnosis through interactive questioning
    - Focuses on discriminative phenotypes with high information content
5. **Phase 5: Novel Disease Flagging**
    - Identifies cases that may represent novel diseases or presentations
    - Calculates a novelty score based on shared and unique phenotypes

## Installation

```bash
# Clone the repository
git clone https://github.com/username/PhenoGenisis.git
cd PhenoGenisis

# Create and activate a virtual environment
python -m venv env
source env/bin/activate  # On Windows: env\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download and prepare data files
python phase5.py
```


## Usage

### Streamlit Web Application

```bash
# Launch the Streamlit web interface
streamlit run app.py
```


## Required Data

PhenoGenisis requires several data files to function properly:

- `hp.obo`: Human Phenotype Ontology terms and structure
- `hpo_synonyms.txt`: HPO term synonyms
- `phenotype_to_genes.txt`: Mapping of phenotypes to genes and diseases
- `mimTitles.csv`: OMIM disease titles and IDs

Place these files in the `data/` directory or specify custom paths in the configuration.

## Dependencies

- Python 3.8+
- NumPy
- pandas
- Streamlit (for web interface)
- NLTK (for text processing)
- scikit-learn (for machine learning components)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.


