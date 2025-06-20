# Thesis Topic: Transforming Unstructured Clinical Text into Actionable Knowledge Graphs: A Multi-Model Approach for Healthcare Information Extraction

## Thesis Structure - Table of Contents

## Front Matter (preserved)

- Declaration of Authorship
- Abstract  
- Acknowledgements

## Main Chapters

### Chapter 1: Introduction

- 1.1 Background and Context
  - 1.1.1 Clinical Documentation in Healthcare
  - 1.1.2 Challenges with Unstructured Clinical Data
- 1.2 Research Motivation
  - 1.2.1 The Need for Structured Information Extraction
  - 1.2.2 Graph Databases for Healthcare Data
- 1.3 Research Question and Objectives
  - 1.3.1 Primary Research Question
  - 1.3.2 Specific Objectives
- 1.4 Ethical Considerations and Data Privacy
  - 1.4.1 Handling Sensitive Clinical Information
  - 1.4.2 Privacy-Preserving Techniques
- 1.5 Thesis Contributions

### Chapter 2: Literature Review and Theoretical Background

- 2.1 Natural Language Processing in Healthcare
  - 2.1.1 Evolution of Clinical NLP
  - 2.1.2 Current State-of-the-Art Approaches
- 2.2 Named Entity Recognition in Medical Text
  - 2.2.1 Traditional NER Methods
  - 2.2.2 Deep Learning Approaches for Medical NER
  - 2.2.3 Biomedical Language Models
- 2.3 Entity Linking Techniques
  - 2.3.1 Knowledge Base Linking Methods
  - 2.3.2 UMLS and MeSH Knowledge Bases
  - 2.3.3 SciSpacy's TF-IDF Character N-gram Matching
  - 2.3.4 Challenges in Medical Entity Linking
- 2.4 From Relationship Extraction to Knowledge Graphs
  - 2.4.1 Relationship Extraction Methods
    - 2.4.1.1 Rule-Based Approaches
    - 2.4.1.2 Machine Learning Techniques
    - 2.4.1.3 Large Language Models for Relationships
  - 2.4.2 Knowledge Graph Construction
    - 2.4.2.1 Graph Database Technologies
    - 2.4.2.2 Neo4j and Cypher Query Language
  - 2.4.3 Clinical Knowledge Graph Applications
- 2.5 Related Work Summary and Research Gap

### Chapter 3: Methodology

- 3.1 System Architecture and Pipeline
  - 3.1.1 Overall System Design
  - 3.1.2 Component Integration Flow
- 3.2 Data and Preprocessing
  - 3.2.1 Clinical Notes Dataset and Preparation
- 3.3 Entity Recognition and Linking
  - 3.3.1 GLiNER Model for Medical Entity Recognition
  - 3.3.2 UMLS Entity Linking with SciSpacy
  - 3.3.3 Confidence Scoring and Abbreviation Detection
- 3.4 Relationship Extraction
  - 3.4.1 Model Selection and Prompt Engineering
  - 3.4.2 Output Parsing and Validation
- 3.5 Knowledge Graph Construction
  - 3.5.1 Graph Schema Design
  - 3.5.2 Cypher Query Generation and Storage
- 3.6 Implementation Optimization
  - 3.6.1 Parallel Processing and Resource Management

### Chapter 4: Implementation and Comparative Analysis: MedGemma vs Gemma Models

- 4.1 Hardware Requirements and Computational Environment
  - Apple Silicon M1/M2 hardware specifications and performance benchmarks
  - Processing time analysis by pipeline component and memory usage patterns
  - Complete environment specification for reproducibility (dependency versions, configuration details)
  - Seed management and deterministic processing setup
- 4.2 Language Model Comparison for Relationship Extraction
  - 4.2.1 MedGemma vs Gemma: Architecture and Domain Specialization
    - MedGemma-4B-IT vs Gemma-3-4B-IT specifications
    - Medical domain training differences and implications
    - MLX framework implementation for both models
  - 4.2.2 Comparative Evaluation Framework
    - Identical prompt engineering for fair comparison
    - Output parsing and validation methodology
    - Performance measurement criteria
  - 4.2.3 Experimental Design for Model Comparison
    - Same dataset processing for both models
    - Statistical testing framework for significance
    - Controlled variables and comparison methodology
- 4.3 Experimental Validation
  - 4.3.1 Quantitative Evaluation Metrics
    - Entity Recognition: Precision/Recall/F1 by entity type
    - Entity Linking: Accuracy and confidence score analysis
    - Relationship Extraction: Triple-level accuracy metrics
    - Knowledge Graph Quality: Connectivity and coverage metrics
  - 4.3.2 Statistical Analysis Methodology
    - Hypothesis testing framework for MedGemma vs Gemma
    - Significance testing and confidence intervals
    - Error analysis categorization and measurement

### Chapter 5: Results, Evaluation and Discussion

- 5.1 Performance Results and Model Comparison
  - 5.1.1 Entity Recognition and Linking Results
    - GLiNER performance by entity type and confidence scores
    - SciSpacy linking success rates and UMLS coverage
    - Processing speed and accuracy trade-offs
  - 5.1.2 MedGemma vs Gemma Comparative Results
    - Quantitative relationship extraction comparison
    - Statistical significance of performance differences
    - Relationship type analysis and model preferences
    - Resource utilization comparison (time/memory)
- 5.2 Knowledge Graph Analysis
  - 5.2.1 Graph Construction Results
    - Final graph statistics (nodes, edges, connectivity)
    - Medical concept coverage and relationship diversity
    - Example Cypher queries and complex traversals
  - 5.2.2 Quality Assessment
    - Error propagation analysis through pipeline stages
    - Data integrity and consistency validation
    - Graph completeness and clinical relevance
- 5.3 Error Analysis and Limitations
  - 5.3.1 Component-Specific Error Analysis
    - GLiNER failure modes and edge cases
    - SciSpacy linking gaps and ambiguity resolution
    - LLM hallucination detection and filtering effectiveness
  - 5.3.2 Computational and Scale Limitations
    - Processing time constraints and scalability challenges
    - Memory limitations affecting batch size and parallelization
    - Hardware-specific optimization limitations
  - 5.3.3 Dataset and Methodological Constraints
    - Synthetic data limitations vs real clinical text
    - Sample size constraints and generalizability
    - MLX framework limitations and portability concerns

### Chapter 6: Conclusions and Future Work

- 6.1 Summary of Contributions
  - 6.1.1 Technical Contributions
  - 6.1.2 Theoretical Contributions
- 6.2 Achievement of Research Objectives
- 6.3 Future Research Directions
  - 6.3.1 Model Improvements
  - 6.3.2 Expanding Beyond UMLS to Other Medical Knowledge Bases
  - 6.3.3 Scaling to Larger Clinical Corpora
  - 6.3.4 Multi-modal Information Extraction
- 6.4 Final Remarks

## Back Matter (preserved)

- Bibliography
- Appendices
