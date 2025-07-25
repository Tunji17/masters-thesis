# Thesis Topic: Transforming Unstructured Biomedical Text into Actionable Knowledge Graphs: A Multi-Model Approach for Healthcare Information Extraction

## Thesis Structure - Table of Contents

## Front Matter

- Declaration of Authorship
- Abstract  
- Acknowledgements

## Main Chapters

### Chapter 1: Introduction

- 1.1 Background and Context
  - 1.1.1 Biomedical Text and Knowledge Sources
  - 1.1.2 Challenges with Unstructured Biomedical Text
- 1.2 Research Motivation
  - 1.2.1 The Need for Structured Information Extraction
  - 1.2.2 Graph Databases for Biomedical Knowledge
- 1.3 Research Question and Objectives
  - 1.3.1 Primary Research Question
  - 1.3.2 Specific Objectives
- 1.4 Ethical Considerations and Data Privacy
  - 1.4.1 Responsible Use of Biomedical Data
- 1.5 Thesis Contributions

### Chapter 2: Literature Review and Theoretical Background

- 2.1 Natural Language Processing in Biomedicine
  - 2.1.1 Evolution of Biomedical NLP
  - 2.1.2 Current State-of-the-Art Approaches
- 2.2 Named Entity Recognition in Biomedical Text
  - 2.2.1 Traditional NER Methods
  - 2.2.2 Deep Learning Approaches for Biomedical NER
  - 2.2.3 Biomedical Language Models
- 2.3 Entity Linking Techniques
  - 2.3.1 Knowledge Base Linking Methods
  - 2.3.2 UMLS and MeSH Knowledge Bases
  - 2.3.3 SciSpacy's TF-IDF Character N-gram Matching
  - 2.3.4 Challenges in Medical Entity Linking
- 2.4 From Relationship Extraction to Knowledge Graphs
  - 2.4.1 Relationship Extraction Methods
    - Rule-Based Approaches
    - Machine Learning Techniques
    - Large Language Models for Relationships
  - 2.4.2 Knowledge Graph Construction
    - Graph Database Technologies
    - Neo4j and Cypher Query Language
  - 2.4.3 Biomedical Knowledge Graph Applications
- 2.5 Related Work Summary and Research Gap

### Chapter 3: Methodology

- 3.1 System Architecture and Pipeline
  - 3.1.1 Overall System Design
  - 3.1.2 Component Integration Flow
- 3.2 Data and Preprocessing
  - 3.2.1 Biomedical Literature Dataset and Preparation
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

### Chapter 4: Results, Evaluation and Discussion

- 4.1 Experimental Setup
  - 4.1.1 BioRED Dataset and Evaluation Metrics
  - 4.1.2 Model Configurations and Prompt Strategies
- 4.2 Named Entity Recognition Performance
  - 4.2.1 GLiNER Results: Threshold Impact and Matching Strategies
  - 4.2.2 Error Analysis and Entity Boundary Challenges
- 4.3 Relationship Extraction Performance
  - 4.3.1 Gemma vs MedGemma: Strategy Comparison and Low F1 Analysis
  - 4.3.2 Model Limitations and Output Quality Issues
- 4.4 End-to-End Pipeline Evaluation
  - 4.4.1 Processing Efficiency and Scalability
- 4.5 Discussion
  - 4.5.1 Key Findings and Clinical Applicability

### Chapter 5: Conclusions and Future Work

- 5.1 Summary of Contributions
  - 5.1.1 Technical Contributions
  - 5.1.2 Theoretical Contributions
- 5.2 Achievement of Research Objectives
- 5.3 Future Research Directions
  - 5.3.1 Model Improvements
  - 5.3.2 Expanding Beyond UMLS to Other Medical Knowledge Bases
  - 5.3.3 Scaling to Larger Clinical Corpora
- 5.4 Final Remarks

## Back Matter

- Bibliography

## Appendices

### Appendix A: Comprehensive Evaluation Results and Statistical Analysis

- Complete BioRED evaluation matrices for all 15 model configurations
- GLiNER threshold sensitivity analysis (0.3, 0.5, 0.7 confidence levels)
- Matching strategy performance comparison (exact, partial, text-based)
- Per-relation-type performance breakdowns across 8 BioRED relation types
- Entity-specific NER performance across 6 biomedical entity types

### Appendix B: Technical Implementation and Reproducibility Guide

- Complete prompt templates for all three relationship extraction strategies
  - Basic prompting approach with system instructions
  - Few-shot prompting with biomedical examples
  - Structured JSON output prompting with schema validation
- Neo4j schema definitions with complete Cypher query templates
- GLiNER biomedical model configuration and optimization parameters

### Appendix C: Error Analysis and Failure Case Studies

- Comprehensive analysis of relationship extraction failures (90%+ missed relationships)
- Prompt strategy effectiveness analysis across different biomedical text structures
- Domain-specific model underperformance investigation (MedGemma vs Gemma)
