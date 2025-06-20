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

- 4.1 Implementation Environment
  - 4.1.1 MLX Framework and Apple Silicon Optimization
  - 4.1.2 Development Setup and Dependencies
- 4.2 Language Model Comparison for Relationship Extraction
  - 4.2.1 MedGemma vs Gemma: Architecture and Domain Specialization
  - 4.2.2 Comparative Evaluation Framework
  - 4.2.3 Prompt Engineering and Output Processing
- 4.3 Experimental Validation
  - 4.3.1 Testing Protocol and Performance Metrics
  - 4.3.2 Implementation Constraints and Scalability

### Chapter 5: Results, Evaluation and Discussion

- 5.1 Experimental Results
  - 5.1.1 Named Entity Recognition Performance
  - 5.1.2 Model Comparison: MedGemma vs Gemma
  - 5.1.3 Knowledge Graph Construction Outcomes
- 5.2 Performance Analysis and Evaluation
  - 5.2.1 Quantitative Analysis and Statistical Testing
  - 5.2.2 Error Analysis and Model Limitations
  - 5.2.3 Graph Quality Metrics
- 5.3 Discussion and Implications
  - 5.3.1 Key Findings and Model Performance Insights
  - 5.3.2 Comparison with Existing Approaches
  - 5.3.3 Healthcare Applications and Practical Considerations
- 5.4 Study Limitations and Constraints
  - 5.4.1 Dataset and Methodological Limitations
  - 5.4.2 Implementation Constraints

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
