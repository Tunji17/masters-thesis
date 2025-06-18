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

### Chapter 4: Comparative Analysis: MedGemma vs Gemma Models

- 4.1 Model Architectures and Capabilities
  - 4.1.1 MedGemma: Medical-Specific Foundation Model
  - 4.1.2 Gemma: General-Purpose Language Model
- 4.2 Relationship Extraction Focus
  - 4.2.1 Why Relationship Extraction Matters for Graph Quality
  - 4.2.2 Model-Specific Strengths and Weaknesses
- 4.3 Comparative Metrics Design
  - 4.3.1 Relationship Accuracy Metrics
  - 4.3.2 Graph Generation Quality Metrics
  - 4.3.3 Domain-Specific Evaluation Criteria
- 4.4 Experimental Design for Model Comparison
  - 4.4.1 Test Dataset Preparation
  - 4.4.2 Evaluation Protocol
- 4.5 Expected Outcomes and Hypotheses

### Chapter 5: Experimental Setup and Implementation

- 5.1 Development Environment
  - 5.1.1 Hardware Configuration (Apple Silicon/MLX)
  - 5.1.2 Software Dependencies
- 5.2 Implementation Details
  - 5.2.1 NER Pipeline Implementation
    - 5.2.1.1 SciSpacy Pipeline Integration
    - 5.2.1.2 UMLS Knowledge Base Loading
  - 5.2.2 Relationship Extraction Implementation
  - 5.2.3 Graph Database Setup
- 5.3 Data Flow from Dataset to Neo4j
  - 5.3.1 Clinical Note Processing
  - 5.3.2 Entity and Relationship Extraction
  - 5.3.3 Cypher Query Generation Pipeline
  - 5.3.4 Graph Population Process
- 5.4 Testing and Validation Framework
  - 5.4.1 Unit Testing Approach
  - 5.4.2 Integration Testing
- 5.5 Limitations of In-Notebook Execution
  - 5.5.1 Memory Constraints
  - 5.5.2 Compute Resource Limitations
  - 5.5.3 Scalability Challenges

### Chapter 6: Results and Evaluation

- 6.1 Named Entity Recognition Results
  - 6.1.1 Entity Detection Performance
  - 6.1.2 Entity Counts and Distribution
  - 6.1.3 UMLS Entity Linking Accuracy
  - 6.1.4 Confidence Score Distributions
  - 6.1.5 Semantic Type Analysis
  - 6.1.6 Abbreviation Resolution Performance
- 6.2 Relationship Extraction Results
  - 6.2.1 MedGemma Performance Analysis
  - 6.2.2 Gemma Performance Analysis
  - 6.2.3 Comparative Results
  - 6.2.4 Relationship Recall and Precision
- 6.3 Graph Construction Outcomes
  - 6.3.1 Graph Quality Metrics
    - 6.3.1.1 Number of Unique Entities with UMLS CUIs
    - 6.3.1.2 Relationship Triple Statistics
    - 6.3.1.3 Graph Connectivity Measures
    - 6.3.1.4 Semantic Type Distribution in Graph
  - 6.3.2 Cypher Query Generation Success Rate
  - 6.3.3 Query Performance Analysis
- 6.4 Error Analysis
  - 6.4.1 Common Failure Patterns
  - 6.4.2 Model Limitations
- 6.5 Statistical Significance Testing

### Chapter 7: Discussion

- 7.1 Interpretation of Results
  - 7.1.1 Key Findings Summary
  - 7.1.2 Model Performance Insights
- 7.2 Implications for Healthcare Information Extraction
  - 7.2.1 Clinical Applications
  - 7.2.2 Research Applications
- 7.3 Comparison with Existing Approaches
  - 7.3.1 Advantages of the Proposed Method
  - 7.3.2 SciSpacy UMLS vs General Knowledge Bases
  - 7.3.3 Trade-offs and Considerations
- 7.4 Limitations of the Study
  - 7.4.1 Dataset Limitations
    - 7.4.1.1 Dataset Availability Constraints
    - 7.4.1.2 Data Quality Issues
  - 7.4.2 Methodological Constraints
  - 7.4.3 Manual Verification Requirements for Clinical Data
- 7.5 Practical Deployment Considerations

### Chapter 8: Conclusions and Future Work

- 8.1 Summary of Contributions
  - 8.1.1 Technical Contributions
  - 8.1.2 Theoretical Contributions
- 8.2 Achievement of Research Objectives
- 8.3 Future Research Directions
  - 8.3.1 Model Improvements
  - 8.3.2 Expanding Beyond UMLS to Other Medical Knowledge Bases
  - 8.3.3 Scaling to Larger Clinical Corpora
  - 8.3.4 Extended Applications
  - 8.3.5 Multi-modal Information Extraction
- 8.4 Recommendations for Practitioners
- 8.5 Final Remarks

## Back Matter (preserved)

- Bibliography
- Appendices
