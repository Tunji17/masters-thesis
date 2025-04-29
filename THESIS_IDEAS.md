# Thesis Topic - Graph Information Extraction from clinical notes (Unstrctured Text) in Healthcare

## Overview

User has some unstructured text data (clinical notes) and wants to query and visualize the data in the form of a graph. The goal is to

---

## How it works

### Step 1: Convert unstructured text data into a graph

1. Extract Named Entities (NEs) from the text data using Clinical Bert
2. Extract relationships between the NEs using hopefully a reasoning model
3. Generate Cypher scripts to create the graph in a Neo4j database using a pre-trained model

## Step 2: Query the graph using Cypher queries

1. User inputs natural language query into text input box.
2. Convert the natural language query into Cypher queries using a pre-trained model.
3. Execute the Cypher queries on the Neo4j database and retrieve the results.
4. Display the results as a graph visualization

---

## Technical Details

- [ ] **Graph Database**: Use Neo4j as the graph database to store and query the extracted information.

- [ ] **Graph Visualization**: Use a graph visualization library, such as D3.js or Cytoscape.js, to display graphs.

- [ ] **Clinical Bert**: Use [Clinical Bert](https://huggingface.co/medicalai/ClinicalBERT) to extract named entities from the clinical notes. Hoping to post-train the model to generate Cypher scripts from NEs and relationships.

- [ ] **Relationship Extraction**: I need a smart and cheap way to extract relationships between the NEs. This could involve using a reasoning model.(Qwen3 32b??). This model could also be used to generate Cypher scripts for training.

- [ ] **DataSet**: Use the [AGBonnet/augmented-clinical-notes](https://huggingface.co/datasets/AGBonnet/augmented-clinical-notes) as the dataset for training and testing the system. This dataset contains de-identified clinical notes and other relevant information. It will also require some pre-processing to convert the data into a format suitable for training and testing the system.

- [ ] **Cypher Query Generation**: Use a pre-trained model to convert natural language queries into Cypher queries.

---

## Future Work

- [ ] **Support for other Data sources**: Extend the system to support other data sources, such as electronic health records (EHRs), clinical trial data, and other unstructured text sources. This could involve developing new algorithms and techniques for extracting relevant information from these data sources.
- [ ] **Reinforcement Learning**: Explore the use of reinforcement learning techniques to improve the accuracy and efficiency of graph information extraction. This could involve training models to learn optimal strategies for extracting relevant information from unstructured text.