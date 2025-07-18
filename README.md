# Masters Thesis Topic - Graph Information Extraction from clinical notes (Unstructured Text) in Healthcare

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

``` bash
docker run \
    --publish=7474:7474 --publish=7687:7687 \
    --volume=$HOME/neo4j/data:/data \
    neo4j
```

- [ ] **Graph Visualization**: Use a graph visualization library, such as D3.js or Cytoscape.js, to display graphs.

- [ ] **Ihor/gliner-biomed-bi-large-v1.0**: Use the [Ihor/gliner-biomed-bi-large-v1.0](https://huggingface.co/Ihor/gliner-biomed-bi-large-v1.0) model for named entity recognition (NER). This model is specifically designed for biomedical text and should be able to extract relevant information from clinical notes.

- [ ] **Entity linking**: Explore the use of entity linking techniques to improve the accuracy of named entity recognition. This could involve using external knowledge bases, such as wikipedia. (Wikipedia is not a good source for clinical notes, but it could be used to link entities to other relevant information.)

- [ ] **Closed Information Extraction Extraction**: I need a smart and cheap way to extract relationships between the NEs. This could involve using a reasoning model.(Qwen3 32b??). This model could also be used to generate Cypher scripts for training.

- [ ] **DataSet**: Use the [AGBonnet/augmented-clinical-notes](https://huggingface.co/datasets/AGBonnet/augmented-clinical-notes) as the dataset for training and testing the system. This dataset contains de-identified clinical notes and other relevant information. It will also require some pre-processing to convert the data into a format suitable for training and testing the system.

- [ ] **Cypher Query Generation**: Use a pre-trained model to convert natural language queries into Cypher queries.

---

## Future Work

- [ ] **Coreference Resolution**: Explore the use of coreference resolution techniques to improve the accuracy of named entity recognition. This could involve using pre-trained models, such as AllenNLP or SpaCy, to identify and resolve coreferences in the text data.

- [ ] **Support for other Data sources**: Extend the system to support other data sources, such as electronic health records (EHRs), clinical trial data, and other unstructured text sources. This could involve developing new algorithms and techniques for extracting relevant information from these data sources.
- [ ] **Reinforcement Learning**: Explore the use of reinforcement learning techniques to improve the accuracy and efficiency of graph information extraction. This could involve training models to learn optimal strategies for extracting relevant information from unstructured text.

## Problems Encountered

- [ ] **Coreference Resolution**: Cannot use coreferee from SpaCy because it is not supported in the latest version of SpaCy and i can't install it on ARM. Need to find an alternative solution for coreference resolution.
