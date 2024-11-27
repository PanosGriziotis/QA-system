
# Description

This repository is part of a project aimed at developing a Greek-language Question-Answering (QA) subsystem for integration with a closed-domain virtual assistant focused on COVID-19, named [Theano](https://aclanthology.org/2021.nlp4posimpact-1.5/).

The QA system can operate as a standalone application or be accessed via an API for integration into other applications.

The `docker-compose` file enables you to start the main backend services:

- An [Elasticsearch](https://www.elastic.co/elasticsearch) container, which serves as the Document Store.
- A Question-Asnwering (QA) REST API container based on the [Haystack](https://docs.haystack.deepset.ai/v1.25/docs/intro) logic. This API consists of an indexing pipeline for storing text documents in the Document Store and a query pipeline for returning answers to given queries.

## Architecture Overview
![alt text](https://github.com/PanosGriziotis/QA-subsystem-thesis/blob/main/qa_system_architecture_v3.png?raw=true)

There are two types of query pipelines available for inferring answers to queries, based on the following approaches:

### Retrieval-Augmented Generation (RAG) 

- **Description:** This query pipeline utilizes a Retrieval-Augmented Generation (RAG) method. The answer is a free text generated from retrieved documents. The Generative Reader component is based on the monolingual instruction-following LLM [Meltemi-7B-Instruct-v1.5](https://huggingface.co/ilsp/Meltemi-7B-Instruct-v1.5).

### Extractive Question Answering

- **Description:** This query pipeline utilizes an Extractive QA method. The answer is extracted as a span from a single retrieved document. The Extractive Reader component is a fine-tuned [multilingual DeBERTaV3](https://huggingface.co/microsoft/mdeberta-v3-base) model on SQuAD V2 and the [COVID-QA-el_small](https://huggingface.co/datasets/panosgriz/COVID-QA-el-small) dataset. 

## Requirements

Before you begin, ensure that:

- **Python** version 3.8 or higher is installed on your system.
- **Docker** is installed on your system.

To run the QA system models for inference, you will need:

- A **GPU** with at least 10GB of available memory.
- **CUDA** version 11.8 or higher.

## Set up steps

1. **Clone this repository.**

2. **Run the services**

    Spin up the multi-container application (Elasticsearch + Haystack REST API) using Docker Compose:

    ```bash
    docker-compose up -d
    ```

4. **Verify that the haystack service is ready:**

    Open a new terminal and run:

    ```bash
    curl http://localhost:8001/ready
    ```

    You should get `true` as a response.

    Note: The haystack service requires some time to download and load the models after it starts.


## Indexing documents in the Document Store

To populate the backend with data about COVID-19 collected from trusted sources (Wikipedia, ECDC, WHO, NPHO, covid19.gov.gr), run the following:

```bash
python3 external_data/ingest_data_to_doc_store.py
```

You can index your own text data using the file-upload endpoint. Text data are extracted from the uploaded files, converted into document objects, and preprocessed before being stored in the Document Store

```bash
curl -X 'POST' \
'http://localhost:8001/file-upload?keep_files=false' \
-H 'accept: application/json' \
-H 'Content-Type: multipart/form-data' \
-F 'files=@YOUR-TEXT-FILE'
```

Note: Acceptable file formats are .txt, .json, .jsonl, .pdf, .docx.


## Start Asking the QA System

### Streamlit Web UI

This is a simple UI app built with Streamlit, providing an interactive user interface for querying the backend server.

### Features

- **Pipeline Selection**: Choose between RAG and Extractive QA query pipelines.
- **Customizable Parameters**: Enter JSON parameters to control the query pipeline components.
- **Interactive Query Input**: Type your question and get answers from the backend server.

### Setup steps

1. **Create a virtual environment:**

```bash
python3 -m venv venv
```
2. **Activate the virtual environment:**

```bash
source venv/bin/activate
```

3. **Install dependecies:**

```bash
pip install -r ui/requirements.txt
```

4. **Ensure Backend services are running:**

- The Haystack service should be running at `http://localhost:8001`
- Elasticsearch should be running at `http://localhost:9200`
- Ensure that documents are indexed in the Document Store. You can inspect indices using this command:

```bash
curl -X GET "localhost:9200/_cat/indices?v"
```

5. **Launch the Web UI:**


```bash
streamlit run ui/web_app.py
```

Access the app in your browser at `http://localhost:8501`

### Using the QA System's REST API

To query the QA system directly via REST API, you can send a POST request. This request will return the full JSON response, including the answer to your query, retrieved documents, confidence scores, and other details.

```bash
curl -X POST http://localhost:8001/rag-query \
     -H "Content-Type: application/json" \
     -d '{
            "query": "Πώς μεταδίδεται ο covid-19;", 
            "params": {
                "BM25Retriever": {"top_k": 10},
                "DenseRetriever": {"top_k": 10},
                "Reranker": {"top_k": 6}, 
                "GenerativeReader": {"max_new_tokens": 150}
            }
        }'
```

Replace the query and parameters as needed. This example is configured to use the RAG pipeline at `http://localhost:8001/rag-query`
