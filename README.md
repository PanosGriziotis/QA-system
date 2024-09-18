> [!WARNING]
> This repository is currently a work in progress and is not yet ready for production use. Features and functionalities are being actively developed and tested. Use at your own risk.

# QA-subsystem

This repository is part of a project aiming to develop a Greek Question-Answering (QA) system that integrates with a closed-domain virtual assistant (dialogue system). As a case study, a greek-speaking conversational agent for COVID-19 is selected.

The repository contains a simple Haystack application with a REST API for indexing and querying purposes.

The application includes:

- An Elasticsearch container
- A Question-Asnwering (QA) REST API container: This container integrates the Haystack logic and uses pipelines for indexing unstructured text data in Elasticsearch document store and querying.

You can find more information in the [Haystack documentation](https://docs.haystack.deepset.ai/v1.25/docs/intro).

### Steps to Set Up

Before you begin, ensure you have Python and Docker installed on your system. 

1. **Clone this repository.**

2. **Run the services**

    Spin up the multi-container application (Elasticsearch + REST API) using Docker Compose:

    ```bash
    docker-compose up -d
    ```

4. **Verify that the haystack service is ready:**

    Open a new terminal and run:

    ```bash
    curl http://localhost:8000/ready
    ```

    You should get `true` as a response.

    Note: The haystack service requires some time to download and load the models after it starts.

## Indexing

To populate the application with data about COVID-19, run the following:

```bash
python3 src/external_data/ingest_data_to_doc_store.py
```

You can also index your own text files using the file-upload endpoint:

```bash
curl -X 'POST' \
'http://localhost:8000/file-upload?keep_files=false' \
-H 'accept: application/json' \
-H 'Content-Type: multipart/form-data' \
-F 'files=@YOUR-TEXT-FILE'
```

Note: Acceptable file formats are .txt, .json, .jsonl, .pdf, .docx.

## Querying

There are two query endpoints available for inferring answers to queries. These endpoints provide different approaches to answering queries:

### Retrieval-Augmented Generation (RAG) Query

- **Description:** This endpoint utilizes a Retrieval-Augmented Generator (RAG) pipeline. It employs a domain-adapted Dense Retriever based on bi-encoder sentence transformer model for retrieving relevant documents followed by a cross-encoder Ranker component. The Generator is based on [Meltemi-7B-Instruct-v1](https://huggingface.co/ilsp/Meltemi-7B-Instruct-v1), an instruct version of Meltemi-7B, the first Greek Large Language Model (LLM).

### Extractive Question Answering (QA) Query

- **Description:** This endpoint utilizes an Extractive QA pipeline based on the Retriever-Reader framework. The answer is extracted as a span from the top-ranked retrieved document. The Reader component is a fine-tuned [multilingual DeBERTaV3](https://huggingface.co/microsoft/mdeberta-v3-base) on SQuAD with further fine-tuning on COVID-QA-el_small, which is a translated small version of the COVID-QA dataset.

### Querying the application

If you want to test the app and get a direct answer to a query of your choic, you can run the test/ask_question.py script. Include the --ex flag to use the extractive QA endpoint or the --rag flag to use the RAG endpoint for yielding the answer:

```bash
python3 test/ask_question.py --rag --query "Πώς μεταδίδεται ο covid-19;"
```

You can query the endpoint using curl to get the full result response, including the answer, retrieved documents, confidence scores, and more. You can also configure the pipeline's parameters as you wish. For example, to query the application using the RAG query pipeline with specific parameters run:

```bash
curl -X POST http://localhost:8000/rag-query \
     -H "Content-Type: application/json" \
     -d '{
            "query": "Πώς μεταδίδεται η covid-19;", 
            "params": {
                "Retriever": {"top_k": 10}, 
                "Ranker": {"top_k": 5}, 
                "Generator": {"max_new_tokens": 100}
            }
        }'
```
