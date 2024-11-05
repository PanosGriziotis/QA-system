
# Description

This repository is part of a project aimed at developing a Greek-language Question-Answering (QA) system for integration with a closed-domain virtual assistant focused on COVID-19, named [Theano](https://aclanthology.org/2021.nlp4posimpact-1.5/)

The QA system can operate as a standalone application or be accessed via an API for integration into other applications

In the `src` directory you can find the following main components:

- An Elasticsearch container that acts as the Document Store.
- A Question-Asnwering (QA) REST API container: This container integrates the [Haystack](https://docs.haystack.deepset.ai/v1.25/docs/intro) logic and uses pipelines for indexing documents in the Document Store and receiving an answer for a given query. 

## Architecture Overview
![alt text](https://github.com/PanosGriziotis/QA-subsystem-thesis/blob/main/qa_system_architecture_v3.png?raw=true)

### Retrieval-Augmented Generation (RAG) Query

- **Description:** This endpoint utilizes a Retrieval-Augmented Generator (RAG) pipeline. The answer is a free text generated from the retrieved documents. The Generative Reader component is based on the monolingual instruction-following LLM [Meltemi-7B-Instruct-v1.5](https://huggingface.co/ilsp/Meltemi-7B-Instruct-v1.5).

### Extractive Question Answering (QA) Query

- **Description:** This endpoint utilizes an Extractive QA pipeline. The answer is extracted as a span from a single document. The Extractive Reader component is a fine-tuned [multilingual DeBERTaV3](https://huggingface.co/microsoft/mdeberta-v3-base) model on SQuAD V2 and the [COVID-QA-el_small](https://huggingface.co/datasets/panosgriz/COVID-QA-el-small) dataset. 

### Set up steps

Before you begin, ensure that Python version 3.8 or higher and Docker are installed on your system. To run the QA system models for inference, a GPU with at least 10GB of available memory is required.

1. **Clone this repository.**

2. **Run the services**

    Spin up the multi-container application (Elasticsearch + Haystack REST API) using Docker Compose:

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
python3 external_data/ingest_data_to_doc_store.py
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


### Querying the application

If you want to directly use the app in terminal and pose a query of your choice, you can run the `ask_qa_system.py` script. Include the `--ex` flag to use the extractive QA endpoint or the `--rag` flag to use the RAG endpoint for receiving an answer. 

```bash
python3 ask_qa_system.py --rag 
```

You can optionally pass query hyperaparameters for each of the query pipeline components like this: 

```bash
python3 ask_qa_system.py --rag '{"BM25Retriever": {"top_k": 10}, "DenseRetriever": {"top_k": 10}, "Reranker": {"top_k": 6}, "Generative_Reader": {"max_new_tokens": 130}}
```

After running the script type your query and get a response. To quit the app just type `quit`. 

To send a POST request to the QA system's API and get the full response, including the answer to your query, retrieved documents, confidence scores, and more, use curl:

```bash
curl -X POST http://localhost:8000/rag-query \
     -H "Content-Type: application/json" \
     -d '{
            "query": "Πώς μεταδίδεται η covid-19;", 
            "params": {
                "BM25Retriever": {"top_k": 10},
                "DenseRetriever": {"top_k: 10},
                "ReRanker": {"top_k": 6}, 
                "GenerativeReader": {"max_new_tokens": 100}
            }
        }'
```
