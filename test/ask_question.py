import argparse
import json
import requests

def main():
    parser = argparse.ArgumentParser(description="Send a query to the server and retrieve answers.")
    parser.add_argument('--rag', action='store_true', help='Use RAG pipeline to answer query.')
    parser.add_argument('--ex', action='store_true', help='Use extractive QA pipeline to answer query.')
    parser.add_argument("--params", type=str, default='{}', help="JSON string of query parameters (defaults to empty).")
    parser.add_argument('--query', type=str, required=True, help='The query to send to the server.')
    
    args = parser.parse_args()

    # Load query params or use default
    try:
        query_params = json.loads(args.params)
    except json.JSONDecodeError:
        print("Invalid JSON format for --params. Using default empty params.")
        query_params = {}

    # Set different default parameters for RAG and Extractive QA pipelines
    if args.rag and not query_params:
        # Default parameters for RAG pipeline
        query_params = {
        "BM25Retriever": 
        {
            "top_k": 10
        },
        "DenseRetriever": 
        {
            "top_k": 10    
        },
        "Ranker":
        {
            "top_k": 4
        },
        "Generator":
        {
            "temperature": 0.75,     # For controlling randomness in generation
            "max_new_tokens": 130,   # Maximum tokens for LLM generation
            "top_p": 0.95,          # For nucleus sampling in generation
            "post_processing": True,
            "apply_cr_threshold": True
        }
        }
    elif args.ex and not query_params:
        # Default parameters for Extractive pipeline
        query_params = {
        "BM25Retriever": 
        {
            "top_k": 10
        },
        "DenseRetriever": 
        {
            "top_k": 10    
        },
        "Ranker":
        {
            "top_k": 4
        },
        "Reader":
        {
            "top_k": 4
        }
        }


    # Prepare request body and endpoint
    query = args.query
    request_body = {
        "query": query,
        "params": query_params
    }

    # Choose the appropriate endpoint based on the pipeline
    endpoint = "rag-query" if args.rag else "extractive-query" if args.ex else None
    url = f"http://localhost:8000/{endpoint}"

    try:
        # Send the request to the server
        r = requests.post(url=url, json=request_body)
        r.raise_for_status()  # Raise an HTTPError for bad responses
        json_response = r.json()
        print(json_response["answers"])
    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")
    except KeyError:
        print("Unexpected response structure:", json_response)

if __name__ == "__main__":
    main()
