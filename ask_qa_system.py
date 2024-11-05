import argparse
import json
import requests

def main():
    parser = argparse.ArgumentParser(description="Send a query to the server and retrieve answers.")
    parser.add_argument('--rag', action='store_true', help='Use RAG pipeline to answer query.')
    parser.add_argument('--ex', action='store_true', help='Use extractive QA pipeline to answer query.')
    parser.add_argument("--params", type=str, default='{}', help="JSON string of query parameters.")
    #parser.add_argument('--query', type=str, required=True, help='The query to send to the server.')
    
    args = parser.parse_args()

    # Load query params or use default
    try:
        query_params = json.loads(args.params)
    except json.JSONDecodeError:
        print("Invalid JSON format for --params. Using default empty params.")
        query_params = {}

    if args.rag and not query_params:
        # Default parameters for RAG pipeline
        query_params = {
        "BM25Retriever": {"top_k": 10}, 
        "DenseRetriever": {"top_k": 10},
        "Reranker":{"top_k": 4},
        "GenerativeReader":{"temperature": 0.75, "max_new_tokens": 150, "top_p": 0.95, "post_processing": True},
        "Responder": {"threshold": 0.17}
        }

    elif args.ex and not query_params:
        # Default parameters for Extractive pipeline
        query_params = {
        "BM25Retriever": {"top_k": 10}, 
        "DenseRetriever": {"top_k": 10},
        "Reranker":{"top_k": 6},
        "ExtractiveReader":{"top_k": 6},
        "Responder": {"threshold": 0.17}
        }

    # Choose the appropriate endpoint based on the pipeline
    endpoint = "rag-query" if args.rag else "extractive-query" if args.ex else None

    while True:
        query = input("=>\t")
        if query.lower() == "exit":
            break
        else: 
            try:
                # Prepare request body and endpoint
                query = query
                request_body = {
                    "query": query,
                    "params": query_params
                }
                # Send the request to the server
                response = requests.post(url=f"http://localhost:8001/{endpoint}", json=request_body)
                response.raise_for_status()  # Raise an HTTPError for bad responses
                json_response = response.json()

                print (json_response["answers"][0]["answer"])

            except requests.exceptions.RequestException as e:
                print(f"Request failed: {e}")
            except KeyError:
                print("Unexpected response structure:", json_response)
            

if __name__ == "__main__":
    main()
