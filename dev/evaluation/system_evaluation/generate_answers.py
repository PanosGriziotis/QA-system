# In this script we generate answers on Theano's training data for evaluation purposes using our QA pipelines. 

from typing import List, Union
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(SCRIPT_DIR, '../../../')))

import logging
import argparse
import requests
from tqdm import tqdm
import json
import pandas as pd
from src.utils.data_handling import flush_cuda_memory

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def post_question_request (query:str, params:dict, endpoint:str):
    """Send post request to query pipeline endpoint"""

    try:
        request_body = {
            "query": query,
            "params": params
        }
        response = requests.post(url=f"http://localhost:8000/{endpoint}", json=request_body)
        response.raise_for_status()  
        json_response = response.json()
    
    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")
    except KeyError:
        print("Unexpected response structure:", json_response)
    
    return json_response

def generate_results_for_queries_list (queries_intents_pairs:list, endpoint:str, params:dict, save_dir:str, file_basename:str):
    """Given a list of 'query-intent_name' pairs, runs qa pipeline using an http request and saves full results output in json files."""
    
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir, exist_ok=True)
        
    results = []

    for pair in tqdm (queries_intents_pairs, desc="Generating responses for given queries..."):       
        query, intent = pair
        #print (query)    
        try:
            result = post_question_request(query=query, params=params, endpoint=endpoint)
            result["intent"] = intent
            #print (result ["answers"][0]["answer"])
            results.append (result)
        except Exception as e:
            print(f"An unexpected error occurred: {str(e)}")
            # write any unexpected errors on log file
            with open ("errors.txt", "a") as error_file:
                answer_string = result ["answers"][0]["answer"]
                error_file.write(f"{query}\t{answer_string}\t{file_basename}\t{e}\n")
            continue
        flush_cuda_memory()
    
    # save results in json file
    with open(f"{save_dir}/{file_basename}_full.json", 'w', encoding='utf-8') as f:
        json.dump(obj=results, fp=f, ensure_ascii=False, indent=4)

def main():
    parser = argparse.ArgumentParser(description="Generate results for queries using extractive or RAG pipelines.")
    parser.add_argument('--pipeline', type=str, required=True, choices=['extractive', 'rag'],help="Specify the query pipeline to run: 'extractive' or 'rag'.")
    parser.add_argument('--input_file', type=str, default='test_data/final_dataset.csv',help='Path to the input file containing queries.')
    parser.add_argument("--params", type=str, default='{}', help="JSON string of query parameters.")
    parser.add_argument("--save_dir", type = str, default= os.path.join(SCRIPT_DIR, "results"))
    parser.add_argument("--output_name", type=str)
    args = parser.parse_args()
    
    df = pd.read_csv(args.input_file)
    queries = [str(query).strip() for query in df["query"].to_list()]
    intents = [str(intent).strip() for intent in df["intent_name"].to_list()]
    query_intent_pairs = list(zip(queries, intents))
    query_params = json.loads(args.params)
    if args.pipeline == 'extractive':
        endpoint = "extractive-query"
    else: 
        endpoint = "rag-query"

    generate_results_for_queries_list(
        queries_intents_pairs=query_intent_pairs,
        endpoint=endpoint,
        params=query_params,
        save_dir=args.save_dir,
        file_basename=args.output_name
        )

if __name__ == "__main__":
    main()
    
