# In this script we generate answers on Theano's training data for evaluation purposes using our QA pipelines. 

from typing import List, Union
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(SCRIPT_DIR, '../../')))

import torch
import gc

import argparse
import requests
from tqdm import tqdm
import json
import pandas as pd

def post_question_request (query:str, params:dict, endpoint:str):

    request_body = {
        "query": query,
        "params": params
    }
    response = requests.post(url= f"http://localhost:8000/{endpoint}", json=request_body)

    result = response.json()
    
    return result

def generate_results_for_queries_list (queries_intents_pairs:list,
                                       endpoint:str,
                                       params:dict, 
                                       save_dir:str,
                                       file_basename:str
                                      ):
    
    """Given a list of 'query \t intent_name' instances, run rag and/or exctractive pipeline using an http request and save results in files."""
    
    results = []

    for pair in tqdm (queries_intents_pairs):
        
        query, intent = pair
        print (query)

        try:
            result = post_question_request(query=query, params=params, endpoint=endpoint)
            result["intent"] = intent
            
            print (result ["answers"][0]["answer"])
            print ("\tc_rel:\t", result["answers"][0]["meta"]["context_relevance"])
            if  result["answers"][0]["type"] == "generative":
                print ("\tgroundedness:\t", result["answers"][0]["meta"]["groundedness"])

            results.append (result)

        except Exception as e:
            print(f"An unexpected error occurred: {str(e)}")
            
            with open ("errors.txt", "a") as error_file:
                answer_string = result ["answers"][0]["answer"]
                error_file.write(f"{query}\t{answer_string}\t{file_basename}\t{e}\n")
            continue

        # clean gpu memory cache
        torch.cuda.empty_cache()
        gc.collect()
    
    # save full results in json file
    with open(f"{save_dir}/{file_basename}_full.json", 'w', encoding='utf-8') as f:
        json.dump(obj=results, fp=f, ensure_ascii=False, indent=4)
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate results for queries using extractive or RAG pipelines.")
    
    # Argument to choose the type of pipeline to use
    parser.add_argument('--pipeline', type=str, required=True, choices=['extractive', 'rag'],
                        help="Specify the pipeline to run: 'extractive' or 'rag'.")

    # Arguments for input file with selected nlu queries form Theano
    parser.add_argument('--input_file', type=str, default='final_dataset.csv',
                        help='Path to the input file containing queries.')
    parser.add_argument("--params", type=str )
    parser.add_argument("--save_dir", type = str, default= "./final_results")
    parser.add_argument("--output_name", type=str)
    args = parser.parse_args()

    df = pd.read_csv(args.input_file)
    #df = df.iloc[164:173]
    queries = [str(query).strip() for query in df["query"].to_list()]
    intents = [str(intent).strip() for intent in df["intent_name"].to_list()]
    query_intent_pairs = list(zip(queries, intents))

    query_params = json.loads(args.params)

    if args.pipeline == 'extractive':
        endpoint = "extractive-query"
    else: 
        endpoint = "rag-query"

    # Post http request and get answer for queries
    generate_results_for_queries_list(
        queries_intents_pairs=query_intent_pairs,
        endpoint=endpoint,
        params=query_params,
        save_dir=args.save_dir,
        file_basename=args.output_name
        )
    
