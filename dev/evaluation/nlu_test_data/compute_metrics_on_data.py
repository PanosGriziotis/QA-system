import os
import json
import sys
import os
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(SCRIPT_DIR, '../../../../')))
from src.utils.data_handling import add_eval_scores_to_result
from src.utils.metrics import compute_answer_relevance, compute_similarity
from tqdm import tqdm
import sys

def process_json_files_in_folder(file_path):
    # Iterate through all files in the folder
#for filename in tqdm(os.listdir(folder_path)):
    # Check if the file is a JSON file and contains the word "extractive" in its basename
    #if filename.endswith('.json') and 'rag' in os.path.splitext(filename)[0]:
    #file_path = os.path.join(folder_path, filename)

    # Open and read the JSON file
    with open(file_path, 'r', encoding='utf-8') as file:
        try:
            data = json.load(file)
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON from file {file_path}: {e}")
    
    print(f"Processing file: {os.path.basename(file_path)}")
    updated_data = [add_eval_scores_to_result(d) for d in data]
    
    with open(file_path, 'w', encoding='utf-8') as fp:
        json.dump(updated_data, fp, ensure_ascii=False, indent=4)

import pandas as pd

def evaluate_responses(dataset, output_file):
    # Read the CSV file into a DataFrame
    df = pd.read_csv(dataset)

    # Initialize lists to store the computed scores
    relevance_scores = []
    accuracy_scores = []
    
    # Iterate over each row and compute the scores
    for query, answer in zip(df["query"], df["response"]):
        # Compute scores
        intent_name = df.loc[df["query"] == query, "intent_name"].values[0]
        if intent_name in ["out_of_scope/general", "out_of_scope/medicine"]:
            answer_accuracy_score = None
            answer_relevance_score = None
        else:
            
            answer_relevance_score = compute_answer_relevance(query=query, answer=answer, context=answer)
            answer_accuracy_score = compute_similarity(query=query, document=answer)
        
        # Append scores to the lists
        relevance_scores.append(answer_relevance_score)
        accuracy_scores.append(answer_accuracy_score)
    
    # Add the computed scores as new columns to the DataFrame
    df["answer_relevance_score"] = relevance_scores
    df["answer_accuracy_score"] = accuracy_scores
    
    # Save the updated DataFrame back to the CSV file
    
    df.to_csv(output_file, index=True, index_label="id")
    
if __name__ == "__main__":
    file_path = sys.argv[1]
    process_json_files_in_folder(file_path)
