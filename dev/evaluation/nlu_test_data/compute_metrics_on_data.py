import os
import json
import sys
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(SCRIPT_DIR, '../../../')))

from src.utils.metrics import compute_answer_relevance, compute_context_relevance, compute_groundedness_rouge_score, compute_similarity
from tqdm import tqdm
import sys

def process_json_files_in_folder(folder_path):
    # Iterate through all files in the folder
    for filename in tqdm(os.listdir(folder_path)):

        if filename.endswith('.json') and 'rag' or "extractive" in os.path.splitext(filename)[0]:
            file_path = os.path.join(folder_path, filename)

            # Open and read the JSON file
            try:
                with open(file_path, 'r', encoding='utf-8') as file:
                    data = json.load(file)

                print(f"Processing file: {os.path.basename(file_path)}")
                updated_data = [add_eval_scores_to_result(d) for d in data]

                # Write the updated data back to the file
                with open(file_path, 'w', encoding='utf-8') as fp:
                    json.dump(updated_data, fp, ensure_ascii=False, indent=4)

            except json.JSONDecodeError as e:
                print(f"Error decoding JSON from file {file_path}: {e}")
            except Exception as e:
                print(f"An error occurred while processing file {file_path}: {e}")

def add_eval_scores_to_result(result):

    query = result["query"]
    answer_objs = result["answers"]
    retrieved_documents = result["documents"]

    first_answer = answer_objs[0]
    answer_text = first_answer["answer"]
    answer_type = first_answer["type"]
    
    # Define context
    if answer_type == "extractive":
        context = first_answer["context"]
    elif answer_type == "generative":
        retrieved_documents = [document["content"] for document in retrieved_documents]
        context = ' '.join(retrieved_documents)
    
    # Context relevance
    if "context_relevance" not in first_answer["meta"]:
        print ("\n===========Computing context relevance==========\n")
        context_relevance = compute_similarity(document_1=answer_text, document_2=retrieved_documents)
        first_answer["meta"]["context_relevance"] = context_relevance
    

    #if "answer_relevance" not in first_answer["meta"] or first_answer["meta"]["answer_relevance"]==0.0:    
    print ("\n===========Computing answer relevance==========\n")
    if answer_type == "generative":
        helping_context = answer_text
    elif answer_text == "extractive":
        helping_context = context
    answer_relevance_ragas = compute_answer_relevance(query=query, answer=answer_text, context=helping_context)
    first_answer["meta"]["answer_relevance"] = answer_relevance_ragas

    if answer_type == "generative" and "groundedness" not in first_answer["meta"]:
        print ("\n===========Computing groundedness==========\n")
        first_answer["meta"]["groundedness"]  = compute_groundedness_rouge_score(
            answer=answer_text,
            context=" ".join(retrieved_documents)
        )
    elif answer_type == "extractive":
        first_answer["meta"]["groundedness"] = None
        
        # Handle no answer strings
    if answer_text == "Συγγνώμη, αλλά δεν μπορώ να δώσω μία απάντηση σε αυτήν την ερώτηση.":
        first_answer["meta"]["answer_relevance"] = 0.0
        first_answer["meta"]["groundedness"] = 0.0 if answer_type == "rag" else None 



    result["answers"][0] = first_answer

    return result    
if __name__ == "__main__":
    folder_path = sys.argv[1]
    process_json_files_in_folder(folder_path)
