import os
import json
import sys
import os
import logging

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(SCRIPT_DIR, '../../../')))

from src.utils.metrics import compute_answer_relevance, compute_groundedness_rouge_score
from src.custom_components.generator import Generator
from tqdm import tqdm
import sys
import transformers

transformers.utils.logging.set_verbosity_error()
logging.basicConfig(level=logging.INFO)
#logger = logging.getLogger(__name__)
from transformers import logging as transformers_logging
# initialize generator for computing answer relevance scores
GENERATOR = Generator()

def process_json_files_in_folder(folder_path):


    # Iterate through all files in the folder
    for filename in tqdm(os.listdir(folder_path), desc=f"compute eval metrics for QA results"):

        if filename.endswith('.json') and 'rag' or "extractive" in os.path.splitext(filename)[0]:
            file_path = os.path.join(folder_path, filename)
            try:
                with open(file_path, 'r', encoding='utf-8') as file:
                    data = json.load(file)

                #logger.info(f"Processing file: {os.path.basename(file_path)}")
                
                # Compute evaluation metrics on results
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
    retrieved_documents = result["documents"]

    answer = result["answers"][0]
    answer_text = answer["answer"]
    answer_type = answer["type"]
    
    # Define context
    if answer_type == "extractive":
        context = answer["context"] # predifined reader's context window
    elif answer_type == "generative":
        retrieved_documents = [document["content"] for document in retrieved_documents]
        context = ' '.join(retrieved_documents)

    # Compute answer relevance score    
    if answer_type == "generative":
        answer_relevance_score = compute_answer_relevance(generator=GENERATOR, query=query, answer=answer_text)
    elif answer_type == "extractive":
        # define a helping context (length=context window) for extractive reader
        answer_relevance_score = compute_answer_relevance(generator=GENERATOR, query=query, answer=answer_text, context=context)
    
    answer["meta"]["answer_relevance"] = answer_relevance_score
    
    # Compute groundedness score for generative reader answers
    if answer_type == "generative":
        answer["meta"]["groundedness"]  = compute_groundedness_rouge_score(answer=answer_text, context=context)
    elif answer_type == "extractive":
        answer["meta"]["groundedness"] = None
        
    # Handle no answer strings
    if answer_text in ["Συγγνώμη, αλλά δεν μπορώ να δώσω μία απάντηση σε αυτήν την ερώτηση.", "Δεν διαθέτω αρκετές πληροφορίες για να απαντήσω αυτήν την ερώτηση."]:
        answer["meta"]["answer_relevance"] = None
        answer["meta"]["groundedness"] = None

    result["answers"][0] = answer

    return result

if __name__ == "__main__":
    folder_path = sys.argv[1]
    process_json_files_in_folder(folder_path)
