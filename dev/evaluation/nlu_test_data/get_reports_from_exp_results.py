import os
import sys
import json
from collections import defaultdict

import pandas as pd
from tqdm import tqdm
import math 

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(SCRIPT_DIR, '../../')))



def load_json_files(files):
    """Load all JSON files into a list."""
    contents = []
    for file in tqdm(files, desc="Loading JSON files"):
        with open(file, "r") as fp:
            contents.append(json.load(fp))
    return contents


def calculate_average_and_std(scores):
    """Calculate the average and standard deviation of non-None scores."""
    # Filter out None values
    filtered_scores = [score for score in scores if score is not None]

    # Return None if no valid scores are left after filtering
    if not filtered_scores:
        return None, None

    # Calculate the average
    average = sum(filtered_scores) / len(filtered_scores)

    # Calculate the standard deviation (population)
    variance = sum((x - average) ** 2 for x in filtered_scores) / len(filtered_scores)
    std_dev = math.sqrt(variance)

    return average, std_dev

def get_results_reports(files, output_dir):
    """Generate an overall report from JSON files."""
    intent_report = []
    filename_report = []

    # First loop to gather all context relevance scores for normalization
    for file_name in files:
        with open (file_name, "r") as fp:
            file_content = json.load(fp)
    
        # initialize a dictionary with scores fo each intent
        intent_scores = defaultdict(lambda: {'answer_relevance': [], 'context_relevance': [], 'groundedness': []})
        
        total_examples = 0
        unanswerable_examples = 0

        total_answer_relevance = []
        total_context_relevance = []
        total_groundedness = []
        
        for result in file_content:

            total_examples += 1

            intent = result.get("intent", "unknown")
            answer_text = result["answers"][0]["answer"]

            if answer_text in ["Συγγνώμη, αλλά δεν μπορώ να δώσω μία απάντηση σε αυτήν την ερώτηση.", 
                           "Δεν διαθέτω αρκετές πληροφορίες για να απαντήσω αυτήν την ερώτηση"]:
                unanswerable_examples += 1
                continue
            
            answer_relevance = result["answers"][0]["meta"].get("answer_relevance")
            context_relevance = result["answers"][0]["meta"].get("context_relevance")
            groundedness = result["answers"][0]["meta"].get("groundedness")


            if context_relevance is not None:
                total_context_relevance.append (context_relevance)
                intent_scores[intent]['context_relevance'].append(context_relevance)
            if answer_relevance is not None:
                total_answer_relevance.append(answer_relevance)
                intent_scores[intent]['answer_relevance'].append(answer_relevance)
            if groundedness is not None:
                total_groundedness.append(groundedness)
                intent_scores[intent]['groundedness'].append(groundedness)
        


        for intent, scores in intent_scores.items():

            avg_answer_relevance, std_answer_relevance = calculate_average_and_std(scores['answer_relevance'])
            avg_context_relevance, std_context_relevance = calculate_average_and_std(scores['context_relevance'])
            avg_groundedness_score, std_groundedness = calculate_average_and_std(scores['groundedness'])

            intent_report.append({
                'file_name': os.path.basename(file_name),
                'intent': intent,
                'average_answer_relevance': avg_answer_relevance,
                'average_context_relevance': avg_context_relevance,
                'average_groundedness': avg_groundedness_score,
                'std_answer_relevance': std_answer_relevance,
                'std_context_relevance': std_context_relevance,
                'std_groundedness': std_groundedness,
                'total_examples': total_examples,
                'unanswerable_examples': unanswerable_examples,
                'unanswerable_percentage': (unanswerable_examples / total_examples) * 100 if total_examples > 0 else 0
            })
        
        avg_answer_relevance, std_answer_relevance = calculate_average_and_std(total_answer_relevance)
        avg_context_relevance, std_context_relevance = calculate_average_and_std(total_context_relevance)
        avg_groundedness_score, std_groundedness = calculate_average_and_std(total_groundedness)

        filename_report.append({
        'file_name': os.path.basename(file_name),
        'average_answer_relevance': avg_answer_relevance,
        'average_context_relevance': avg_context_relevance,
        'average_groundedness': avg_groundedness_score,
        'std_answer_relevance': std_answer_relevance,
        'std_context_relevance': std_context_relevance,
        'std_groundedness': std_groundedness,
        'total_examples': total_examples,
        'unanswerable_examples': unanswerable_examples,
        'unanswerable_percentage': (unanswerable_examples / total_examples) * 100 if total_examples > 0 else 0
    })
# Save DataFrame to CSV
    intent_df = pd.DataFrame(intent_report)
    intent_df.to_csv(os.path.join(output_dir, "results_report_per_intent.csv"), index=False)
    file_name_df = pd.DataFrame(filename_report)
    file_name_df.to_csv(os.path.join(output_dir, "results_report.csv"), index=False)

    print(f"Results saved to {output_dir}")

if __name__ == "__main__":
    # Directory containing the JSON files
    import sys
    
    RESULTS_DIR = sys.argv[1]
    FILES = [os.path.join(RESULTS_DIR, f) for f in os.listdir(RESULTS_DIR) 
            if f.endswith(".json") and (f.startswith("extractive") or f.startswith("rag"))]
    
    get_results_reports(files= FILES, output_dir= RESULTS_DIR)