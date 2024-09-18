import os
import sys
import json
from collections import defaultdict

import pandas as pd
from tqdm import tqdm

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(SCRIPT_DIR, '../../')))



def load_json_files(files):
    """Load all JSON files into a list."""
    contents = []
    for file in tqdm(files, desc="Loading JSON files"):
        with open(file, "r") as fp:
            contents.append(json.load(fp))
    return contents

def calculate_average(scores):
    """Calculate the average of non-None scores."""
    filtered_scores = [score for score in scores if score is not None]
    return sum(filtered_scores) / len(filtered_scores) if filtered_scores else None

def normalize_scores(scores, min_score, max_score):
    """Normalize scores to a 0-1 range."""
    if max_score - min_score == 0:
        return [0.0] * len(scores)
    return [(score - min_score) / (max_score - min_score) for score in scores]


def get_overall_report(files, output_file):
    """Generate an overall report from JSON files."""
    data = []

    # First loop to gather all context relevance scores for normalization
    for file_name in files:
        with open (file_name, "r") as fp:
            file_content = json.load(fp)

        # List to store context relevance scores for normalization
        context_relevance_scores = []
                
        # initialize a dictionary with scores fo each intent
        intent_scores = defaultdict(lambda: {"answer_accuracy":[], 'answer_relevance': [], 'context_relevance': [], 'groundedness': []})

        total_examples = 0
        unanswerable_examples = 0

        for result in file_content:

            total_examples += 1

            intent = result.get("intent", "unknown")
            answer_text = result["answers"][0]["answer"]

            if answer_text in ["Συγγνώμη, αλλά δεν μπορώ να δώσω μία απάντηση σε αυτήν την ερώτηση.", 
                           "Δεν διαθέτω αρκετές πληροφορίες για να απαντήσω αυτήν την ερώτηση"]:
                unanswerable_examples += 1
                continue
            
            answer_accuracy = result["answers"][0]["meta"].get("answer_accuracy")
            answer_relevance = result["answers"][0]["meta"].get("answer_relevance_ragas")
            context_relevance = result["answers"][0]["meta"].get("context_relevance")
            groundedness = result["answers"][0]["meta"].get("groundedness_score")

            if answer_accuracy is not None:
                intent_scores[intent]["answer_accuracy"].append(answer_accuracy)
            if context_relevance is not None:
                intent_scores[intent]['context_relevance'].append(context_relevance)
                context_relevance_scores.append(context_relevance)
            if answer_relevance is not None:
                intent_scores[intent]['answer_relevance'].append(answer_relevance)
            if groundedness is not None:
                intent_scores[intent]['groundedness'].append(groundedness)
        
        # Calculate min and max for normalization
        if context_relevance_scores:
            min_relevance = min(context_relevance_scores)
            max_relevance = max(context_relevance_scores)

        # Normalize and compute averages
        for intent, scores in intent_scores.items():
            if scores['context_relevance']:
                scores['context_relevance'] = normalize_scores(scores['context_relevance'], min_relevance, max_relevance)
            
            avg_answer_accuracy = calculate_average(scores["answer_accuracy"])
            avg_answer_relevance = calculate_average(scores['answer_relevance'])
            avg_context_relevance = calculate_average(scores['context_relevance'])
            avg_groundedness_score = calculate_average(scores['groundedness'])

            data.append({
                'file_name': os.path.basename(file_name),
                'intent': intent,
                "average_answer_accuracy": avg_answer_accuracy,
                'average_answer_relevance_ragas': avg_answer_relevance,
                'average_context_relevance': avg_context_relevance,
                'average_groundedness': avg_groundedness_score,
                'total_examples': total_examples,
                'unanswerable_examples': unanswerable_examples,
                'unanswerable_percentage': (unanswerable_examples / total_examples) * 100 if total_examples > 0 else 0
            })

    # Save DataFrame to CSV
    df = pd.DataFrame(data)
    
    df.to_csv(output_file, index=False)
    print(f"Results saved to {output_file}")

def generate_summary_csv(input_csv):
    """Generate a summary CSV from the overall report CSV."""
    df = pd.read_csv(input_csv)

    df['basename'] = df['file_name'].apply(lambda x: os.path.basename(x))
    df['file_type'] = df['basename'].apply(lambda x: 'extractive' if 'extractive' in x.lower() else 'rag')

    summary_df = df.groupby(['basename', 'file_type']).agg(
        avg_answer_accuracy = ("average_answer_accuracy", "mean"),
        avg_answer_relevance=('average_answer_relevance_ragas', 'mean'),
        avg_context_relevance=('average_context_relevance', 'mean'),
        avg_answer_groundedness=('average_groundedness', 'mean'),
        unanswerable_percentage=('unanswerable_percentage', "mean")
    ).reset_index()

    output_file = os.path.join (os.path.dirname(input_csv), "summary_scores_per_experiment.csv")
    summary_df.to_csv(output_file, index=False)
    print(f"Summary CSV saved to {output_file}")

if __name__ == "__main__":
    # Directory containing the JSON files
    import sys
    
    RESULTS_DIR = sys.argv[1]
    FILES = [os.path.join(RESULTS_DIR, f) for f in os.listdir(RESULTS_DIR) 
            if f.endswith(".json") and (f.startswith("extractive") or f.startswith("rag"))]
    output_file = os.path.join (RESULTS_DIR, "average_scores_per_exp_file.csv")

    get_overall_report(FILES, output_file=output_file)
    generate_summary_csv(input_csv=output_file)