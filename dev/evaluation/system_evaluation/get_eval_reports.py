import os
import sys
import json
import math
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import defaultdict
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

# Load JSON files
def load_json_file(file_path):
    """Load a single JSON file with results."""
    with open(file_path, "r") as fp:
        return json.load(fp)

# Calculate average and standard deviation
def calculate_average_and_std(scores):
    """Calculate the average and standard deviation of non-None scores."""
    filtered_scores = [score for score in scores if score is not None]
    if not filtered_scores:
        return None, None
    average = sum(filtered_scores) / len(filtered_scores)
    variance = sum((x - average) ** 2 for x in filtered_scores) / len(filtered_scores)
    std_dev = math.sqrt(variance)
    return average, std_dev

# Analyze scores and generate individual reports
def generate_individual_reports(file_path, output_dir):
    intent_report = []
    filename_report = []
    total_examples, unanswerable_examples = 0, 0

    file_content = load_json_file(file_path)
    intent_scores = defaultdict(lambda: {'answer_relevance': [], 'context_relevance': [], 'groundedness': []})
    total_answer_relevance, total_context_relevance, total_groundedness = [], [], []

    for result in file_content:
        total_examples += 1
        intent = result.get("intent", "unknown")
        answer_text = result["answers"][0]["answer"]

        if answer_text in ["Συγγνώμη, αλλά δεν μπορώ να δώσω μία απάντηση σε αυτήν την ερώτηση.", 
                           "Δεν διαθέτω αρκετές πληροφορίες για να απαντήσω αυτήν την ερώτηση."]:
            unanswerable_examples += 1
            continue

        answer_relevance = result["answers"][0]["meta"].get("answer_relevance")
        context_relevance = result.get("cr_score")
        groundedness = result["answers"][0]["meta"].get("groundedness")

        if context_relevance is not None:
            total_context_relevance.append(context_relevance)
            intent_scores[intent]['context_relevance'].append(context_relevance)
        if answer_relevance is not None:
            total_answer_relevance.append(answer_relevance)
            intent_scores[intent]['answer_relevance'].append(answer_relevance)
        if groundedness is not None:
            total_groundedness.append(groundedness)
            intent_scores[intent]['groundedness'].append(groundedness)

    # Intent-level report
    for intent, scores in intent_scores.items():
        avg_answer_relevance, std_answer_relevance = calculate_average_and_std(scores['answer_relevance'])
        avg_context_relevance, std_context_relevance = calculate_average_and_std(scores['context_relevance'])
        avg_groundedness_score, std_groundedness = calculate_average_and_std(scores['groundedness'])

        intent_report.append({
            'file_name': os.path.basename(file_path),
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

    # File-level report
    avg_answer_relevance, std_answer_relevance = calculate_average_and_std(total_answer_relevance)
    avg_context_relevance, std_context_relevance = calculate_average_and_std(total_context_relevance)
    avg_groundedness_score, std_groundedness = calculate_average_and_std(total_groundedness)

    filename_report.append({
        'file_name': os.path.basename(file_path),
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

    # Save intent-level and file-level reports as CSV
    intent_df = pd.DataFrame(intent_report)
    intent_df.to_csv(os.path.join(output_dir, f"{os.path.basename(file_path)}_intent_report.csv"), index=False)
    file_name_df = pd.DataFrame(filename_report)
    file_name_df.to_csv(os.path.join(output_dir, f"{os.path.basename(file_path)}_file_report.csv"), index=False)
    print(f"Reports for {os.path.basename(file_path)} saved to {output_dir}")

    return total_context_relevance, total_answer_relevance

# Plot distribution
def plot_distribution(scores, title, save_path, threshold):
    context_relevance = np.array(scores)
    plt.figure(figsize=(8, 6))
    plt.hist(context_relevance, bins=30, edgecolor='black')
    plt.axvline(threshold, color='red', linestyle='dashed', linewidth=1, label=f'Threshold')
    plt.title(title)
    plt.xlabel('Context Relevance')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()

# Plot correlation between context and answer relevance
def plot_correlation(context_relevance, answer_relevance, save_path):
    context_relevance = np.array(context_relevance)
    answer_relevance = np.array(answer_relevance)

    if len(context_relevance) > 0 and len(answer_relevance) > 0:
        correlation, _ = pearsonr(context_relevance, answer_relevance)
    else:
        correlation = 0

    plt.figure(figsize=(8, 6))
    plt.scatter(context_relevance, answer_relevance, alpha=0.7)
    plt.xlabel('Context Relevance')
    plt.ylabel('Answer Relevance')
    plt.title(f"Pearson Correlation: {correlation:.2f}")
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()

# Main function
if __name__ == "__main__":
    import sys
    RESULTS_DIR = sys.argv[1]
    OUTPUT_DIR = os.path.join(RESULTS_DIR, "reports")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Filter JSON files
    FILES = [os.path.join(RESULTS_DIR, f) for f in os.listdir(RESULTS_DIR) 
             if f.endswith(".json") and (f.startswith("extractive") or f.startswith("rag"))]

    # Process each file individually
    for file_path in FILES:
        total_context_relevance, total_answer_relevance = generate_individual_reports(file_path, OUTPUT_DIR)

        # Calculate threshold for distribution plot
        if total_context_relevance:
            threshold = np.percentile(total_context_relevance, 10)
            plot_distribution(
                total_context_relevance,
                f'{os.path.basename(file_path)} Context Relevance Distribution',
                os.path.join(OUTPUT_DIR, f"{os.path.basename(file_path)}_context_relevance_distribution.png"),
                threshold
            )

        # Plot correlation
        if total_context_relevance and total_answer_relevance:
            plot_correlation(
                total_context_relevance,
                total_answer_relevance,
                os.path.join(OUTPUT_DIR, f"{os.path.basename(file_path)}_context_answer_correlation.png")
            )
