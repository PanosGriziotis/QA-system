import os
import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

def analyze_relevance(directory, percentile=10):
    scores = {'context_relevance': [], 'answer_relevance': []}
    #rag_scores = {'context_relevance': [], 'answer_relevance_ragas': []}

    # Iterate through all files in the directory
    for filename in os.listdir(directory):
        if filename.endswith('.json'):
            file_path = os.path.join(directory, filename)

            # Open and load the JSON file
            with open(file_path, 'r') as file:
                try:
                    data = json.load(file)
                    for item in data:
                        # Extract scores if the required keys exist
                        context_relevance = item['answers'][0]['meta'].get('context_relevance')
                        answer_relevance = item['answers'][0]['meta'].get('answer_relevance')

                        if context_relevance is not None and answer_relevance is not None:

                            scores['context_relevance'].append(context_relevance)
                            scores['answer_relevance_ragas'].append(answer_relevance)
                except json.JSONDecodeError:
                    print(f"Error decoding JSON in file: {filename}")

    # Calculate the percentile threshold
    def calculate_percentile_threshold(scores, percentile):
        context_relevance = np.array(scores['context_relevance'])
        if len(context_relevance) > 0:
            return np.percentile(context_relevance, percentile)
        return None

    threashold = calculate_percentile_threshold(scores, percentile)
    #rag_threshold = calculate_percentile_threshold(rag_scores, percentile)

    print(f"Scores {percentile}th Percentile Threshold: {threashold}")
    #print(f"RAG Scores {percentile}th Percentile Threshold: {rag_threshold}")

    # Plot distribution of context relevance scores
    def plot_distribution(scores, title, save_path, threashold):
        context_relevance = np.array(scores['context_relevance'])
        plt.figure(figsize=(8, 6))
        plt.hist(context_relevance, bins=30, edgecolor='black')
        plt.axvline(threashold, color='red', linestyle='dashed', linewidth=1, label=f'{percentile}th Percentile Threshold')
        plt.title(title)
        plt.xlabel('Context Relevance')
        plt.ylabel('Frequency')
        plt.legend()
        plt.grid(True)
        plt.savefig(save_path)
        plt.close()
    
    if "extractive" in filename:
        type = "Extractive"
    else:
        type = "RAG"
    plot_distribution(scores, f'{type} Scores Distribution', os.path.join(directory,'scores_distribution.png'), threashold=threashold)
    #plot_distribution(rag_scores, 'RAG Scores Distribution', 'rag_scores_distribution.png')

    return scores, threashold, type

def plot_and_analyze(scores, title, save_path):
    context_relevance = np.array(scores['context_relevance'])
    answer_relevance = np.array(scores['answer_relevance_ragas'])

    # Calculate correlation
    if len(context_relevance) > 0 and len(answer_relevance) > 0:
        correlation, _ = pearsonr(context_relevance, answer_relevance)
    else:
        correlation = 0

    # Plotting
    plt.figure(figsize=(8, 6))
    plt.scatter(context_relevance, answer_relevance, alpha=0.7)
    plt.title(f'{title}\nCorrelation: {correlation:.2f}')
    plt.xlabel('Context Relevance')
    plt.ylabel('Answer Relevance')
    plt.grid(True)

    # Save the plot to a file
    plt.savefig(save_path)
    plt.close()

    # Summary statistics
    print(f"{title} Summary Statistics:")
    print(f"Min Context Relevance: {np.min(context_relevance) if len(context_relevance) > 0 else 'N/A'}")
    print(f"Max Context Relevance: {np.max(context_relevance) if len(context_relevance) > 0 else 'N/A'}")
    print(f"Mean Context Relevance: {sum(context_relevance)/len(context_relevance) if len(context_relevance) > 0 else 'N/A'}")
    print(f"Min Answer Relevance: {np.min(answer_relevance) if len(answer_relevance) > 0 else 'N/A'}")
    print(f"Max Answer Relevance: {np.max(answer_relevance) if len(answer_relevance) > 0 else 'N/A'}")
    print(f"Mean Answer Relevance: {sum(answer_relevance)/len(context_relevance) if len(answer_relevance) > 0 else 'N/A'}")
    print(f"Pearson Correlation: {correlation:.2f}\n")

def modify_json_files(input_directory, output_directory, threashold=-1.5):
    # Create output directory if it doesn't exist
    os.makedirs(output_directory, exist_ok=True)

    # Process each file in the input directory
    for filename in os.listdir(input_directory):
        if filename.endswith('.json'):
            input_file_path = os.path.join(input_directory, filename)
            output_file_path = os.path.join(output_directory, filename)
            total_items = 0
            changed_items = 0

            # Open and load the JSON file
            with open(input_file_path, 'r', encoding='utf-8') as file:
                try:
                    data = json.load(file)  # Assume `data` is a list of dictionaries

                    # Iterate over each dictionary in the list
                    for item in data:
                        total_items += 1
                        context_relevance = item["answers"][0]["meta"].get('context_relevance')

                        if context_relevance is not None:
                            if context_relevance < threashold:
                                # Modify extractive type files if context_relevance is below the threshold
                                item["answers"][0]["answer"] = "Δεν διαθέτω αρκετές πληροφορίες για να απαντήσω αυτήν την ερώτηση"
                                item["answers"][0]["meta"]["answer_relevance_ragas"] = 0
                                item["answers"][0]["meta"]["answer_accuracy"] = 0
                                changed_items += 1

                    # Save the modified data to the output directory
                    with open(output_file_path, 'w', encoding='utf-8') as output_file:
                        json.dump(data, output_file, ensure_ascii=False, indent=4)

                    # Print the percentage of items changed
                    if total_items > 0:
                        percentage_changed = (changed_items / total_items) * 100
                        print(f"File: {filename}")
                        print(f"Total Items: {total_items}")
                        print(f"Changed Items: {changed_items}")
                        print(f"Percentage Changed: {percentage_changed:.2f}%")
                        print()

                except json.JSONDecodeError:
                    print(f"Error decoding JSON in file: {filename}")

if __name__ == "__main__":
    import sys
    
    directory = sys.argv[1] 
    percentile = 10  # Choose the percentile here
    scores, threashold, type = analyze_relevance(directory, percentile)
    plot_and_analyze(scores, f'{type} Scores Analysis', os.path.join(directory,'scores_plot.png'))
    modify_json_files(directory, os.path.join(directory,'final_results_threashold'),threashold)
