import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
from datasets import load_dataset
import requests

def check_file_exists(file_path):
    return os.path.isfile(file_path)

def load_and_save_xquad_dataset():
    url = "https://github.com/google-deepmind/xquad/raw/master/xquad.el.json"
    save_to = "datasets/xquad-el.json"
    save_dir = os.path.dirname(save_to)  # Extract directory path
    os.makedirs(save_dir, exist_ok=True)
    
    if os.path.exists(save_to):
        print(f"XQuAD dataset already exists at {save_to}.")
        return

    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        with open(save_to, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        print(f"File downloaded and saved to {save_to}")
    else:
        print(f"Failed to download file. Status code: {response.status_code}")

def check_file_exists(file_path):
    return os.path.isfile(file_path)

def load_and_save_npho_datasets(): 
    # Define the path for the save file


    # Ensure the 'datasets' directory exists
    
    save_file = "datasets/npho-covid-SQuAD-el_20.json"
    save_dir = os.path.dirname(save_file)  # Extract directory path
    os.makedirs(save_dir, exist_ok=True)
    # Check if the file already exists
    if check_file_exists(save_file):
        print("npho dataset files already exist.")
        return
    
    # Load dataset from Hugging Face hub
    dataset = load_dataset("panosgriz/npho-covid-SQuAD-el")

    # Convert to dictionary and format to ensure we don’t have extra nesting
    test_data = dataset["test"]

    # Process to ensure 'data' is the top-level key with a list of 'paragraphs'
    formatted_data = {"data": [{"paragraphs": [item]} for item in test_data]}

    # Save the formatted data to JSON
    with open(save_file, "w", encoding="utf-8") as f:
        json.dump(formatted_data, f, ensure_ascii=False, indent=4)

    print("npho dataset files loaded and saved.")
    
def plot_retrievers_eval_report(json_path,top_k_values=list(range(1, 21))):
    """Plot evaluation results for document retrieval methods"""

    if not os.path.exists(json_path):
        print(f"File not found: {json_path}")
        return
    
    dirname = os.path.dirname(json_path)
    with open(json_path) as fp:
        data = json.load(fp)

    # Define methods and metrics
    methods = ['bm25','dpr', 'emb_base', "hybrid"]
    retriever_types = methods  
    metrics = ["recall_single_hit", 'mrr', "ndcg", "map"] 
    plot_data = {metric: {method: [] for method in retriever_types} for metric in metrics}


    for method in methods:
        for k in top_k_values:
            entry = data[method][str(k)]
            retriever_entry = entry['JoinDocuments'] if method == "hybrid" else entry['Retriever']
            ranker_entry = entry['Ranker']
            
            for metric in metrics:
                if metric == 'recall_single_hit':
                    plot_data[metric][method].append((retriever_entry[metric], None))
                else:
                    plot_data[metric][method].append((retriever_entry[metric], ranker_entry[metric]))

    # Create individual plots for each metric
    sns.set_theme(style="whitegrid")

    # Define distinct colors for the lines
    colors = sns.color_palette('bright', n_colors=len(methods))

    metric_titles = {
        'recall_single_hit': 'Recall@k',
        'mrr': 'MRR@k',
        'map': 'MAP@k',
        'ndcg': 'NCDG@k'
    }

    method_titles = {
        'bm25': 'BM25',
        'emb_base': 'SBERT',
        'embedding': 'SBERT_adapted',
        'dpr': 'DPR',
        'hybrid': 'BM25 + SBERT_adapted'
    }

    # Iterate over metrics and create a separate plot for each
    for metric in metrics:
        plt.figure(figsize=(12, 6))  # Set figure size for each plot
        
        # Plot only the "hybrid" method for 'ndcg' and 'mrr' metrics
        if metric in ['mrr', "map", "ndcg"]:
            retriever_data, ranker_data = zip(*plot_data[metric]['hybrid'])
            plt.plot(top_k_values, retriever_data, label=f"{method_titles['hybrid']}", color=colors[methods.index('hybrid')])
            plt.plot(top_k_values, ranker_data, label=f"{method_titles['hybrid']} + Reranker", linestyle='--', color=colors[methods.index('hybrid')])
            
        else:
            for j, method in enumerate(methods):
                retriever_data, ranker_data = zip(*plot_data[metric][method])
                plt.plot(top_k_values, retriever_data, label=f"{method_titles[method]}", color=colors[j])
                if metric != 'recall_single_hit':
                    plt.plot(top_k_values, ranker_data, label=f"{method_titles[method]} + Reranker", linestyle='--', color=colors[j])
        
        plt.xlabel('k: number of retrieved documents', fontsize=16)
        plt.ylabel(metric_titles[metric], fontsize=16)
        plt.legend(fontsize=16)

        # Set x-axis ticks to integers from 0 to 20, no decimals
        x_axis_end = top_k_values[-1] +1 
        plt.xticks(range(0, x_axis_end, 2), fontsize=16)
        plt.yticks(fontsize=16)
        plt.gca().xaxis.get_major_formatter().set_useOffset(False)

        # Derive the output file name from the input JSON file name
        base_name = os.path.splitext(os.path.basename(json_path))[0]
        output_path = f'{dirname}/{base_name}_{metric}.png'
        
        # Save the plot
        plt.tight_layout()
        plt.savefig(output_path)
        print(f"Plot for {metric} saved as {output_path}")
        plt.close()  # Close the current figure to avoid overla