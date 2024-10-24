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

def load_and_save_npho_datasets():
    npho_files = {
        '10': "datasets/npho-covid-SQuAD-el_10.json",
        '20': "datasets/npho-covid-SQuAD-el_20.json"
    }

    if all(check_file_exists(file) for file in npho_files.values()):
        print("npho dataset files already exist.")
        return

    # Load dataset from Hugging Face hub
    dataset = load_dataset("panosgriz/npho-covid-SQuAD-el")
    for key, local_file in npho_files.items():
        if key == "10":

            # Save relevant part of the dataset to local file
            with open(local_file, 'w', encoding='utf-8') as f:
                
                json.dump(dataset["test"][0],f, ensure_ascii=False, indent=4)
        else: 
                        # Save relevant part of the dataset to local file
            with open(local_file, 'w', encoding='utf-8') as f:
                json.dump(dataset["test"][1], f, ensure_ascii=False, indent=4)

    print("npho dataset files loaded and saved.")

def plot_retrievers_eval_report(json_path):
    # Check if the JSON file exists
    if not os.path.exists(json_path):
        print(f"File not found: {json_path}")
        return

    with open(json_path) as fp:
        data = json.load(fp)

    # Define methods and metrics
    methods = ['bm25','dpr', 'emb_base', 'embedding', "hybrid"]
    retriever_types = methods  # Methods already include _&_ranker suffix
    metrics = ["recall_single_hit", 'map', 'mrr', 'ndcg']  # Removed 'precision'
    plot_data = {metric: {method: [] for method in retriever_types} for metric in metrics}
    top_k_values = list(range(1, 21))  # top_k from 1 to 20

    for method in methods:
        for k in top_k_values:
            entry = data[method][str(k)]
            retriever_entry = entry['Retriever']
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
        'map': 'MAP@k',
        'mrr': 'MRR@k',
        'ndcg': 'NDCG@k'
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
        if metric in ['mrr', 'ndcg']:
            retriever_data, ranker_data = zip(*plot_data[metric]['hybrid'])
            plt.plot(top_k_values, retriever_data, label=f"{method_titles['hybrid']}", color=colors[methods.index('hybrid')])
            plt.plot(top_k_values, ranker_data, label=f"{method_titles['hybrid']} + Re-ranker", linestyle='--', color=colors[methods.index('hybrid')])
            
        else:
            for j, method in enumerate(methods):
                retriever_data, ranker_data = zip(*plot_data[metric][method])
                plt.plot(top_k_values, retriever_data, label=f"{method_titles[method]}", color=colors[j])
                if metric != 'recall_single_hit':
                    plt.plot(top_k_values, ranker_data, label=f"{method_titles[method]} + Re-ranker", linestyle='--', color=colors[j])
        
        plt.xlabel('k: number of retrieved documents')
        plt.ylabel(metric_titles[metric])
        plt.legend()

        # Set x-axis ticks to integers from 0 to 20, no decimals
        plt.xticks(range(0, 21, 2))
        plt.gca().xaxis.get_major_formatter().set_useOffset(False)

        # Derive the output file name from the input JSON file name
        base_name = os.path.splitext(os.path.basename(json_path))[0]
        output_path = f'reports/{base_name}_{metric}.png'
        
        # Save the plot
        plt.tight_layout()
        plt.savefig(output_path)
        print(f"Plot for {metric} saved as {output_path}")
        plt.close()  # Close the current figure to avoid overla