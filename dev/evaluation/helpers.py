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
        # Save relevant part of the dataset to local file
        with open(local_file, 'w', encoding='utf-8') as f:
            f.write(dataset[f"test_npho_{key}tok"].to_json())

    print("npho dataset files loaded and saved.")

def plot_retrievers_eval_report(json_path):
    # Check if the JSON file exists
    if not os.path.exists(json_path):
        print(f"File not found: {json_path}")
        return

    with open(json_path) as fp:
        data = json.load(fp)

    # Define methods and metrics
    methods = ['bm25', 'embedding_retriever', 'dpr']
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

    # Create the plots
    sns.set_theme(style="whitegrid")

    # Define distinct colors for the lines
    colors = sns.color_palette('bright', n_colors=len(methods))

    fig, axes = plt.subplots(4, 1, figsize=(12, 24))  # Adjust the grid to fit 4 metrics
    fig.suptitle('Performance Metrics for Retrieval Methods', fontsize=16)

    metric_titles = {
        'recall_single_hit': 'Recall@k',
        'map': 'MAP@k',
        'mrr': 'MRR@k',
        'ndcg': 'NDCG@k'
    }

    method_titles = {
        'bm25': 'BM25',
        'embedding_retriever': 'SBERT',
        'dpr': 'DPR'
    }

    for i, metric in enumerate(metrics):
        ax = axes[i]
        for j, method in enumerate(methods):
            retriever_data, ranker_data = zip(*plot_data[metric][method])
            ax.plot(top_k_values, retriever_data, label=f"{method_titles[method]}", color=colors[j])
            if metric != 'recall_single_hit':
                ax.plot(top_k_values, ranker_data, label=f"{method_titles[method]} & Re-ranker", linestyle='--', color=colors[j])
        ax.set_title(metric_titles[metric])
        ax.set_xlabel('k: number of retrieved documents')
        ax.set_ylabel(metric_titles[metric])
        ax.legend()

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    # Derive the output file name from the input JSON file name
    base_name = os.path.splitext(os.path.basename(json_path))[0]
    output_path = f'reports/{base_name}_performance_metrics.png'
    plt.savefig(output_path)
    print(f"Combined plot saved as {output_path}")

    