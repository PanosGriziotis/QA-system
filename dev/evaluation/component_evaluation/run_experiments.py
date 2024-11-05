from typing import Union, List

import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(SCRIPT_DIR, '../../../')))

import json
import os
import argparse
import requests
from src.custom_components.ranker import SentenceTransformersRanker
from haystack.nodes import FARMReader, BM25Retriever, EmbeddingRetriever, DensePassageRetriever
from haystack.document_stores import ElasticsearchDocumentStore
from evaluate import evaluate_retriever_ranker_pipeline, evaluate_pipeline, evaluate_hybrid_retriever_ranker_pipeline
from helpers import load_and_save_npho_datasets, load_and_save_xquad_dataset
from haystack import Pipeline
from helpers import plot_retrievers_eval_report
import torch 
import gc

def evaluate_on_xquad(eval_type):
    load_and_save_xquad_dataset()
    if eval_type == "reader":
        run_readers_evaluation("datasets/xquad-el.json")
    elif eval_type == "retriever":
        run_retriever_evaluation("datasets/xquad-el.json")

 
def evaluate_on_npho(eval_type):
    load_and_save_npho_datasets()
    if eval_type == "reader":
        run_readers_evaluation(f"datasets/npho-covid-SQuAD-el_20.json")
    elif eval_type == "retriever":
        run_retriever_evaluation(f"datasets/npho-covid-SQuAD-el_20.json")

def evaluate_on_other_dataset(file_path, eval_type):
    if not os.path.isfile(file_path):
        print(f"File {file_path} does not exist.")
        return
    if eval_type == "reader":
        run_readers_evaluation(file_path)
    elif eval_type == "retriever":
        run_retriever_evaluation(file_path)

# Reader evaluation
def run_readers_evaluation(eval_filename, params:dict ={
            "Retriever": {"top_k": 20},
            "Ranker": {"top_k": 10},
            "Reader": {"top_k": 10}
            }
     ):
    
    """
    Run full extractive pipeline evaluation using different readers. The results contain both integrated and isolated evaluations.
    The parameters are the same across all initiated pipeline instances for better comparison between the reader models.
    """
    
    models = [
        "panosgriz/xlm-roberta-squad2-covid-el_small",
        "panosgriz/xlm-roberta-squad2-covid_el",
        "panosgriz/mdeberta-v3-base-squad2-covid-el",
        "panosgriz/mdeberta-v3-base-squad2-covid-el_small",
        "timpal0l/mdeberta-v3-base-squad2",
        "deepset/xlm-roberta-base-squad2"
    ]
    
    reports = {}
    requests.delete("http://localhost:9200/*")

    
    for model_name in models:
        
        #reports[reader.model_name_or_path] = evaluate_reader(reader, eval_filename, calibrate_scores=calibrate_scores)
        # construct pipeline

        ds = ElasticsearchDocumentStore(embedding_dim=384)
        retriever = EmbeddingRetriever(
            embedding_model="panosgriz/covid_el_paraphrase-multilingual-MiniLM-L12-v2",
            document_store=ds
        )

        ranker = SentenceTransformersRanker(
            model_name_or_path="amberoad/bert-multilingual-passage-reranking-msmarco",
            use_gpu=True
            )
        
        reader = FARMReader(model_name_or_path=model_name, devices=["cuda:0"])

        p = Pipeline()
        p.add_node(component=retriever, name ="Retriever", inputs=["Query"])
        p.add_node(component=ranker, name="Ranker", inputs=["Retriever"])
        p.add_node(component=reader, name="Reader", inputs=["Ranker"])
        
        eval_results = evaluate_pipeline(pipeline=p, document_store=ds, params=params, filename=eval_filename)
        
        reports[model_name] = eval_results
        # clean gpu
        torch.cuda.empty_cache()
        gc.collect()

    with open(f"reports_test/full_pipeline_readers_experiment_{os.path.basename(eval_filename).split('.')[0]}.json", "w") as fp:
        json.dump([reports], fp, ensure_ascii=False, indent=4)

# Retriever evaluation
# Modified retriever evaluation to include hybrid retriever
def run_retriever_evaluation(eval_filename, top_k_values=[x for x in range(1, 3)], hybrid_eval=True):
    """
    Evaluate Retrieval methods (Retriever + Ranker pipeline) on different top_k values.
    The results contain both Retriever and Ranker results on basic Document Retrieval metrics (Recall, MRR, MAP, NCDG).
    """
    
    results = {}
    ranker = SentenceTransformersRanker(model_name_or_path="amberoad/bert-multilingual-passage-reranking-msmarco")
    retriever_configs = {
        "bm25": {
            "document_store": {
                "embedding_dim": 384
            },
            "retriever_class": BM25Retriever,
            "retriever_args": {}
        },
        "emb_base": {
            "document_store": {
                "embedding_dim": 384
            },
            "retriever_class": EmbeddingRetriever,
            "retriever_args": {
                "embedding_model": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
            }
        },
        "embedding": {
            "document_store": {
                "embedding_dim": 384
            },
            "retriever_class": EmbeddingRetriever,
            "retriever_args": {
                "embedding_model": "panosgriz/covid_el_paraphrase-multilingual-MiniLM-L12-v2"
            }
        },
        "dpr": {
            "document_store": {
                "embedding_dim": 768
            },
            "retriever_class": DensePassageRetriever,
            "retriever_args": {
                "query_embedding_model": "panosgriz/bert-base-greek-covideldpr-query_encoder",
                "passage_embedding_model": "panosgriz/bert-base-greek-covideldpr-ctx_encoder",
                "max_seq_len_passage": 300
            }
       }
    }
    
    for retriever_type, config in retriever_configs.items():
        # we need to first delete indices from DS to initialize a new instance with different embedding dimensions.
        requests.delete("http://localhost:9200/*")

        ds = ElasticsearchDocumentStore(**config["document_store"])
        retriever = config["retriever_class"](document_store=ds, **config["retriever_args"])
        reports = evaluate_retriever_ranker_pipeline(
            retriever=retriever,
            ranker=ranker,
            document_store=ds,
            eval_filename=eval_filename,
            top_k_list=top_k_values
        )
        results[f"{retriever_type}"] = reports
    
    
    if hybrid_eval:
        results["hybrid"] = run_hybrid_retriever_eval(eval_filename, top_k_values=top_k_values)
    save_dir = os.path.join(SCRIPT_DIR, "retrievers_reports")
    os.makedirs(save_dir, exist_ok=True)
    output_file = f"{save_dir}/retrievers_eval_report_combined_{os.path.basename(eval_filename)}"
    with open(output_file, "a", encoding="utf-8") as fp:
        json.dump(results, fp, ensure_ascii=False, indent=4)
    
    plot_retrievers_eval_report(json_path=output_file,top_k_values=top_k_values)


def run_hybrid_retriever_eval(eval_filename, top_k_values=[x for x in range(1, 21)]):
    """
    Evaluate Hybrid Retrieval method (BM25 + Embedding Retriever) with a Ranker.
    Returns results as a dictionary.
    """
    requests.delete("http://localhost:9200/*")
    ds = ElasticsearchDocumentStore(embedding_dim=384)

    emb_retriever = EmbeddingRetriever(document_store=ds, embedding_model="panosgriz/covid_el_paraphrase-multilingual-MiniLM-L12-v2")
    bm25 = BM25Retriever(document_store=ds)
    results = evaluate_hybrid_retriever_ranker_pipeline(
        retrievers=[emb_retriever, bm25],
        document_store=ds,
        ranker=SentenceTransformersRanker(model_name_or_path="amberoad/bert-multilingual-passage-reranking-msmarco"),
        eval_filename=eval_filename,
        top_k_list=top_k_values
    )
    return results

def main():
    parser = argparse.ArgumentParser(description="Evaluation script")
    parser.add_argument("--evaluate", choices=['xquad', 'npho_test', 'other'], required=True,
                        help="Choose the dataset to evaluate on")
    parser.add_argument("--eval_type", choices=['reader', 'retriever'], required=True,
                        help="Choose the type of evaluation to run")
    parser.add_argument("--file_path", type=str, help="File path for 'other' dataset evaluation")

    args = parser.parse_args()

    if args.evaluate == "xquad":
        evaluate_on_xquad(args.eval_type)
    elif args.evaluate == "npho_test":
        evaluate_on_npho( args.eval_type)
    elif args.evaluate == "other":
        if args.file_path:
            evaluate_on_other_dataset(args.file_path, args.eval_type)
        else:
            print("Please provide a file path for 'other' dataset evaluation.")

if __name__ == "__main__":
    main()