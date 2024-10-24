from typing import Union, List

import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(SCRIPT_DIR, '../../')))

import json
import os
import argparse
import requests
from src.pipelines.ranker import SentenceTransformersRanker
from haystack.nodes import FARMReader, BM25Retriever, EmbeddingRetriever, DensePassageRetriever
from haystack.document_stores import ElasticsearchDocumentStore
from evaluate import evaluate_retriever_ranker_pipeline, evaluate_pipeline, evaluate_hybrid_retriever_ranker_pipeline
from helpers import load_and_save_npho_datasets, load_and_save_xquad_dataset
from haystack import Pipeline
import mlflow
import torch 
import gc
from haystack.schema import Document
from src.utils.metrics import compute_similarity
import pandas as pd
import ast
def evaluate_on_xquad(eval_type):
    load_and_save_xquad_dataset()
    if eval_type == "reader":
        run_readers_evaluation("datasets/xquad-el.json")
    elif eval_type == "retriever":
        run_retriever_evaluation("datasets/xquad-el.json")
    elif eval_type == "hybrid_retriever":
        run_hybrid_retriever_eval("datasets/xquad-el.json")
    elif eval_type == "full_pipeline":
        eval_full_pipeline(eval_filename="datasets/xquad-el.json", top_k_retrievers=[10,20], top_k_rankers= [5,10], top_k_readers=[5,10])

def evaluate_on_npho(version, eval_type):
    load_and_save_npho_datasets()
    if eval_type == "reader":
        run_readers_evaluation(f"datasets/npho-covid-SQuAD-el_{version}.json")
    elif eval_type == "retriever":
        run_retriever_evaluation(f"datasets/npho-covid-SQuAD-el_{version}.json")
    elif eval_type == "generator":
        evaluate_generator(f"datasets/npho-covid-SQuAD-el_{version}.json", output_dir="./reports")
    elif eval_type == "hybrid_retriever":
        run_hybrid_retriever_eval(f"datasets/npho-covid-SQuAD-el_{version}.json")
    elif eval_type == "full_pipeline":
        eval_full_pipeline(eval_filename=f"datasets/npho-covid-SQuAD-el_{version}.json", top_k_retrievers=[10,20], top_k_rankers= [5,10], top_k_readers=[5,10])

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

    with open(f"reports/full_pipeline_readers_experiment_{os.path.basename(eval_filename).split('.')[0]}.json", "w") as fp:
        json.dump([reports], fp, ensure_ascii=False, indent=4)

# Retriever evaluation
def run_retriever_evaluation(eval_filename, top_k_values = [x for x in range(1, 21)], hybrid_eval=True):
    """
    Evaluate Retrieval methods (Retriever + Ranker pipeline) on different top_k values.
    The results contain both Retriever and Ranker results on basic Document Retrieval metrics (Recall, MRR, MAP, NCDG).
    """
    
    results = {}
    ranker = SentenceTransformersRanker(model_name_or_path="amberoad/bert-multilingual-passage-reranking-msmarco")
    retriever_configs = {
        #"bm25": {
         #   "document_store": {
          #      "embedding_dim": 384
           # },
            #"retriever_class": BM25Retriever,
            #"retriever_args": {}
        #},
        "emb_base": {
            "document_store": {
                "embedding_dim": 384
            },
            "retriever_class": EmbeddingRetriever,
            "retriever_args": {
                "embedding_model": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
            }
        }
       # "dpr": {
        #    "document_store": {
         #       "embedding_dim": 768
          #  },
           # "retriever_class": DensePassageRetriever,
            #"retriever_args": {
             #   "query_embedding_model": "panosgriz/bert-base-greek-covideldpr-query_encoder",
              #  "passage_embedding_model": "panosgriz/bert-base-greek-covideldpr-ctx_encoder",
               # "max_seq_len_passage": 300
      #      }
       # }
    }
    
    for retriever_type, config in retriever_configs.items():
        # we need to first delete indeces from DS to be able to initialize a new instance with different embedding dimensions.
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
    
    output_file = f"reports/retrievers_eval_report_2_{os.path.basename(eval_filename)}"
    with open(output_file, "a", encoding="utf-8") as fp:
        json.dump(results, fp, ensure_ascii=False, indent=4)

def run_hybrid_retriever_eval(eval_filename, top_k_values = [x for x in range(1, 21)]):
    
    requests.delete("http://localhost:9200/*")
    ds = ElasticsearchDocumentStore(embedding_dim=384)

    emb_retriever = EmbeddingRetriever(document_store=ds, embedding_model="panosgriz/covid_el_paraphrase-multilingual-MiniLM-L12-v2")
    bm25 = BM25Retriever(document_store=ds)
    results = evaluate_hybrid_retriever_ranker_pipeline(retrievers=[emb_retriever,bm25], document_store=ds, ranker=SentenceTransformersRanker(model_name_or_path="amberoad/bert-multilingual-passage-reranking-msmarco"), eval_filename=eval_filename, top_k_list=top_k_values)
    output_file = f"reports/retrievers_hybrid_eval_report_{os.path.basename(eval_filename)}"
    with open(output_file, "a", encoding="utf-8") as fp:
        json.dump(results, fp, ensure_ascii=False, indent=4)
    
def evaluate_generator (evaluation_file, output_dir):
    max_new_tokens = [100]

    from src.pipelines.query_pipelines import Generator

    # get ground truth data from file
    with open (evaluation_file, "r") as fp:
        test_data = json.load(fp)["data"]
    
    generator = Generator()

    summary_results = {"max_new_tokens": [], "avg_sas_score": [], "unanswerable_queries": []}
    
    for max_new_token in max_new_tokens:
        
        unanswerable_queries = 0
        total_num_examples = 0
        eval_results = {"query": [], "context": [], "answer":[], "generated_answer":[], "sas_score":[]}
        
        for example in test_data:
            total_num_examples += 1
            example = example["paragraphs"][0]
            context = example['context']
            for qa_pair in example["qas"]:
                query = qa_pair["question"]
                answer = qa_pair["answers"][0]["text"]
        
        
                result, _ = generator.run(query=query, documents=[Document(content=context)], max_new_tokens=max_new_token, post_processing=True)
                generated_answer = result["answers"][0].answer
                
                if generated_answer == "Συγγνώμη, αλλά δεν μπορώ να δώσω μία απάντηση σε αυτήν την ερώτηση.":
                    sas_score = 0.0
                    unanswerable_queries += 1
                else:
                    sas_score = compute_similarity(document_1=answer, document_2=generated_answer)

                eval_results["query"].append(query)
                eval_results["context"].append(context)
                eval_results["answer"].append(answer)
                eval_results["generated_answer"].append(generated_answer)
                eval_results["sas_score"].append(sas_score)

        
        df = pd.DataFrame(eval_results)
        df.to_csv(os.path.join(output_dir, f"generator_eval_{max_new_token}_2.csv"))

        unanswerable_percentage = (unanswerable_queries/total_num_examples)*100
        avg_score = sum(eval_results["sas_score"]) / len(eval_results["sas_score"])
        summary_results["max_new_tokens"].append(max_new_token)
        summary_results["avg_sas_score"].append(avg_score)
        summary_results["unanswerable_queries"].append(unanswerable_percentage)

    df = pd.DataFrame(summary_results)
    df.to_csv(os.path.join (output_dir, "summary_generator_eval_results_2.csv"))

def eval_full_pipeline(
        pipeline:Pipeline,
        eval_filename: str,
        top_k_retrievers: Union[List[int], int],
        top_k_rankers: Union[List[int], int],
        top_k_readers: Union[List[int], int]):
    
    requests.delete("http://localhost:9200/*")
    #from src.pipelines.extractive_qa_pipeline import extractive_qa_pipeline
    
    # Ensure the inputs are lists
    if isinstance(top_k_retrievers, int):
        top_k_retrievers = [top_k_retrievers]
    if isinstance(top_k_rankers, int):
        top_k_rankers = [top_k_rankers]
    if isinstance(top_k_readers, int):
        top_k_readers = [top_k_readers]
    
    results = []
    for retriever_k in top_k_retrievers:
        for ranker_k in top_k_rankers:
            for reader_k in top_k_readers:
                result = evaluate_pipeline(
                    pipeline=pipeline,
                    document_store=ElasticsearchDocumentStore(embedding_dim=384),
                    params={
                        "Retriever": {"top_k": retriever_k},
                        "Ranker": {"top_k": ranker_k},
                        "Reader": {"top_k": reader_k}
                    },
                    filename=eval_filename
                )
                
                result_entry = {
                    "top_k_retriever": retriever_k,
                    "top_k_ranker": ranker_k,
                    "top_k_reader": reader_k,
                    "results": result
                }
                results.append(result_entry)
    
    report_filename = f"reports/extractive_pipeline_full_eval_{os.path.basename(eval_filename).split('.')[0]}.json"
    with open(report_filename, "w") as fp:
        json.dump(results, fp, ensure_ascii=False, indent=4)


def main():
    parser = argparse.ArgumentParser(description="Evaluation script")
    parser.add_argument("--evaluate", choices=['xquad', 'npho_10', 'npho_20', 'other'], required=True,
                        help="Choose the dataset to evaluate on")
    parser.add_argument("--eval_type", choices=['reader', "generator", 'retriever', "hybrid_retriever", "full_pipeline"], required=True,
                        help="Choose the type of evaluation to run")
    parser.add_argument("--file_path", type=str, help="File path for 'other' dataset evaluation")

    args = parser.parse_args()

    if args.evaluate == "xquad":
        evaluate_on_xquad(args.eval_type)
    elif args.evaluate == "npho_10":
        evaluate_on_npho('10', args.eval_type)
    elif args.evaluate == "npho_20":
        evaluate_on_npho('20', args.eval_type)
    elif args.evaluate == "other":
        if args.file_path:
            evaluate_on_other_dataset(args.file_path, args.eval_type)
        else:
            print("Please provide a file path for 'other' dataset evaluation.")

if __name__ == "__main__":
    main()