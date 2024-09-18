
from typing import List, Optional, Dict, Any, Union, Callable, Tuple
import sys
import logging
import tempfile
import os
import sys
from haystack import Pipeline
from haystack.document_stores import ElasticsearchDocumentStore
from haystack.nodes import EmbeddingRetriever, FARMReader
import json

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(SCRIPT_DIR, '../../')))

#from src.pipelines.extractive_qa_pipeline import extractive_qa_pipeline
from src.pipelines.indexing_pipeline import indexing_pipeline
from src.pipelines.ranker import SentenceTransformersRanker
from src.pipelines.query_pipelines import init_extractive_qa_pipeline, init_rag_pipeline
from evaluate import index_eval_labels, get_eval_labels_and_paths
import torch
import gc
import argparse

logging.basicConfig(format="%(levelname)s - %(name)s -  %(message)s", level=logging.INFO)
logging.getLogger("haystack").setLevel(logging.INFO)

def run_experiment(
        exp_name: str,
        eval_filename: str,
        query_pipeline:Pipeline,
        index_pipeline:Pipeline,
        run_name: str,
        query_params=dict):
    """
    Run an experiment with the given parameters.

    exp_name: The name of the experiment.
    eval_filename: The path to the evaluation dataset file.
    pipeline_path: The path to the pipeline YAML file.
    run_name: The name of the experiment run.
    query_params: Parameters for query pipeline
    """
    

    # Create a temporary directory and document store to get eval data file paths
    temp_dir = tempfile.TemporaryDirectory()
    eval_ds = ElasticsearchDocumentStore(embedding_dim=384)
    index_eval_labels(eval_ds, eval_filename)
    evaluations_set_labels, file_paths = get_eval_labels_and_paths(eval_ds, temp_dir)
    
    eval_ds.delete_documents()
    eval_ds.delete_labels()
    print ("NUMBERS BEFORE RUNNING EXECUTE_EVAL_RUN")
    print (len(evaluations_set_labels))
    print (len(file_paths))
    # Execute experiment run
    Pipeline.execute_eval_run(
        index_pipeline=index_pipeline,
        query_pipeline=query_pipeline,
        evaluation_set_labels=evaluations_set_labels,
        corpus_file_paths=file_paths,
        experiment_name=exp_name,
        experiment_run_name=run_name,
        evaluation_set_meta=os.path.basename(eval_filename),
        add_isolated_node_eval=True,
        experiment_tracking_tool="mlflow",
        experiment_tracking_uri="http://localhost:5001",
        query_params=query_params,
        sas_model_name_or_path="lighteternal/stsb-xlm-r-greek-transfer",
        sas_batch_size=32,
        sas_use_gpu=True,
        reuse_index=False
    )
    
    temp_dir.cleanup()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_type", choices=["extractive", "rag"], required=True,
                        help="Choose the type of evaluation to run")
    parser.add_argument("--reader_model", required=False, type=str)
    parser.add_argument("--params", type=str)

    args = parser.parse_args()

    #datasets = ["datasets/npho-covid-SQuAD-el_10.json", "datasets/npho-covid-SQuAD-el_20.json", "datasets/xquad-el.json"]
    datasets = ["datasets/npho-covid-SQuAD-el_20.json", "datasets/xquad-el.json"]
    query_params = json.loads(args.params)

    if args.eval_type == "extractive":
        
        exp_name = args.reader_model
        ds = ElasticsearchDocumentStore(embedding_dim=384)
        retriever = EmbeddingRetriever(
            embedding_model="panosgriz/covid_el_paraphrase-multilingual-MiniLM-L12-v2", document_store=ds, use_gpu=True)

        ranker = SentenceTransformersRanker(
            model_name_or_path="amberoad/bert-multilingual-passage-reranking-msmarco", use_gpu=True)
        
        reader = FARMReader(model_name_or_path=args.reader_model, use_gpu=True)

        query_pipeline = Pipeline()
        query_pipeline.add_node(component=retriever, name="Retriever", inputs=["Query"])
        query_pipeline.add_node(component=ranker, name="Ranker", inputs=["Retriever"])
        #query_pipeline.add_node(component=reader, name="Reader", inputs=["Retriever"])
        query_pipeline.add_node(component=reader, name="Reader", inputs=["Ranker"])
    
        reader_top_k = query_params["Reader"]["top_k"]


    elif args.eval_type == "rag":
        query_pipeline = init_rag_pipeline()
        exp_name = "rag_pipeline_2"
    
        
        max_new_tokens=query_params["Generator"]["max_new_tokens"]

    
    retriever_top_k= query_params["Retriever"]["top_k"]

    try:
        ranker_top_k=query_params["Ranker"]["top_k"]
    except:
        KeyError

    for dataset in datasets:
        if args.eval_type == "rag":

            run_name = f"{os.path.basename(dataset).split('.')[0]}_{ranker_top_k}_{max_new_tokens}"
        else:
            run_name = f"{os.path.basename(dataset).split('.')[0]}_{ranker_top_k}_{reader_top_k}"
        run_experiment(
            exp_name=exp_name, 
            eval_filename=dataset,
            query_pipeline=query_pipeline,
            index_pipeline=indexing_pipeline,  
            run_name=run_name,
            query_params=query_params
        )
    
        torch.cuda.empty_cache()
        gc.collect()