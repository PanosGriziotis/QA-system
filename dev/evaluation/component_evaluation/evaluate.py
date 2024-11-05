from typing import List, Optional, Dict, Any, Union, Callable, Tuple
from tqdm import tqdm
import os
import sys
import tempfile
import json
from pathlib import Path
from typing import List, Optional
from haystack import Pipeline
from haystack.document_stores import InMemoryDocumentStore, ElasticsearchDocumentStore
from haystack.nodes import PreProcessor
from haystack.nodes import FARMReader, EmbeddingRetriever, DensePassageRetriever, BM25Retriever, SentenceTransformersRanker
from haystack.nodes import JoinDocuments
import logging
import os 
import sys
import logging

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(SCRIPT_DIR, '../../../')))

logging.basicConfig(format="%(levelname)s - %(name)s -  %(message)s", level=logging.INFO)
logging.getLogger("haystack").setLevel(logging.INFO)


def index_eval_labels(document_store, eval_filename: str):
    """
    Index evaluation labels into the document store.
    """
    # Delete indeces in Document Store ("document", "labels") and rewrite docs and labels respectively
    document_store.delete_index(index=document_store.index)
    document_store.delete_index (index=document_store.label_index)

    label_preprocessor = PreProcessor(
        split_by="word",
        split_length=128,
        split_respect_sentence_boundary=False,
        clean_empty_lines=False,
        clean_whitespace=False,
        language="el"
    )
    
    document_store.add_eval_data(
        filename=eval_filename,
        doc_index=document_store.index,
        label_index=document_store.label_index,
        preprocessor=label_preprocessor,
    )

def get_eval_labels_and_paths(document_store, tempdir) -> Tuple[List[dict], List[Path]]:
    """
    Get evaluation labels and file paths for indexed eval documents in the document store. 
    This function is only used when we want to run an expirement in mlflow, evaluating a pipeline.

    document_store (DocumentStore): The document store instance.
    tempdir: A temporary directory instance for storing document files.
    """

    file_paths = []
    docs = document_store.get_all_documents()

    # Save docs in temporary files and get file paths
    for doc in docs:
        file_name = f"{doc.id}.txt"
        file_path = Path(tempdir.name) / file_name
        file_paths.append(file_path)
        
        with open(file_path, "w") as f:
            f.write(doc.content)
    
    # Get MultiLabel objects 
    evaluation_set_labels = document_store.get_all_labels_aggregated(drop_negative_labels=True, drop_no_answers=True)

    return evaluation_set_labels, file_paths


def evaluate_retriever(
        retriever:Union[BM25Retriever, EmbeddingRetriever, DensePassageRetriever],
        document_store: ElasticsearchDocumentStore,
        eval_filename: str,
        top_k: Optional[int] = None,
        top_k_list: Optional[List[int]] = None) -> Dict[int, dict]:
    
    """
    Evaluate a retriever on a SQuAD format evaluation dataset. 
    If a top_k_list is provided, the evaluation is iterative for 
    each top_k value, generating 1 evaluation report for each value.
    """

    # Step 1: Index eval documents and labels in document store and compute doc embeddings for dense retrievers.
    index_eval_labels(document_store, eval_filename)
    if isinstance(retriever, (EmbeddingRetriever, DensePassageRetriever)):
        document_store.update_embeddings(retriever= retriever)
    
    # Step 2a: Generate evaluation report for each top_k value and save in nested dictionary.
    if top_k_list is not None:
        reports = {}
        for k in tqdm(top_k_list):
            reports[k] = retriever.eval(label_index=document_store.label_index, doc_index=document_store.index, top_k=k, document_store=document_store)
        return reports
    # Step 2b: Generate evaluation report for single top_k value.
    else:
        if top_k is None:
            top_k = retriever.top_k
        report = retriever.eval(label_index=document_store.label_index, doc_index=document_store.index, top_k=top_k, document_store=document_store)
        return report

def evaluate_reader(
        reader:FARMReader,
        eval_filename: str,
        calibrate_scores:bool=True,
        top_k: Optional[int] = None,
        top_k_list: Optional[List[int]] = None) -> Dict[int, dict]:
    """
    Evaluate reader in isolation on a SQuAD format evaluation dataset.
    """

    data_dir = os.path.dirname(eval_filename)
    filename = os.path.basename(eval_filename)
    if top_k_list is not None:
        reports = {}
        for k in tqdm(top_k_list, desc="Evaluating reader"):
            reader.top_k = k
            reports[k] = reader.eval_on_file(data_dir, filename,calibrate_conf_scores=calibrate_scores)
        return reports
    elif top_k is not None:
        reader.top_k = top_k
        result = reader.eval_on_file(data_dir, filename)
    else:
        result =  reader.eval_on_file(data_dir, filename)
    
    return result

def evaluate_retriever_ranker_pipeline(
        retriever:Union[BM25Retriever, EmbeddingRetriever, DensePassageRetriever],
        ranker:SentenceTransformersRanker,
        document_store: ElasticsearchDocumentStore,
        eval_filename: str,
        top_k: Optional[int] = None,
        top_k_list: Optional[List[int]] = None) -> Dict[int, dict]:
    
    """Evaluate both Retriever and Ranker components in a pipeline fashion"""

    # Step 1: Index eval documents and labels in document store and compute doc embeddings for dense retrievers.
    index_eval_labels(document_store, eval_filename)
    if isinstance(retriever, (EmbeddingRetriever, DensePassageRetriever)):
        document_store.update_embeddings(retriever=retriever)
    
    # Step 2: Build a pipeline of both components to evaluate.
    p = Pipeline()
    p.add_node(retriever, name="Retriever", inputs=["Query"])
    p.add_node(ranker, name="Ranker", inputs=["Retriever"])

    # Get labels from document store. Note: embeddings are not returned in     
    labels = document_store.get_all_labels_aggregated()

    # Step 2a: Generate evaluation report for each top_k value and save in nested dictionary.
    if top_k_list is not None:
        reports = {}
        for top_k in tqdm(top_k_list):

            report = p.eval(
                labels=labels,
                params={"top_k": top_k})
            
            reports[top_k] = report.calculate_metrics()
        
        return reports
    
    # Step 2b: Generate evaluation report for single top_k value.
    report = p.eval(labels=labels,
            add_isolated_node_eval=True)
    
    return report

def evaluate_hybrid_retriever_ranker_pipeline (
        retrievers,
        ranker,
        document_store,
        eval_filename: str,
        top_k: Optional[int] = None,
        top_k_list: Optional[List[int]] = None) -> Dict[int, dict]:
    
    """Evaluate both Retriever and Ranker components in a pipeline fashion"""

    # Step 1: Index eval documents and labels in document store and compute doc embeddings for dense retrievers.
    index_eval_labels(document_store, eval_filename)
    for retriever in retrievers:
        if isinstance(retriever, (EmbeddingRetriever, DensePassageRetriever)):
            dense_retriever = retriever
            document_store.update_embeddings(retriever=dense_retriever)
        elif isinstance(retriever, (BM25Retriever)):
            bm25_retriever = retriever
    
    join_documents = JoinDocuments(join_mode="concatenate")
    # Step 2: Build a pipeline of both components to evaluate.
    p = Pipeline()
    p.add_node(bm25_retriever, name="BM25Retriever", inputs=["Query"])
    p.add_node(dense_retriever, name="DenseRetriever", inputs=["Query"])
    p.add_node(component=join_documents, name="JoinDocuments", inputs=["BM25Retriever", "DenseRetriever"])
    p.add_node(component=ranker, name="Ranker", inputs=["JoinDocuments"])

    # Get labels from document store. Note: embeddings are not returned in     
    labels = document_store.get_all_labels_aggregated()


    # Step 2a: Generate evaluation report for each top_k value and save in nested dictionary.
    if top_k_list is not None:
        reports = {}
        for top_k in tqdm(top_k_list):
            if top_k % 2 == 0:
                top_k_bm25 = top_k // 2
                top_k_dense = top_k // 2
            else:
                top_k_dense = (top_k // 2) + 1  # Give priority to dense
                top_k_bm25 = top_k // 2
                        
            report = p.eval(
                labels=labels,
                params= {"BM25Retriever": {"top_k": top_k_bm25}, "DenseRetriever": {"top_k": top_k_dense}, "JoinDocuments": {"top_k_join": top_k}, "Ranker": {"top_k": top_k}})
            
            reports[top_k] = report.calculate_metrics()
        
        return reports
    
    # Step 2b: Generate evaluation report for single top_k value.
    report = p.eval(labels=labels,
            add_isolated_node_eval=True)
    
    return report

def evaluate_pipeline(pipeline:Pipeline, document_store:ElasticsearchDocumentStore, params:dict, filename:str):
    
    """Evaluate haystack pipeline end-to-end on SQuAD-like data"""
    
    index_eval_labels(document_store=document_store, eval_filename=filename)
    document_store.update_embeddings(retriever=pipeline.get_node("Retriever"))
    labels = document_store.get_all_labels_aggregated()
    documents = document_store.get_all_documents()    
    
    result = pipeline.eval(labels=labels,
                  documents=documents,
                  params=params,
                  sas_model_name_or_path="panosgriz/covid_el_paraphrase-multilingual-MiniLM-L12-v2",
                  add_isolated_node_eval=True,
    )

    integrated = result.calculate_metrics(eval_mode="integrated")
    isolated = result.calculate_metrics(eval_mode="isolated")

    return {"integrated":integrated, "isolated":isolated}
