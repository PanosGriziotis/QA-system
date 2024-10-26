import os 
import sys
import logging

from haystack.pipelines import Pipeline
from haystack.nodes import  EmbeddingRetriever,BM25Retriever, FARMReader
from haystack.nodes import JoinDocuments

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(SCRIPT_DIR, os.pardir, 'models_cache')
sys.path.append(os.path.dirname(SCRIPT_DIR))

from document_store.initialize_document_store import document_store as DOCUMENT_STORE
from custom_components.ranker import SentenceTransformersRanker
from custom_components.cr_relevance_evaluator import ContextRelevanceEvaluator
from custom_components.generator import Generator
from custom_components.responder import Responder

logging.basicConfig(format="%(levelname)s - %(name)s -  %(message)s", level=logging.INFO)
logging.getLogger("haystack").setLevel(logging.INFO)

DENSE_RETRIEVER_MODEL= os.environ["DENSE_RETRIEVER_MODEL"] if "DENSE_RETRIEVER_MODEL" in os.environ else "panosgriz/covid_el_paraphrase-multilingual-MiniLM-L12-v2"
RERANKER_MODEL = os.environ["RERANKER_MODEL"] if "RERANKER_MODEL" in os.environ else "amberoad/bert-multilingual-passage-reranking-msmarco"
EXTRACTIVE_READER_MODEL = os.environ["EXTRACTIVE_READER_MODEL"] if "EXTRACTIVE_READER_MODEL" in os.environ else "panosgriz/mdeberta-v3-base-squad2-covid-el_small"
GENERATIVE_READER_MODEL = os.environ["GENERATIVE_READER_MODEL"] if "GENERATIVE_READER_MODEL" in os.environ else "ilsp/Meltemi-7B-Instruct-v1.5"

if DOCUMENT_STORE is None:
    raise ValueError("the imported document_store is None. Please make sure that the Elasticsearch service is properly launched")

def load_retrievers(use_gpu=False):
    """Load dense and sparse document Retrievers for conducting a hybrid retrieval process (BM25 + SBERT)"""
    bm25_retriever = BM25Retriever(document_store=DOCUMENT_STORE, top_k=10)
    dense_retriever = EmbeddingRetriever(
        embedding_model=DENSE_RETRIEVER_MODEL,
        document_store=DOCUMENT_STORE,
        use_gpu=use_gpu,
        top_k=10
        )
    return bm25_retriever, dense_retriever

def load_ranker(use_gpu=False):
    """Load document Reranker (fine-tuned SBERT cross-encoder) to rerank the retrieved documents"""  
    ranker = SentenceTransformersRanker(
        model_name_or_path=RERANKER_MODEL,
        use_gpu=use_gpu,
        top_k=4    
        )    
    return ranker

def init_rag_pipeline (use_gpu:bool=False):
    """initialize a generative query pipeline"""

    bm25_retriever, dense_retriever = load_retrievers(use_gpu=use_gpu)
    join_documents = JoinDocuments(join_mode="concatenate")
    ranker = load_ranker(use_gpu=use_gpu)
    generator = Generator()
    cr_evaluator = ContextRelevanceEvaluator()
    responder = Responder()

    p = Pipeline()
    p.add_node(component=bm25_retriever, name="BM25Retriever", inputs=["Query"])
    p.add_node(component=dense_retriever, name="DenseRetriever", inputs=["Query"])
    p.add_node(component=join_documents, name="JoinDocuments", inputs=["BM25Retriever", "DenseRetriever"])
    p.add_node(component=ranker, name="Reranker", inputs=["JoinDocuments"])
    p.add_node(component=cr_evaluator, name="CREvaluator", inputs=["Reranker"])
    p.add_node(component=generator, name="GenerativeReader", inputs=["CREvaluator.output_1"])
    p.add_node(component=responder, name="Responder", inputs=["CREvaluator.output_2", "GenerativeReader"])

    return p

def init_extractive_qa_pipeline (use_gpu:bool=True):
    """initialize an extractive query pipeline"""

    bm25_retriever, dense_retriever = load_retrievers(use_gpu=use_gpu)
    join_documents = JoinDocuments(join_mode="concatenate")
    ranker = load_ranker(use_gpu=use_gpu)
    reader = FARMReader(model_name_or_path=EXTRACTIVE_READER_MODEL, use_gpu=True, use_confidence_scores=True, top_k=10)
    cr_evaluator = ContextRelevanceEvaluator()
    responder = Responder()
    
    p = Pipeline()
    p.add_node(component=bm25_retriever, name="BM25Retriever", inputs=["Query"])
    p.add_node(component=dense_retriever, name="DenseRetriever", inputs=["Query"])
    p.add_node(component=join_documents, name="JoinDocuments", inputs=["BM25Retriever", "DenseRetriever"])
    p.add_node(component=ranker, name="Reranker", inputs=["JoinDocuments"])
    p.add_node(component=cr_evaluator, name="CREvaluator", inputs=["Reranker"])
    p.add_node(component=reader, name="ExtractiveReader", inputs=["CREvaluator.output_1"])
    p.add_node(component=responder, name="Responder", inputs=["CREvaluator.output_2", "ExtractiveReader"])

    return p