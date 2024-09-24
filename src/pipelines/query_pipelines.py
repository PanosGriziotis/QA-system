
from typing import List, Dict, Any, Optional, Union

import os 
import sys
import logging
import string

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch

from haystack.pipelines import Pipeline
from haystack.nodes import  EmbeddingRetriever,BM25Retriever, FARMReader, PromptNode, PromptTemplate, AnswerParser 
from haystack.nodes.base import BaseComponent

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from document_store.initialize_document_store import document_store as DOCUMENT_STORE
from pipelines.ranker import SentenceTransformersRanker
from utils.data_handling import post_process_generator_answers
import pandas as pd
from haystack.nodes import JoinDocuments

logging.basicConfig(format="%(levelname)s - %(name)s -  %(message)s", level=logging.WARNING)
logging.getLogger("haystack").setLevel(logging.INFO)

if DOCUMENT_STORE is None:
    raise ValueError("the imported document_store is None. Please make sure that the Elasticsearch service is properly launched")


class Generator(BaseComponent):
    """"""
    outgoing_edges = 1

    def __init__(self,
                model_name = "ilsp/Meltemi-7B-Instruct-v1.5",
                prompt_messages:List[Dict]=[
                     {"role": "system", "content": 'Είσαι ένας ψηφιακός βοηθός που απαντάει σε ερωτήσεις. Δώσε μία σαφή και ολοκληρωμένη απάντηση στην ερώτηση του χρήστη με βάση τις σχετικές πληροφορίες.'},
                     {"role": "user", "content": 'Ερώτηση:\n {query} \n Πληροφορίες: \n {join(documents)} \n Απάντηση: \n'}
                     ]):
        
        self.model_name = model_name
        self.model = load_model(self.model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.prompt = self.tokenizer.apply_chat_template(prompt_messages, add_generation_prompt=True, tokenize=False)
        self.prompt_template = PromptTemplate(prompt = self.prompt, output_parser=AnswerParser(pattern = r"(?<=<\|assistant\|>\n)([\s\S]*)")) # pattern = r"(?<=<\|assistant\|>\n)([\s\S]*)"
        super().__init__()

    def run(self, query, documents, max_new_tokens:int=150, temperature:float = 0.75, top_p:float = 0.95, top_k:int=50, post_processing = True):
        """"""
        generation_kwargs={
                            'max_new_tokens': max_new_tokens,
                            'top_k': top_k,
                            'top_p': top_p,
                            'temperature': temperature,
                            'do_sample': True
                            }
        
        generator = PromptNode(model_name_or_path = self.model_name,
                               default_prompt_template = self.prompt_template,
                               top_k= 1,
                               model_kwargs = {
            'model': self.model,
            'tokenizer': self.tokenizer,
            'task_name': 'text2text-generation',
            'device': None,
            "generation_kwargs": generation_kwargs
        })


        result, _ = generator.run(query=query, documents=documents)
        # Remove invocation context from final result for being redundant information
        result.pop("invocation_context")

        if post_processing:
            # Post-process Generator's output: 
            # a) remove trailing words in incomplete outputs that end abruptly.
            # b) remove words or phrases in the output coming accidently from the prompt message (e.g., Ερώτηση:)

            c_result = post_process_generator_answers(result)
            answer = result["answers"][0].answer.strip()

            if answer == '' or (len(answer) == 1 and answer in string.punctuation):
                pass
            else:
                return c_result, 'output_1'
        
        return result, 'output_1'
        
    def run_batch(
        self):
         return
    
def load_model (model_name):
    """Load LLM to run in 4bit quantization"""

    bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    load_in_8bit_fp32_cpu_offload=True
    
    )

    model = AutoModelForCausalLM.from_pretrained(model_name,  quantization_config=bnb_config, device_map="cuda")
    # Disable Tensor Parallelism 
    model.config.pretraining_tp=1
    
    return model

def init_rag_pipeline (use_gpu:bool=True):
    
    bm25_retriever  = BM25Retriever(document_store=DOCUMENT_STORE)
    dense_retriever = EmbeddingRetriever(
        embedding_model="panosgriz/covid_el_paraphrase-multilingual-MiniLM-L12-v2",
        document_store=DOCUMENT_STORE,
        use_gpu=use_gpu,
        top_k=20
        )
    
    join_documents = JoinDocuments(join_mode="concatenate")

    ranker_model_name_or_path  = os.path.join(SCRIPT_DIR, "models/bert-multilingual-passage-reranking-msmarco")
    if not os.path.exists(ranker_model_name_or_path):

        ranker_model_name_or_path = "amberoad/bert-multilingual-passage-reranking-msmarco"
    
    ranker = SentenceTransformersRanker(
        model_name_or_path=ranker_model_name_or_path,
        use_gpu=use_gpu,
        top_k=10
        )    
    
    generator = Generator()

    p = Pipeline()
    
    p.add_node(component=bm25_retriever, name="BM25Retriever", inputs=["Query"])
    p.add_node(component=dense_retriever, name="DenseRetriever", inputs=["Query"])
    p.add_node(component=join_documents, name="JoinDocuments", inputs=["BM25Retriever", "DenseRetriever"])
    p.add_node(component=ranker, name="Ranker", inputs=["JoinDocuments"])
    p.add_node(component=generator, name="Generator", inputs=["Ranker"])

    
    return p

def init_extractive_qa_pipeline (use_gpu:bool=True):

    bm25_retriever  = BM25Retriever(document_store=DOCUMENT_STORE)
    dense_retriever = EmbeddingRetriever(
        embedding_model="panosgriz/covid_el_paraphrase-multilingual-MiniLM-L12-v2",
        document_store=DOCUMENT_STORE,
        use_gpu=use_gpu,
        top_k=20
        )
    join_documents = JoinDocuments(join_mode="concatenate")

    
    ranker_model_name_or_path  = os.path.join(SCRIPT_DIR, "models/bert-multilingual-passage-reranking-msmarco")
    if not os.path.exists(ranker_model_name_or_path):

        ranker_model_name_or_path = "amberoad/bert-multilingual-passage-reranking-msmarco"
    
    ranker = SentenceTransformersRanker(
        model_name_or_path=ranker_model_name_or_path,
        use_gpu=use_gpu,
        top_k=10
        )
    
    reader = FARMReader(
        model_name_or_path="panosgriz/mdeberta-v3-base-squad2-covid-el_small",
        use_gpu=use_gpu,
        top_k = 10 
        )
    
    p = Pipeline()
    p.add_node(component=bm25_retriever, name="BM25Retriever", inputs=["Query"])
    p.add_node(component=dense_retriever, name="DenseRetriever", inputs=["Query"])
    p.add_node(component=join_documents, name="JoinDocuments", inputs=["BM25Retriever", "DenseRetriever"])
    p.add_node(component=ranker, name="Ranker", inputs=["JoinDocuments"])
    p.add_node(component=reader, name="Reader", inputs=["Ranker"])
    return p
