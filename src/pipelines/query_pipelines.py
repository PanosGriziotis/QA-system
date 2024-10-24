import os 
import sys
import logging

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch
import gc

from haystack.pipelines import Pipeline
from haystack.nodes import  EmbeddingRetriever,BM25Retriever, FARMReader, PromptNode, PromptTemplate, AnswerParser 
from haystack.nodes.base import BaseComponent

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from document_store.initialize_document_store import document_store as DOCUMENT_STORE
from pipelines.ranker import SentenceTransformersRanker
from utils.data_handling import post_process_generator_answers, flash_cuda_memory
from utils.metrics import ContextRelevanceEvaluator
from haystack.nodes import JoinDocuments

logging.basicConfig(format="%(levelname)s - %(name)s -  %(message)s", level=logging.INFO)
logging.getLogger("haystack").setLevel(logging.INFO)

if DOCUMENT_STORE is None:
    raise ValueError("the imported document_store is None. Please make sure that the Elasticsearch service is properly launched")


class Generator(BaseComponent):
    """"Loads a predefined instruction-following LLM for causal generation on a given prompt. Returns an object containing the answer and outputs of previous nodes (e.g., documents)."""
    outgoing_edges = 1

    def __init__(self,
                model_name = "ilsp/Meltemi-7B-Instruct-v1.5",
                prompt_messages=[
                     {"role": "system", "content": 'Είσαι ένας ψηφιακός βοηθός που απαντάει σε ερωτήσεις. Δώσε μία σαφή και ολοκληρωμένη απάντηση στην ερώτηση του χρήστη με βάση τις σχετικές πληροφορίες.'},
                     {"role": "user", "content": 'Ερώτηση:\n {query} \n Πληροφορίες: \n {join(documents)} \n Απάντηση: \n'}
                     ]):
        
        self.model_name = model_name
        self.model = load_model(self.model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.prompt = self.tokenizer.apply_chat_template(prompt_messages, add_generation_prompt=True, tokenize=False)
        self.prompt_template = PromptTemplate(prompt = self.prompt, output_parser=AnswerParser(pattern = r"(?<=<\|assistant\|>\n)([\s\S]*)")) # pattern = r"(?<=<\|assistant\|>\n)([\s\S]*)"
        
        super().__init__()

    def run(self, query, documents, cr_score, max_new_tokens:int=150, temperature:float = 0.75, top_p:float = 0.95, top_k:int=50, post_processing = True, apply_cr_threshold=True):
        
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
   

        def generate_answer():
            generator_output, _ = generator.run(
                query=query,
                documents=documents)
            
            return post_process_generator_answers(generator_output) if post_processing else generator_output

        result = generate_answer()
        answer = result["answers"][0].answer.strip()
        attempt = 0
        max_attempts = 3 

        # Retry logic with max_attempts check
        while (answer == '' or answer == "Συγγνώμη, αλλά δεν μπορώ να δώσω μία απάντηση σε αυτήν την ερώτηση.") and attempt < max_attempts:
            logging.warning(f"Empty or invalid answer received. Retrying... Attempt {attempt+1}")
            result = generate_answer()
            answer = result["answers"][0].answer.strip() 
            attempt += 1

        # If max attempts reached and still invalid answer
        if attempt == max_attempts and (answer == '' or answer == "Συγγνώμη, αλλά δεν μπορώ να δώσω μία απάντηση σε αυτήν την ερώτηση."):
            logging.error(f"Failed to generate a valid answer after {max_attempts} attempts.")

        if apply_cr_threshold and cr_score <= 0.17:
           result["answers"][0].answer = "Συγγνώμη, αλλά δεν διαθέτω αρκετές πληροφορίες για να σου απαντήσω σε αυτήν την ερώτηση."

        result["attempts"] = attempt   
        return result, "output_1"

    def run_batch(self):
        return

    
def load_model(model_name):
    """Load LLM from the cache directory or download if not available. Load with 4bit quantization to not result in OOM error."""
    
    #torch.cuda.set_per_process_memory_fraction(0.8)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        load_in_4bit_fp32_cpu_offload=True
    )

    # Load model with cache_dir set to './models_cache'
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        cache_dir='./models_cache', 
        quantization_config=bnb_config,
        device_map="cuda",
        #max_memory = {0: '14GB'},
    )

    # Enable gradient checkpointing to reduce memory
    model.gradient_checkpointing_enable()
    
    # Disable Tensor Parallelism 
    model.config.pretraining_tp = 1
    
    flash_cuda_memory()
    
    return model

def load_retrievers(use_gpu=False):
    """Load document Retrievers (BM25 + SBERT)."""

    bm25_retriever = BM25Retriever(document_store=DOCUMENT_STORE, top_k=10)
    dense_retriever = EmbeddingRetriever(
        embedding_model="panosgriz/covid_el_paraphrase-multilingual-MiniLM-L12-v2",
        document_store=DOCUMENT_STORE,
        use_gpu=use_gpu,
        top_k=10
        )
    return bm25_retriever, dense_retriever

def load_ranker(ranker_model_name_or_path="amberoad/bert-multilingual-passage-reranking-msmarco", use_gpu=False):
    """Load document Reranker (cross-encoder model)"""
    
    model_name_part = ranker_model_name_or_path.split("/")[1]  
    model_path = os.path.join(SCRIPT_DIR, f'models/{model_name_part}')
    if os.path.exists(model_path):
        ranker_model = model_path
    else:
        ranker_model = ranker_model_name_or_path

    ranker = SentenceTransformersRanker(
        model_name_or_path=ranker_model,
        use_gpu=use_gpu,
        top_k=4    
        )
    return ranker

def load_reader(use_gpu=True):
    """Load a BERT-based extractive Reader."""
    reader = FARMReader(
        model_name_or_path="panosgriz/mdeberta-v3-base-squad2-covid-el_small",
        use_gpu=use_gpu,
        use_confidence_scores=True,
        top_k=10 
    )
    
    return reader

def init_rag_pipeline (use_gpu:bool=False):
    """initialize a RAG pipeline"""

    bm25_retriever, dense_retriever = load_retrievers(use_gpu=use_gpu)
    join_documents = JoinDocuments(join_mode="concatenate")
    ranker = load_ranker(use_gpu=use_gpu)
    generator = Generator()
    cr_evaluator = ContextRelevanceEvaluator()

    p = Pipeline()
    p.add_node(component=bm25_retriever, name="BM25Retriever", inputs=["Query"])
    p.add_node(component=dense_retriever, name="DenseRetriever", inputs=["Query"])
    p.add_node(component=join_documents, name="JoinDocuments", inputs=["BM25Retriever", "DenseRetriever"])
    p.add_node(component=ranker, name="Ranker", inputs=["JoinDocuments"])
    p.add_node(component=cr_evaluator, name="CREvaluator", inputs=["Ranker"])
    p.add_node(component=generator, name="Generator", inputs=["CREvaluator"])

    return p

def init_extractive_qa_pipeline (use_gpu:bool=True):
    """initialize extractive qa pipeline"""

    bm25_retriever, dense_retriever = load_retrievers(use_gpu=use_gpu)
    join_documents = JoinDocuments(join_mode="concatenate")
    ranker = load_ranker(use_gpu=use_gpu)
    reader=load_reader(use_gpu=use_gpu)
    
    p = Pipeline()
    p.add_node(component=bm25_retriever, name="BM25Retriever", inputs=["Query"])
    p.add_node(component=dense_retriever, name="DenseRetriever", inputs=["Query"])
    p.add_node(component=join_documents, name="JoinDocuments", inputs=["BM25Retriever", "DenseRetriever"])
    p.add_node(component=ranker, name="Ranker", inputs=["JoinDocuments"])
    p.add_node(component=reader, name="Reader", inputs=["Ranker"])

    return p
