import os 
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(SCRIPT_DIR, os.pardir, 'models_cache')
sys.path.append(os.path.dirname(SCRIPT_DIR))

from transformers import AutoTokenizer
from haystack.nodes import  PromptNode, PromptTemplate, AnswerParser 
from haystack.nodes.base import BaseComponent
from utils.data_handling import post_process_generator_answers, flush_cuda_memory
import logging
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch

logging.basicConfig(format="%(levelname)s - %(name)s -  %(message)s", level=logging.INFO)
logging.getLogger("haystack").setLevel(logging.INFO)

class Generator(BaseComponent):
    """"Loads a predefined instruction-following LLM for causal generation on a given prompt."""
    outgoing_edges = 1

    def __init__(self,
                model_name = "ilsp/Meltemi-7B-Instruct-v1.5",
                prompt_messages=[
                     {"role": "system", "content": 'Είσαι ένας ψηφιακός βοηθός που απαντάει σε ερωτήσεις. Δώσε μία σαφή και ολοκληρωμένη απάντηση στην ερώτηση του χρήστη με βάση τις σχετικές πληροφορίες.'},
                     {"role": "user", "content": 'Ερώτηση:\n {query} \n Πληροφορίες: \n {join(documents)} \n Απάντηση: \n'}
                     ]):
        
        self.model_name = model_name
        self.model = self._load_model(self.model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.prompt = self.tokenizer.apply_chat_template(prompt_messages, add_generation_prompt=True, tokenize=False)
        self.prompt_template = PromptTemplate(prompt = self.prompt, output_parser=AnswerParser(pattern = r"(?<=<\|assistant\|>\n)([\s\S]*)")) # pattern = r"(?<=<\|assistant\|>\n)([\s\S]*)"
        
        super().__init__()

    def _load_model(self, model_name):
        """Μethod to load the model using 4bit quantization"""

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            load_in_4bit_fp32_cpu_offload=True
        )

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            cache_dir=CACHE_DIR,
            quantization_config=bnb_config,
            device_map="cuda"
        )

        model.gradient_checkpointing_enable()
        model.config.pretraining_tp = 1
        flush_cuda_memory()
        
        return model
    
    def run(self, query, documents, cr_score, max_new_tokens:int=150, temperature:float = 0.75, top_p:float = 0.95, top_k:int=50, post_processing = True):
        
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

        results = generate_answer()
        answer = results["answers"][0].answer.strip()
        attempt = 0
        max_attempts = 3 

        # Retry logic with max_attempts in the rare case where the first attempt resulted to invalid answer
        while (answer == '' or answer == "Συγγνώμη, αλλά δεν μπορώ να δώσω μία απάντηση σε αυτήν την ερώτηση.") and attempt < max_attempts:
            logging.warning(f"Empty or invalid answer received. Retrying... Attempt {attempt+1}")
            results = generate_answer()
            answer = results["answers"][0].answer.strip() 
            attempt += 1

        # If max attempts reached and still invalid answer
        if attempt == max_attempts and (answer == '' or answer == "Συγγνώμη, αλλά δεν μπορώ να δώσω μία απάντηση σε αυτήν την ερώτηση."):
            logging.error(f"Failed to generate a valid answer after {max_attempts} attempts.")

        results["attempts"] = attempt
        results["cr_score"] = cr_score
        results.pop("invocation_context")

        return results, "output_1"

    def run_batch(self):
        return