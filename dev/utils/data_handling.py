from typing import List, Union
from haystack.utils import SquadData
from haystack import Document
import random
import os
import sys
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(SCRIPT_DIR)))
from haystack.nodes import PreProcessor
from haystack.nodes import TransformersTranslator
import json 
from transformers import AutoTokenizer
from tqdm import tqdm

def fit_passage_in_max_len ( passage, answer, tokenizer, max_seq_len):
    """
    Truncates provided passage based on the maximum sequence length of the sentence transformer model.
    Simultaneously, it ensures that the response to the associated question remains contained within the truncated passage
    """

    # Tokenize according to the given model's tokenizer 
    passage_tokens = tokenizer.tokenize (passage)
    answer_tokens = tokenizer.tokenize (answer)

    # Calculate total tokens to keep while Reserving tokens for [CLS], [SEP], and space
    tokens_to_keep = max_seq_len - len(answer_tokens) - 3 

    if len(passage_tokens) <= tokens_to_keep:
        tokenizer.convert_tokens_to_string (answer_tokens), tokenizer.convert_tokens_to_string (passage_tokens)
    else:
        #try:
        # Find the token number in which the answer starts
            try:
                answer_start = passage_tokens.index(answer_tokens[0])

                # Calculate context window AutoTokenizer.from_pretrained("nlpaueb/bert-base-greek-uncased-v1")
                passage_start = max(0, answer_start - (tokens_to_keep // 2))
                passage_end = min(len(passage_tokens), answer_start + len(answer_tokens) + (tokens_to_keep // 2))

                # Adjust context window if needed
                if passage_end - passage_start < tokens_to_keep:
                    if passage_end == len(passage_tokens):
                        passage_start = max(0, passage_end - tokens_to_keep)
                    else:
                        passage_end  = min(len(passage_tokens), passage_start + tokens_to_keep)
                        
                # Truncate context
                truncated_passage_tokens = passage_tokens[passage_start : passage_end]
                truncated_passage =tokenizer.convert_tokens_to_string(truncated_passage_tokens)
                return  tokenizer.convert_tokens_to_string (answer_tokens), truncated_passage
            except ValueError:
                print (f"Answer: {answer} not found in given passage")

class DataExtractor:
    """This class helps extract query-answer pairs from datasets in SQuAD, DPR, or simple JSON format."""
    
    def __init__(self, tokenizer:AutoTokenizer, max_seq_len:int) -> None:
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        pass

    def get_query_doc_pairs_from_squad_file(self, squad_file):
        """"
        Loads query-positive passage pairs from a SQuAD dataset file, including the step of truncating passages to fit the encoder's maximum sequence length
        """
        query_doc_pairs = []
        with open (squad_file, "r") as fp:
            squad_data = json.load(fp)["data"]
            pairs_count = 0
        for data in tqdm(squad_data):
            for paragraph in data["paragraphs"]:
                for qa in paragraph["qas"]:
                    question = qa["question"]
                    passage = paragraph["context"]
                    answer = [a["text"] for a in qa["answers"]][0]

                    # truncate document to max_seq_length to fit in the encoder model
                    results = fit_passage_in_max_len (passage= passage, answer=answer, tokenizer=self.tokenizer, max_seq_len=self.max_seq_len)
                    if results is not None:
                        _, passage = results
                        query_doc_pairs.append ({"question": question, "document": passage})
                        pairs_count += 1
                    else:
                        continue

        return query_doc_pairs
    
    def get_query_doc_pairs_from_json_file (self, filepath):
        """
        loads query-passage pairs from a json file. The file must contain the data in the following right format: [{"question": str, "document":str}]
        """
        with open(filepath, "r") as fp:
            data =  json.load(fp)
            truncated_data = []
            for qd_pair in tqdm(data):
                tokenized_doc = self.tokenizer.tokenize (qd_pair["document"])[:self.max_seq_len]
                truncated_data.append ({"question": qd_pair["question"], "document": self.tokenizer.convert_tokens_to_string(tokenized_doc)})
            return truncated_data
        
    def get_query_doc_pairs_from_dpr_file (self, dpr_filename):

        """
        loads query-positive passage pairs from a DPR dataset file, assuming that the passages are already truncated to fit the maximum sequence length of the model.
        """
        query_doc_pairs = []
        with open (dpr_filename, "r") as fp:
            data = json.load(fp)
            for d in data:
                question = d["question"]
                for positive_ctx in d["positive_ctxs"]:
                    document = positive_ctx["text"]
            query_doc_pairs.append({"question": question, "document": document})
        return query_doc_pairs
    
def split_squad_dataset (filepath, split_ratio:int = 0.1):
    """Splits SQuAD to train and dev sets"""

    with open(filepath, encoding="utf-8") as f:
        # load and count total num of examples
        data = json.load(f)
        num_of_examples =  SquadData (data).count()

        # shuffle examples
        data = data["data"]
        random.shuffle(data)
        counter = 0
        test_set = []
        for article in data:            
            for paragraph in article["paragraphs"]:
                counter += (len(paragraph["qas"]))
            if counter >= round (num_of_examples * split_ratio):
                break
            else:
                test_set.append (article)
        train_set = {"data" : data [len(test_set):]}
        test_set = {"data" : test_set}
    print (f"train set instances: {(num_of_examples-counter)}\n dev set instances: {counter}")
    # Write datasets
    path = os.path.dirname (filepath)
    with open(os.path.join(path, "train_file.json"), 'w') as train_file:
        json.dump(train_set, train_file, ensure_ascii=False, indent=4)
    with open(os.path.join(path,"dev_file.json"), 'w') as dev_file:       
        json.dump (test_set, dev_file, ensure_ascii=False, indent=4)

def split_dpr_dataset(dpr_filepath, ratio: float = 0.1):
    """Splits DPR dataset to train and dev sets"""

    with open(dpr_filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)

    random.shuffle(data)
    split_idx = int(len(data) * (1 - ratio))
    train_data = data[:split_idx]
    dev_data = data[split_idx:]
    base_filename = os.path.splitext(os.path.basename(dpr_filepath))[0]
    train_filepath = os.path.join(os.path.dirname(dpr_filepath), f"{base_filename}_train.json")
    dev_filepath = os.path.join(os.path.dirname(dpr_filepath), f"{base_filename}_dev.json")
    with open(train_filepath, 'w', encoding='utf-8') as f:
        json.dump(train_data, f, ensure_ascii=False, indent=4)
    with open(dev_filepath, 'w', encoding='utf-8') as f:
        json.dump(dev_data, f, ensure_ascii=False, indent=4)
    print(f"Training data saved to {train_filepath}")
    print(f"Dev data saved to {dev_filepath}")

def translate_docs (docs:List[str], use_gpu:bool=False):
    """Translates documnets using a greek NMT model"""
    max_seq_len = 512
    translator = TransformersTranslator(model_name_or_path="Helsinki-NLP/opus-mt-en-el", use_gpu=use_gpu, max_seq_len=max_seq_len)
    try:
        t_docs = translator.translate_batch(documents=docs)[0]
    except AttributeError:
        t_docs = ['<ukn>']
    return t_docs

def split_and_translate (docs:List[str], max_seq_len:int = 512):

    preprocessor = PreProcessor (language='en', split_by='word', split_length=max_seq_len,  split_respect_sentence_boundary=True, progress_bar=False)
    idx_to_splitted_docs = {}

    for idx, doc in enumerate (docs):
        tokens = [word for word in doc.strip().split(' ')]
        if len(tokens) > max_seq_len:
            docs.pop(idx)
            doc = Document (content=doc)
            splitted_docs = [doc.content for doc in preprocessor.process([doc])]
            # keep track splitted long answers 
            idx_to_splitted_docs[idx] = splitted_docs
        else:
            continue
    
    t_docs = translate_docs(docs)

    if idx_to_splitted_docs:

        for idx, splitted_docs in idx_to_splitted_docs.items():
            # translate splitted docs and join them 
            t_answer = ' '.join (translate_docs(splitted_docs))
            t_docs.insert(idx, t_answer)

    return t_docs

def translate_docs (docs:List[str], use_gpu:bool=False):
    
    max_seq_len = 512
    translator = TransformersTranslator(model_name_or_path="Helsinki-NLP/opus-mt-en-el", use_gpu=use_gpu, max_seq_len=max_seq_len)
    try:
        t_docs = translator.translate_batch(documents=docs)[0]
    except AttributeError:
        t_docs = ['<ukn>']
    return t_docs

def join_punctuation(seq, characters='.,;?!:'):
    characters = set(characters)
    seq = iter(seq)
    current = next(seq)

    for nxt in seq:
        if nxt in characters:
            current += nxt
        else:
            yield current
            current = nxt

    yield current
    return ' '.join(seq)

