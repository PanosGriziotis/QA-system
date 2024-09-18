import json
from tqdm.auto import tqdm
import random
from transformers import AutoTokenizer
from sentence_transformers import SentenceTransformer
from unidecode import unidecode
from datasets import load_dataset

class GPL_data:

    def __init__(self, tokenizer, max_seq_len) -> None:
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        pass


    def fit_passage_in_max_len (self, passage, answer):
        """
        Truncates provided passage based on the maximum sequence length of the sentence transformer model.
        Simultaneously, it ensures that the response to the associated question remains contained within the truncated passage
        """

        # Tokenize according to the given model's tokenizer 
        passage_tokens = self.tokenizer.tokenize (passage)
        answer_tokens = self.tokenizer.tokenize (answer)

        # Calculate total tokens to keep while Reserving tokens for [CLS], [SEP], and space
        tokens_to_keep = self.max_seq_len - len(answer_tokens) - 3 

        if len(passage_tokens) <= tokens_to_keep:
            self.tokenizer.convert_tokens_to_string (answer_tokens), self.tokenizer.convert_tokens_to_string (passage_tokens)
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

                    # Truncate context, including the answer
                    truncated_passage_tokens = passage_tokens[passage_start : passage_end]
                    truncated_passage = self.tokenizer.convert_tokens_to_string(truncated_passage_tokens)
                    return  self.tokenizer.convert_tokens_to_string (answer_tokens), truncated_passage
                except ValueError:
                    print (f"Answer: {answer} not found in given passage")


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

    def get_query_doc_pairs_from_squad_file(self, squad_file):

        """"
        Loads query-passage pairs from a SQuAD dataset file, including the step of truncating passages to fit the encoder's maximum sequence length
        
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

                    # truncate document to max_seq_length of encoder model
                    results = self.fit_passage_in_max_len (passage= passage, answer=answer)
                    if results is not None:
                        _, passage = results
                        query_doc_pairs.append ({"question": question, "document": passage})
                        pairs_count += 1
                    else:
                        continue

        print (f"Input query-doc pairs count: {pairs_count}")
        return query_doc_pairs
    
    def get_query_doc_pairs_from_file (self, filepath):
        """
        loads query passage pairs from a json file. The file must contain the data in the right format: [{"question": str, "document":str}]
        """
        with open(filepath, "r") as fp:
            
        
            data =  json.load(fp)

            truncated_data = []

            for qd_pair in tqdm(data):

                tokenized_doc = self.tokenizer.tokenize (qd_pair["document"])[:self.max_seq_len]
                truncated_data.append ({"question": qd_pair["question"], "document": self.tokenizer.convert_tokens_to_string(tokenized_doc)})

            return truncated_data


