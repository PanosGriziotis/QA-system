from haystack.utils import SquadData
import json
import sys
from nltk.tokenize import word_tokenize
import argparse

class GreekTokenizer:
    def __init__(self):
        pass  # No need for initialization with word_tokenize

    def tokenize(self, text):
        """
        Tokenizes Greek text into words using NLTK's word tokenizer.

        Args:
            text (str): The Greek text to be tokenized.

        Returns:
            list: A list of tokens extracted from the text.
        """
        tokens = word_tokenize(text, language='greek')
        return tokens

tokenizer = GreekTokenizer()

def get_all_answers(data):
    answers = []
    for article in data['data']:
        for paragraph in article["paragraphs"]:
            for qa in paragraph["qas"]:
                for answer in qa["answers"]:
                    answer = answer["text"]
                    answers.append(answer)
    return answers

def get_avg_tokens(list_of_texts):
    if not list_of_texts:
        return 0
    return sum(len(tokenizer.tokenize(text)) for text in list_of_texts) / len(list_of_texts)

def get_squad_dataset_counts(filename):
    with open(filename, "r", encoding="utf-8") as reader:
        data = json.load(reader)
        answers = get_all_answers(data)
        data = SquadData(data)
        num_of_examples = data.count()
        paragraphs = data.get_all_paragraphs()
        questions = data.get_all_questions()

    return {
        "filename": filename,
        "num_of_examples": num_of_examples,
        "num_of_paragraphs": len(paragraphs),
        "avg_q_len": get_avg_tokens(questions),
        "avg_ctx_len": get_avg_tokens(paragraphs),
        "avg_a_len": get_avg_tokens(answers)
    }

def get_who_dataset_count(filename):
    with open(filename, "r", encoding="utf-8") as fp:
        data = json.load(fp)
        questions = [d["question"] for d in data]
        answers = [d["document"] for d in data]
        num_of_examples = len(data)
        
        return {
            "filename": filename,
            "num_of_examples": num_of_examples,
            "avg_q_len": get_avg_tokens(questions),
            "avg_a_len": get_avg_tokens(answers)
        }

def detect_json_format(data):
    # Check for SQuAD format by looking for keys specific to SQuAD
    if 'data' in data and 'paragraphs' in data['data'][0]:
        return 'squad'
    # Check for WHO format by looking for the presence of 'question' and 'document'
    elif 'question' in data[0] and 'document' in data[0]:
        return 'who'
    else:
        raise ValueError("Unsupported JSON format")

if __name__ == "__main__":
    filename = sys.argv[1]
    
    with open(filename, "r", encoding="utf-8") as fp:
        data = json.load(fp)
    
    json_format = detect_json_format(data)
    
    if json_format == 'squad':
        result = get_squad_dataset_counts(filename)
    elif json_format == 'who':
        result = get_who_dataset_count(filename)
    else:
        raise ValueError("Unsupported JSON format")
    
    print(result)