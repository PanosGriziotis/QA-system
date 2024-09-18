from haystack.utils import SquadData
import json
import sys

def tokenize (text):
    
    #tokenizer = MosesTokenizer(lang='el')
    return text.split (" ")
    #return     tokenizer.tokenize(text.strip(), return_str=True)


def get_all_answers (data):
    answers = []
    for article in data['data']:
        for paragraph in article ["paragraphs"]:
            for qa in paragraph["qas"]:
                for answer in qa["answers"]:
                    answer = answer["text"]
                    answers.append(answer)
    return answers

def get_avg_tokens (list_of_texts):
    return (sum ([len(tokenize(text)) for text in list_of_texts])) / (len (list_of_texts))

def get_squad_dataset_counts (filename):
    with open (filename, "r", encoding= "utf-8") as reader:
        data = json.load(reader)
        answers = get_all_answers(data)
        data =  SquadData (data)
        num_of_examples = data.count()
        paragraphs = data.get_all_paragraphs()
        questions = data.get_all_questions()

    return {
        "filename": filename,
        "num_of_examples": num_of_examples,
        "num_of_paragraphs": len(paragraphs),
        "avg_q_len": (get_avg_tokens(questions)),
        "avg_ctx_len": (get_avg_tokens(paragraphs)),
        "avg_a_len": (get_avg_tokens(answers))
        }

if __name__ == "__main__":
    filename = sys.argv[1]
    print (get_squad_dataset_counts(filename))