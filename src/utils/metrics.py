
import os 
import sys
from typing import List, Union
from haystack.schema import Document

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from sentence_transformers import SentenceTransformer

from rouge_score import rouge_scorer
from nltk.tokenize import word_tokenize

import re

def compute_similarity (document_1:str, document_2:Union[str, List[str]], model_name_or_path="lighteternal/stsb-xlm-r-greek-transfer"):
    """
    Generates Cosine Similarity scores between query and a document (or a list of documents) using a Bi-encoder and cosine similarity.
    """
    # Load Bi-encoder
    
    model_name_or_path = os.path.join(SCRIPT_DIR, "models/stsb-xlm-r-greek-transfer")
    if not os.path.exists(model_name_or_path):

        model_name_or_path = "lighteternal/stsb-xlm-r-greek-transfer"

    model = SentenceTransformer(model_name_or_path, device= "cuda")
    embedding_1 = model.encode(document_1)
    embedding_2 = model.encode(document_2)
    
    cos_sim = model.similarity(embedding_1, embedding_2)[0].tolist()

    return cos_sim[0]   


def generate_questions(answer, context):

    from pipelines.query_pipelines import Generator

    answer_text = answer
    context_text = context
    prompt_messages = [
        {"role": "system", "content": 'Δημιούργησε ερωτήσεις με βάση την απάντηση που σου δίνει ο χρήστης. Για να σε βοηθήσω, σου δίνω κάποια παραδείγματα: \n Απάντηση: Ο Άλμπερτ Αϊνστάιν γεννήθηκε στη Γερμανία. \nΠερικείμενο: Ο Άλμπερτ Αϊνστάιν ήταν ένας γερμανικής καταγωγής θεωρητικός φυσικός, ο οποίος θεωρείται ευρέως ως ένας από τους μεγαλύτερους και πιο επιδραστικούς επιστήμονες όλων των εποχών \n Ερωτήσεις:\n 1. Πού γεννήθηκε ο Άλμπερτ Αϊνστάιν;\n 2. Ποιά είναι η χώρα γέννησης του Αϊνστάιν;\n 3. Σε ποιά χώρα γεννήθηκε Αϊνστάιν; \n\n Απάντηση: Everest \n Περικείμενο: Το ψηλότερο βουνό στη Γη, μετρούμενο από το επίπεδο της θάλασσας, είναι μια διάσημη κορυφή που βρίσκεται στα Ιμαλάια. \n Ερωτήσεις:\n 1. Ποιο είναι το ψηλότερο βουνό στη Γη;\n 2. Πώς ονομάζεται το πιο ψηλό βουνό στον κόσμο;\n 3. Ποιό βουνό θεωρείται αυτό με το πιο μεγάλο υψόμετρο;'},
        {"role": "user", "content": f'Δημιούργησε 3 σύντομες και απλές ερωτήσεις με βάση την απάντηση. \n Απάντηση: {answer_text} \n Περικείμενο: {context_text} \n Ερωτήσεις: \n '}
    ]
    generator = Generator(prompt_messages=prompt_messages)

    result, _ = generator.run(query=answer_text, documents=[], max_new_tokens=100, post_processing=False)

    # Extract the matches (get generated questions into a list)
    pattern = r'(1\.|2\.|3\.)\s(.*?)(?=\n(1\.|2\.|3\.)|\n*$)'
    matches = re.findall(pattern, result["answers"][0].answer)

    # Extract only the questions (ignore the numbering)
    generated_queries = [match[1] for match in matches]

    return generated_queries
    
def compute_answer_relevance(query: str, answer: str, context):

    # Keep generating questions until we get at least 3
    generated_queries = []
    max_attempts = 5  # Set a limit to avoid infinite loops
    attempt = 0

    while len(generated_queries) < 3 and attempt < max_attempts:
        generated_queries = generate_questions(answer, context)

        attempt += 1

    # If we still don't have 3 questions after retries, set relevance to 0
    if len(generated_queries) < 3:
        print(f"Failed to generate 3 questions after {attempt} attempts")
        return 0.0

    # Compute relevance if 3 questions are generated
    scores = []
    for generated_query in generated_queries:
        score = compute_similarity(document_1=generated_query, document_2=query)
        scores.append(score)

    try:
        answer_relevance = sum(scores) / len(scores)
    except ZeroDivisionError:
        print(f"Error: Division by zero occurred when computing relevance for query '{query}' and answer '{answer}'. Returning None.")
        answer_relevance = 0.0
    except Exception as e:
        print(f"An unexpected error occurred: {str(e)}")
        answer_relevance = 0.0

    return answer_relevance


def compute_context_relevance (query:str, retrieved_documents:List):

    if len(retrieved_documents) != 0:
        scores = []

        for doc in retrieved_documents:
            cos_sim = compute_similarity(document_1=query, document_2=doc)
            scores.append(cos_sim)
        
        return sum(scores)/len(scores)
    else: 
        cos_sim = compute_similarity(document_1=query, document_2=retrieved_documents[0])
        return cos_sim


def compute_groundedness_rouge_score (answer:str, context:str):
    """
    Determines whether the output answer is grounded on the retrieved documents information using rouge score.
    Rouge-L precision is chosen because it measures how much of the answer's content is directly derived from the reference context. 
    By using precision, we avoid the bias where longer answers might artificially achieve higher scores simply due to their length, regardless of their relevance or accuracy.
    """

    tokenizer = GreekTokenizer()
    scorer = rouge_scorer.RougeScorer(rouge_types = ['rougeL'], tokenizer=tokenizer)
    score = scorer.score(context, answer)["rougeL"].precision
    return score
    
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
    
