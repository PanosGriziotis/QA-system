
import os 
import sys
from typing import List, Union
from haystack.schema import Document



SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from sentence_transformers import SentenceTransformer, CrossEncoder

from rouge_score import rouge_scorer
from nltk.tokenize import word_tokenize

from haystack.schema import Document
import re

def compute_similarity (document_1:str, document_2:Union[str, List[str]], model_name_or_path="lighteternal/stsb-xlm-r-greek-transfer"):
    """
    Generates Cosine Similarity scores between query and a document (or a list of documents) using a Bi-encoder and cosine similarity.
    """
    # Load Bi-encoder
    try:
        model_name_or_path = os.path.join(SCRIPT_DIR, "models/stsb-xlm-r-greek-transfer")
    except IOError:
        model_name_or_path = "lighteternal/stsb-xlm-r-greek-transfer"

    model = SentenceTransformer(model_name_or_path, device= "cuda")
    q_embedding = model.encode(document_1)
    d_embedding = model.encode(document_2)
    
    cos_sim = model.similarity(q_embedding, d_embedding)[0].tolist()

    if len(cos_sim) > 1:
        return cos_sim
    else:
        return cos_sim[0]   

def compute_context_relevance (answer, retrieved_documents):
    """
    Determines the relevance of the documents from which the answer was generated or extracted relative to the query.
    This method calculates the mean relevance score of the documents that comprise the input context. 
    In the case of an extractive QA pipeline, it returns a single relevance score, coming from the document that the 
    answer was extracted from.

    Args:
        answer (Answer): The first Answer object in the result, which contains the extracted answer.
        retrieved_documents (list of Document): A list of ranked Document objects, representing the documents retrieved for the query.
    """
    # Get the ids of the documents that the answer is generated from 
    scores = []
    answer_documents_ids = answer["document_ids"]

    for doc in retrieved_documents:
        if doc["id"] in answer_documents_ids:
            scores.append(doc["score"])
    
    if len (scores) > 1:
        return sum (scores) / len (scores)
    else: 
        return scores[0]

def compute_answer_relevance(query: str, answer: str, context: str):

    from pipelines.query_pipelines import Generator

    prompt_messages = [
        {"role": "system", "content": 'Είσαι το Μελτέμι, ένα γλωσσικό μοντέλο για την ελληνική γλώσσα. Είσαι ικανό να δημιουργείς ερωτήσεις που αντιστοιχούν στις απαντήσεις που σου δίνει ο χρήστης και σχετίζονται με την πανδημία του COVID-19.'},
        {"role": "user", "content": f'Δώσε μου 3 σύντομες και απλοϊκές ερωτήσεις με βάση αυτήν την απάντηση: "{answer}"'}
    ]

    generator = Generator(prompt_messages=prompt_messages)

    if not answer:
        
        answer_relevance = 0.0
        return answer_relevance

    result, _ = generator.run(query=answer, documents=[Document(content=context)], max_new_tokens=100, post_processing=False)
    
    # Extract the matches (get generated questions into a list)
    pattern = r'\d+\.\s(.*?)(?=\n\d+\.|\n*$)'
    generated_queries = re.findall(pattern, result["answers"][0].answer, re.DOTALL)
    print (generated_queries)
    if len(generated_queries) < 3:
        answer_relevance = 0.0
        #raise ValueError(f"Less than 3 questions were generated for answer '{answer}' and ground truth question '{query}'")
    
    else:
        scores = []

        for generated_query in generated_queries:
            score = compute_similarity (document_1=generated_query, document_2=query)
            scores.append(score)

        try:
            answer_relevance = sum(scores) / len(scores)

        except ZeroDivisionError:
            answer_relevance = 0.0
            print(f"Error: Division by zero occurred when computing relevance for query '{query}' and answer '{answer}'. Returning None.")

        except Exception as e:
            print(f"An unexpected error occurred: {str(e)}")
            answer_relevance = 0.0

    return answer_relevance

############### UNUSED FUNCTIONS #############

def cross_encoder_similarity (sentences,query):
    model = CrossEncoder(model_name="amberoad/bert-multilingual-passage-reranking-msmarco",  max_length=512)
    scores = []
    for sentence in sentences:
        score = model.predict([query, sentence])
        scores.append(score)
    return scores

def compute_groundedness_score (answer:str, retrieved_documents:List[str]):
    """
    Determines whether the output answer is grounded on the retrieved documents information using cosine similarity.
    """
    scores = compute_similarity(query=answer, document=retrieved_documents)

    return sum (scores)/ len(retrieved_documents)

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
    
