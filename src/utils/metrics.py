
import os 
import sys
from typing import List, Union

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(SCRIPT_DIR, os.pardir, 'models_cache')
sys.path.append(os.path.dirname(SCRIPT_DIR))

from sentence_transformers import SentenceTransformer
from rouge_score import rouge_scorer
from transliterate import translit
import re
import numpy as np
from utils.data_handling import GreekTokenizer


def cosine_similarity(embedding_1, embedding_2):
    """
    Compute cosine similarity between two embeddings. Ensure that embeddings are 1D vectors.
    """
    # Flatten the embeddings to 1D arrays
    embedding_1 = embedding_1.flatten()
    embedding_2 = embedding_2.flatten()

    # Compute the dot product and norms
    return np.dot(embedding_1, embedding_2) / (np.linalg.norm(embedding_1) * np.linalg.norm(embedding_2))

def compute_similarity(query: str, documents: Union[str, List[str]], model_name_or_path="lighteternal/stsb-xlm-r-greek-transfer"):
    """
    Generates a cosine similarity score between a query and a list of concatenated documents using a bi-encoder model. 
    """

    biencoder_model = os.path.join(SCRIPT_DIR, f'models/{model_name_or_path.split("/")[1]}')
    if not os.path.exists(biencoder_model):
        biencoder_model = model_name_or_path

    model = SentenceTransformer(biencoder_model, cache_folder=CACHE_DIR, device="cpu")

    # Ensure documents is a list
    if isinstance(documents, str):
        documents = [documents]

    # Transliterate the query and documents (assuming Greek text handling)
    trans_query = translit(query.lower(), 'el')
    trans_docs = [translit(doc.lower(), 'el') for doc in documents]

    # Step 1: Batch Encoding for Embedding-based similarity ---
    all_texts = [trans_query] + trans_docs
    all_embeddings = model.encode(all_texts, convert_to_tensor=True, batch_size=8) 

    # Extract the query and document embeddings from the batch
    query_embedding = all_embeddings[0]  # Query embedding
    docs_embeddings = all_embeddings[1:]  # Document embeddings

    # Step 2: Compute cosine similarity ---
    cosine_similarities = []
    for doc_embedding in docs_embeddings:
        # Compute the cosine similarity between the query and document embeddings
        sim = cosine_similarity(query_embedding.cpu().numpy(), doc_embedding.cpu().numpy())
        cosine_similarities.append(sim)

    return cosine_similarities


def compute_context_relevance(query: str, documents: List[str]) -> float:
    """
    Computes context relevance by averaging similarity scores between a query and a set of concatenated documents.
    
    :param query: The input query string.
    :param documents: List of document strings to be considered as the context.
    :return: A context relevance score (average of similarity scores).
    """
    similarities = compute_similarity(query, documents)

    # If multiple similarities, compute the mean score
    if len(similarities) > 1:
        return np.mean(similarities)
    else:
        return similarities[0]
    
def generate_questions(answer, context):
    """Generates 3 artificial queries based on given answer text (and optionally based on a small context window for extractive qa answers spans) Important function for computing answer relevance score."""

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
    """"""
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
    similarities = compute_similarity(document_1=query, documents=generated_queries)
    return np.mean(similarities)

def compute_groundedness_rouge_score (answer:str, context:str):
    """
    Determines whether the output answer is grounded on the retrieved documents information using rouge-l precision score.
    """

    tokenizer = GreekTokenizer()
    scorer = rouge_scorer.RougeScorer(rouge_types = ['rougeL'], tokenizer=tokenizer)
    score = scorer.score(context, answer)["rougeL"].precision
    return score

