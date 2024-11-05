
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
from custom_components.generator import Generator

def cosine_similarity(embedding_1, embedding_2):
    """
    Compute cosine similarity between two embeddings. Ensure that embeddings are 1D vectors.
    """
    # Flatten the embeddings to 1D arrays
    embedding_1 = embedding_1.flatten()
    embedding_2 = embedding_2.flatten()

    # Compute the dot product and norms
    return np.dot(embedding_1, embedding_2) / (np.linalg.norm(embedding_1) * np.linalg.norm(embedding_2))

def compute_similarity(document: str, documents: Union[str, List[str]], model_name_or_path="lighteternal/stsb-xlm-r-greek-transfer"):
    
    """Generate a similarity score between a document (e.g., query) and a list of documents (or another single document) by using their embeddings"""

    biencoder_model = os.path.join(SCRIPT_DIR, f'models/{model_name_or_path.split("/")[1]}')
    if not os.path.exists(biencoder_model):
        biencoder_model = model_name_or_path

    model = SentenceTransformer(biencoder_model, cache_folder=CACHE_DIR, device="cpu")

    # Ensure documents is a list
    if isinstance(documents, str):
        documents = [documents]

    # Transliterate the query and documents (assuming Greek text handling)
    trans_query = translit(document.lower(), 'el')
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
    """Computes context relevance by averaging similarity scores between a query and a set of concatenated documents."""

    similarities = compute_similarity(query, documents)

    # If multiple similarities, compute the mean score
    if len(similarities) > 1:
        return float(np.mean(similarities))
    else:
        return similarities[0]
    
def compute_answer_relevance(generator:Generator, query: str, answer: str, context:str=None):
    """Compute answer relevance score based on the RAGAS framework: https://arxiv.org/abs/2309.15217"""

    def generate_questions():
        """Generates 3 artificial queries based on given answer text (optionally context can be provided)."""

        if context is not None:
            prompt_messages = [
                {"role": "system", "content": 'Δημιούργησε ερωτήσεις με βάση την απάντηση που σου δίνει ο χρήστης. Για να σε βοηθήσω, σου δίνω κάποια παραδείγματα: \n Απάντηση: Ο Άλμπερτ Αϊνστάιν γεννήθηκε στη Γερμανία. \nΠερικείμενο: Ο Άλμπερτ Αϊνστάιν ήταν ένας γερμανικής καταγωγής θεωρητικός φυσικός, ο οποίος θεωρείται ευρέως ως ένας από τους μεγαλύτερους και πιο επιδραστικούς επιστήμονες όλων των εποχών \n Ερωτήσεις:\n1. Πού γεννήθηκε ο Άλμπερτ Αϊνστάιν;\n2. Ποιά είναι η χώρα γέννησης του Αϊνστάιν;\n3. Σε ποιά χώρα γεννήθηκε Αϊνστάιν; \n\n Απάντηση: Everest \n Περικείμενο: Το ψηλότερο βουνό στη Γη, μετρούμενο από το επίπεδο της θάλασσας, είναι μια διάσημη κορυφή που βρίσκεται στα Ιμαλάια. \n Ερωτήσεις:\n1. Ποιο είναι το ψηλότερο βουνό στη Γη;\n2. Πώς ονομάζεται το πιο ψηλό βουνό στον κόσμο;\n3. Ποιό βουνό θεωρείται αυτό με το πιο μεγάλο υψόμετρο;'},
                {"role": "user", "content": f'Δημιούργησε 3 σύντομες και απλές ερωτήσεις με βάση την απάντηση. \n Απάντηση: {answer} \n Περικείμενο: {context} \n Ερωτήσεις: \n '}
            ]
        else:
            prompt_messages = [
                {"role": "system", "content": 'Δημιούργησε ερωτήσεις με βάση την απάντηση που σου δίνει ο χρήστης. Για να σε βοηθήσω, σου δίνω κάποια παραδείγματα: \nΑπάντηση: Ο Άλμπερτ Αϊνστάιν γεννήθηκε στη Γερμανία. \n1. Πού γεννήθηκε ο Άλμπερτ Αϊνστάιν;\n2. Ποιά είναι η χώρα γέννησης του Αϊνστάιν;\n3. Σε ποιά χώρα γεννήθηκε Αϊνστάιν; \n\n Απάντηση: Everest\n Ερωτήσεις:\n1. Ποιο είναι το ψηλότερο βουνό στη Γη;\n2. Πώς ονομάζεται το πιο ψηλό βουνό στον κόσμο;\n3. Ποιό βουνό θεωρείται αυτό με το πιο μεγάλο υψόμετρο;'},
                {"role": "user", "content": f'Δημιούργησε 3 σύντομες και απλές ερωτήσεις με βάση την απάντηση. \n Απάντηση: {answer} \n Περικείμενο: {context} \n Ερωτήσεις: \n '}
            ]

        # Initialize generator class to implement query generation
        generator.set_prompt_messages(prompt_messages=prompt_messages) 
        result, _ = generator.run(query=answer, documents=[], max_new_tokens=100, post_processing=False)

        # Extract the matches (get generated questions into a list)
        pattern = r'\s*(1\.|2\.|3\.)\s(.*?)(?=\n\s*(1\.|2\.|3\.)|\n*$)'
        matches = re.findall(pattern, result["answers"][0].answer)

        # Extract only the questions (ignore the numbering)
        generated_queries = [match[1] for match in matches]

        return generated_queries

    generated_queries = []
    max_attempts = 5  # Set a limit to avoid infinite loops
    attempt = 0

    while len(generated_queries) < 3 and attempt < max_attempts:
        generated_queries = generate_questions()
        attempt += 1

    if len(generated_queries) < 3:
       raise ValueError(f"Failed to generate 3 questions after {attempt} attempts")

    similarities = compute_similarity(document=query, documents=generated_queries)

    return float(np.mean(similarities))

def compute_groundedness_rouge_score (answer:str, context:str):
    """Determines whether the output answer is grounded on the retrieved documents information using rouge-l precision score."""

    tokenizer = GreekTokenizer()
    scorer = rouge_scorer.RougeScorer(rouge_types = ['rougeL'], tokenizer=tokenizer)
    score = scorer.score(context, answer)["rougeL"].precision
    return score

