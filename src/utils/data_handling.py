from typing import List
from langdetect import detect, DetectorFactory
from langdetect.lang_detect_exception import LangDetectException
import re
import nltk
from metrics import compute_context_relevance, compute_similarity, compute_answer_relevance, compute_groundedness_rouge_score
import re
from haystack.nodes import TransformersTranslator
import string

def post_process_generator_answers (result):
    """ 
    Post-process Generator's output: 
        a) remove trailing words in incomplete outputs that end abruptly.
        b) remove words or phrases in the output coming accidently from the prompt (e.g., Ερώτηση: )    
    """
    
    #answer = result ["answers"][0]["answer"]

    answer = result["answers"][0].answer
    answer = remove_prompt_words(answer)
    
    answer = truncate_incomplete_sentence(answer)
   
    if answer == '' or (len(answer) == 1 and answer in string.punctuation):
        answer = result["answers"][0].answer = "Συγγνώμη, αλλά δεν μπορώ να δώσω μία απάντηση σε αυτήν την ερώτηση."
        #answer = result["answers"][0]["answer"] = "Συγγνώμη, αλλά δεν μπορώ να δώσω μία απάντηση σε αυτήν την ερώτηση."

    result ["answers"][0].answer = answer
    #result ["answers"][0]["answer"] = answer

    return result

def truncate_incomplete_sentence(text):
    """
    Truncate the text at the last complete sentence.
    
    Args:
        text (str): The text to be truncated.
    
    Returns:
        str: The truncated text with only complete sentences.
    """
    # Replace newline characters with spaces
    text = text.replace('\n', ' ')

    # Split the text into sentences (we assume it's in Greek)
    sentences = nltk.sent_tokenize(text, language='greek')
    
    # Greek punctuation marks that indicate the end of a sentence
    ending_punctuation = ('.', '!', ';')

    # Pattern to detect sentences like "1." or "2)" that are just enumerations
    enumeration_pattern = re.compile(r'^\s*\d+[.)]\s*$')

    # Process each sentence
    while sentences:
        last_sentence = sentences[-1].strip()  # Check the last sentence

        # If it's just an enumeration like "1.", remove it
        if enumeration_pattern.match(last_sentence):
            sentences.pop()
            continue

        # If the sentence ends with valid punctuation, we're done
        if last_sentence.endswith(ending_punctuation):
            break
        
        # Otherwise, remove incomplete sentences
        sentences.pop()

    # Join the remaining complete sentences
    truncated_text = ' '.join(sentences)
    return truncated_text.strip()

def remove_prompt_words (text):
    # Check if "Απάντηση: " appears
    text_match_2 = re.search(r'Απάντηση:\s*(.*)', text)
    
    if text_match_2:
        return text_match_2.group(1)  # Return the text after "Απάντηση: "
    else:
        return text  

def add_eval_scores_to_result(result):

    query = result["query"]
    answer_objs = result["answers"]
    retrieved_documents = result["documents"]

    first_answer = answer_objs[0]
    answer_text = first_answer["answer"]
    answer_type = first_answer["type"]
    
    # Define context
    if answer_type == "extractive":
        context = first_answer["context"]
    elif answer_type == "generative":
        context = " ".join([document["content"] for document in retrieved_documents])


    # Context relevance
    if "context_relevance" not in first_answer["meta"]:
        context_relevance = compute_similarity(document_1=answer_text, document_2=context)
        first_answer["meta"]["context_relevance"] = context_relevance

    # Handle no answer strings
    if answer_text == "Συγγνώμη, αλλά δεν μπορώ να δώσω μία απάντηση σε αυτήν την ερώτηση.":
        first_answer["meta"]["answer_accuracy"] = 0.0
        first_answer["meta"]["answer_relevance_ragas"] = 0.0
        first_answer["meta"]["groundedness"] = 0.0 if answer_type == "rag" else None 
    else:
        # 1) Answer accuracy
        if "answer_accuracy" not in first_answer["meta"]:
            answer_accuracy = compute_similarity(document_1=query, document_2=answer_text)
            first_answer["meta"]["answer_accuracy"] = answer_accuracy

        # 2) Answer relevance ragas score
        answer_relevance_ragas = compute_answer_relevance(query=query, answer=answer_text, context=context)
        first_answer["meta"]["answer_relevance_ragas"] = answer_relevance_ragas

        # 3) Groundedness score (only rag)
        if answer_type == "generative":
            first_answer["meta"]["groundedness"]  = compute_groundedness_rouge_score(
                answer=answer_text,
                context=" ".join([document["content"] for document in retrieved_documents])
            )
        elif answer_type == "extractive":
            first_answer["meta"]["groundedness"] = None 

    result["answers"][0] = first_answer

    return result

def is_english(text):
    DetectorFactory.seed = 0
    try:
        return detect(text) == 'en'
    except LangDetectException:
        return False

def remove_english_text(lines):
    non_english_lines = []
    for line in lines:
        if not is_english(line):
            non_english_lines.append(line)
    return non_english_lines


def is_english(text):
    DetectorFactory.seed = 0
    try:
        return detect(text) == 'en'
    except LangDetectException:
        return False

def translate_docs (docs:List[str], use_gpu:bool=False):
    
    max_seq_len = 512
    translator = TransformersTranslator(model_name_or_path="Helsinki-NLP/opus-mt-en-el", use_gpu=use_gpu, max_seq_len=max_seq_len)
    #c_docs = clean_and_split_docs(docs, max_seq_len=max_seq_len)
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



