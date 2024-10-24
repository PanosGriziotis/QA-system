from typing import List
from langdetect import detect, DetectorFactory
from langdetect.lang_detect_exception import LangDetectException
import re
import nltk
import re
from haystack.nodes import TransformersTranslator
import string
import gc
import torch

def post_process_generator_answers (result):
    """ 
    Post-process Generator's output: 
        a) remove trailing words in incomplete outputs that end abruptly.
        b) remove words or phrases in the output coming accidently from the prompt (e.g., Απάντηση: )    
    """
    
    answer = result["answers"][0].answer
    
    answer = remove_prompt_words(answer)
    answer = truncate_incomplete_sentence(answer)
   
    if answer == '' or (len(answer) == 1 and answer in string.punctuation):
        answer = result["answers"][0].answer = "Συγγνώμη, αλλά δεν μπορώ να δώσω μία απάντηση σε αυτήν την ερώτηση."

    result ["answers"][0].answer = answer

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

    # Process each sentence
    while sentences:
        last_sentence = sentences[-1].strip()  # Check the last sentence

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


def translate_docs (docs:List[str], use_gpu:bool=False):
    
    max_seq_len = 512
    translator = TransformersTranslator(model_name_or_path="Helsinki-NLP/opus-mt-en-el", use_gpu=use_gpu, max_seq_len=max_seq_len)
    #c_docs = clean_and_split_docs(docs, max_seq_len=max_seq_len)
    try:
        t_docs = translator.translate_batch(documents=docs)[0]
    except AttributeError:
        t_docs = ['<ukn>']
    return t_docs

def flash_cuda_memory():
    gc.collect()
    torch.cuda.empty_cache()



