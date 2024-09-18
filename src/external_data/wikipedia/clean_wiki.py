#!/usr/bin/env python3

import argparse
import json
import os
import logging
import time

from tqdm import tqdm

import regex as re
import unicodedata
import html

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)

def normalize(text):
    """Normalize text"""
    return unicodedata.normalize('NFC', text.replace('\n', ' '))

def remove_structured_pages(article):
    """Remove disambiguation, list, category and prototype wiki pages."""

    for k, v in article.items():      
        article[k] =  html.unescape(normalize(v))

    if '(αποσαφήνιση)'in article['title'].lower():
        return None

    if re.match(r'(Κατάλογος .+)|(Κατηγορία:.+)|(Πρότυπο:.+)',
                article['title']):
        return None

    return {'id': article['id'], 'title': article['title'], 'content': article['text']}

def preprocess_wiki_docs(filename):
    """"""
    
    logging.info(f"Starting to preprocess file: {filename}")
    start_time = time.time()

    documents = []
    with open(filename, "r") as f:
        for line in tqdm(f.readlines()):
            doc = remove_structured_pages(json.loads(line))
            if not doc:
                continue
            documents.append(doc)
    
    elapsed_time = time.time() - start_time
    logging.info(f"Finished preprocessing file: {filename} in {elapsed_time:.2f} seconds")
    logging.info (f"Number of wiki articles after preprocessing: {len(documents)}")

    return documents