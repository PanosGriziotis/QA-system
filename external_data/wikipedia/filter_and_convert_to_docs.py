#!/usr/bin/env python3
# Apply  pre-processing and domain filtering (optional) on fetched wikipedia dump articles
# input file: each line should be a wikipedia article in JSON format

import logging
import tempfile
import argparse
import json
import os
from pathlib import Path

from haystack.pipelines import Pipeline
from haystack.nodes import PreProcessor, JsonConverter

from clean_wiki import preprocess_wiki_docs
from keyword_filtering import KeywordFilterer

logging.basicConfig(format="%(levelname)s - %(name)s -  %(message)s", level=logging.INFO)
logging.getLogger("haystack").setLevel(logging.INFO)

PREPROCESS_FN = None
DOMAIN_FILTERING = False

def init (preprocess=False, apply_filter=False):
    global PREPROCESS_FN
    global DOMAIN_FILTERING
    if preprocess:
        PREPROCESS_FN = preprocess_wiki_docs
    if apply_filter:
        DOMAIN_FILTERING = True

def load_docs (filename):
    """Open .json file with documents as lines"""

    with open (filename, "r") as fp:
        return [json.loads(line) for line in fp.readlines()]

def preporocess_wiki_files (file_paths):
    """pipeline for cleaning and converting wiki data to haystack Document objects.
        If --apply_filter, an extra node for keeping only keyword related articles is added to the pipeline"""
    
    global FILTERING

    preprocessor = PreProcessor(
        clean_empty_lines=True,
        split_by = "word",
        split_length=256,
        split_respect_sentence_boundary=True,
        language= 'el'
        )
    
    #document_store = ElasticsearchDocumentStore(host="localhost", username="", password="", index=index, recreate_index=True)
    p = Pipeline()
    p.add_node(component=JsonConverter(), name = "JsonConverter", inputs=["File"])
    p.add_node(component=preprocessor, name="Preprocessor", inputs=["JsonConverter"])

    if DOMAIN_FILTERING:
        p.add_node(component=KeywordFilterer(), name= "KeywordFilterer", inputs =["Preprocessor"])
    #p.add_node(component=document_store, name="DocumentStore", inputs=previous_node)
    
    docs= p.run_batch(file_paths=file_paths)
    return docs["documents"]


def main (data_filename):
    global PREPROCESS_FN
    """Index wiki documents in ES doument store

    Args: 
    data_filename: file containing all extracted wiki articles

    """
    if not os.path.isfile(data_filename):
        raise RuntimeError('%s This is not a path to a filename' % data_filename)
    
    if PREPROCESS_FN is not None:
        documents = PREPROCESS_FN(data_filename)
    else:
        documents = load_docs(data_filename)
    
    temp_dir = tempfile.TemporaryDirectory()

    file_paths = []
    for doc in documents:
        file_name = doc["id"] + ".json"
        file_path = Path(temp_dir.name) / file_name
        file_paths.append(str(file_path))
        with open(file_path, "w") as f:
            f.write(json.dumps(doc, ensure_ascii=False))

    return preporocess_wiki_files(file_paths)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('data_filename', type=str, help='/path/to/data_filename')
    parser.add_argument('--preprocess', action='store_true',
                        help='Preprocess wiki documents')
    parser.add_argument('--apply_filter', action='store_true',
                        help='filter wiki documents by applying keyword matching')
    args = parser.parse_args()

    init(preprocess=args.preprocess, apply_filter=args.apply_filter)
    
    save_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

 
    docs =  main(
        args.data_filename,
    )

    with open (os.path.join(save_dir, 'wiki_docs.json'), "w") as fp:
        for doc in docs:
            fp.write (json.dumps (doc.to_dict(), ensure_ascii=False) + '\n')