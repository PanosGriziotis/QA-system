"""
Script to convert a SQuAD-like QA-dataset format JSON file to DPR Dense Retriever training format
"""

from typing import Dict, Iterator, Tuple, List, Union
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(SCRIPT_DIR)))

import requests
import json
import logging
import argparse
from pathlib import Path
from itertools import islice
import random
from tqdm import tqdm

from haystack.document_stores.elasticsearch import ElasticsearchDocumentStore 
from haystack.nodes.retriever.sparse import BM25Retriever  
from haystack.nodes.preprocessor import PreProcessor
from haystack.nodes.retriever.base import BaseRetriever
from haystack.schema import Document
from transformers import AutoTokenizer
from data_handling import DataExtractor, fit_passage_in_max_len

logger = logging.getLogger(__name__)


def add_is_impossible(squad_data: dict, json_file_path: Path):
    new_path = json_file_path.parent / Path(f"{json_file_path.stem}_impossible.json")
    squad_articles = list(squad_data["data"])  # create new list with this list although lists are inmutable :/
    for article in squad_articles:
        for paragraph in article["paragraphs"]:
            for question in paragraph["qas"]:
                question["is_impossible"] = False

    squad_data["data"] = squad_articles
    with open(new_path, "w", encoding="utf-8") as filo:
        json.dump(squad_data, filo, indent=4, ensure_ascii=False)

    return new_path, squad_data


def get_number_of_questions(squad_data: dict, qa_pairs:dict):
    nb_questions = 0
    for article in squad_data:
        for paragraph in article["paragraphs"]:
            nb_questions += len(paragraph["qas"])
    
    nb_questions +=len(qa_pairs)
    return nb_questions


def has_is_impossible(squad_data: dict):
    for article in squad_data["data"]:
        for paragraph in article["paragraphs"]:
            for question in paragraph["qas"]:
                if "is_impossible" in question:
                    return True
    return False


def extract_data_from_squad(squad_data:dict):
    extracted_data = []
    for article in tqdm(squad_data, unit="article"):
        #article_title = article.get("title", "")
        for paragraph in article["paragraphs"]:
            context = paragraph["context"]
            for question in paragraph["qas"]:
                # get first answers from query (in out datasets we dont have multiple answers)
                answer = [a["text"] for a in question["answers"]][0]
                data = {"question": question["question"], "document": answer, "context": context}
                extracted_data.append(data)
    return extracted_data

def create_dpr_training_dataset(extracted_qa_pairs: dict, retriever: BaseRetriever,tokenizer:AutoTokenizer, num_hard_negative_ctxs: int = 30, max_seq_len_passage: int = 512):
    n_non_added_questions = 0
    n_questions = 0

    for data in tqdm(extracted_qa_pairs, desc="Creating positive and negative passages from input data files..."):
        question = data["question"]
        answer = data ["document"]

        # Retrieve hard negatives from document store
        hard_negative_ctxs = get_hard_negative_contexts(
            retriever=retriever, question=question, answers=[answer], n_ctxs=num_hard_negative_ctxs
        )

        # for squad data:
        if "context" in list(data.keys()):
            context = data["context"]
            # Get positive document by spotting the answer in context and getting text before and after within a context window of max_seq_len_passage length
            try:
                answer, pos_ctx = fit_passage_in_max_len(passage=context, answer=answer, max_seq_len=max_seq_len_passage, tokenizer=tokenizer)
            except TypeError:
                print (answer)   
        # for who data the pos_document is just the plain answer. 
        else:
            pos_ctx = answer
        # create positive context document dict
        positive_ctxs = [{"title": "", "text": pos_ctx, "passage_id": ""}]

        if not hard_negative_ctxs:  
            logger.error(
                "No retrieved hard negative candidates for question %s", question
            )

        elif not positive_ctxs:
            logger.error(
                "No retrieved positive ctx candidates for question %s", question
            )
            n_non_added_questions += 1
            continue

        dict_DPR = {
            "question": question,
            "answers": [answer],
            "positive_ctxs": positive_ctxs,
            "negative_ctxs": [],
            "hard_negative_ctxs": hard_negative_ctxs,
        }
        n_questions += 1
        yield dict_DPR

    logger.info("Number of skipped questions: %s", n_non_added_questions)
    logger.info("Number of added questions: %s", n_questions)


def save_dataset(iter_dpr: Iterator, dpr_output_filename: Path, total_nb_questions: int, split_dataset: bool):
    if split_dataset:
        nb_train_examples = int(total_nb_questions * 0.9)
        #nb_dev_examples = int(total_nb_questions * 0.1)

        train_iter = islice(iter_dpr, nb_train_examples)
        #dev_iter = islice(iter_dpr, nb_dev_examples)

        dataset_splits = {
            dpr_output_filename.parent / f"{dpr_output_filename.stem}.train.json": train_iter,
            dpr_output_filename.parent / f"{dpr_output_filename.stem}.dev.json": iter_dpr,
        }
    else:
        dataset_splits = {dpr_output_filename: iter_dpr}
    for path, set_iter in dataset_splits.items():
        with open(path, "w", encoding="utf-8") as json_ds:
            json.dump(list(set_iter), json_ds, indent=4, ensure_ascii=False)


def get_hard_negative_contexts(retriever: BaseRetriever, question: str, answers: List[str], n_ctxs: int = 2):
    top_k = 20
    if n_ctxs > top_k:
        logger.error("Number of retrieved hard negative contexts can't be smaller than the number of hard negative contexts included in the final dataset")
    list_hard_neg_ctxs = []
    retrieved_docs = retriever.retrieve(query=question, top_k=top_k, index="dpr")
    for retrieved_doc in retrieved_docs:
        retrieved_doc_id = retrieved_doc.meta.get("name", "")
        retrieved_doc_text = retrieved_doc.content
        if any(str(answer).lower() in retrieved_doc_text.lower() for answer in answers):

            continue
        list_hard_neg_ctxs.append({"title": retrieved_doc_id, "text": retrieved_doc_text, "passage_id": ""})

    return list_hard_neg_ctxs[:n_ctxs]


def load_squad_file(squad_file_path: Path):
    if not squad_file_path.exists():
        raise FileNotFoundError

    with open(squad_file_path, encoding="utf-8") as squad_file:
        squad_data = json.load(squad_file)

    if not has_is_impossible(squad_data=squad_data):
        squad_file_path, squad_data = add_is_impossible(squad_data, squad_file_path)

    return squad_file_path, squad_data["data"]


def main(
    squad_input_filename: Path,
    qa_pairs_input_filename:Path,
    dpr_output_filename: Path,
    max_seq_len:int,
    tokenizer_model_name="nlpaueb/bert-base-greek-uncased-v1",
    num_hard_negative_ctxs: int = 30,
    split_dataset: bool = False,
):

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_model_name)

    # Load and prepare data
    squad_file_path, squad_data = load_squad_file(squad_file_path=squad_input_filename)
    extracted_squad_data = extract_data_from_squad(squad_data=squad_data)
    data_extractor = DataExtractor(tokenizer=tokenizer, max_seq_len=300)
    who_qa_pairs = data_extractor.get_query_doc_pairs_from_json_file(filepath=qa_pairs_input_filename)
    qa_data = who_qa_pairs + extracted_squad_data
    random.shuffle(qa_data)

    # 2. Initialize document store and retriever and index documents
    requests.delete("http://localhost:9200/*")
    document_store = ElasticsearchDocumentStore()
    preprocessor = PreProcessor(split_length=max_seq_len, clean_empty_lines=False, clean_whitespace=False, split_respect_sentence_boundary=False)
    document_store.add_eval_data(squad_file_path.as_posix(), doc_index="dpr", preprocessor=preprocessor)
    document_store.write_documents(documents=[Document(content=qa_pair["document"]) for qa_pair in who_qa_pairs], index="dpr")
    retriever = BM25Retriever(document_store=document_store)

    # Create and save DPR dataset
    iter_DPR = create_dpr_training_dataset(
        extracted_qa_pairs=qa_data, retriever=retriever, num_hard_negative_ctxs=num_hard_negative_ctxs, max_seq_len_passage=max_seq_len, tokenizer=tokenizer)
    total_nb_questions = get_number_of_questions(squad_data, who_qa_pairs)
    
    save_dataset(
        iter_dpr=iter_DPR,
        dpr_output_filename=dpr_output_filename,
        total_nb_questions=total_nb_questions,
        split_dataset=split_dataset,
    )

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Convert a SQuAD JSON format dataset to DPR format.")
    parser.add_argument(
        "--squad_file",
        dest="squad_input_filename",
        help="A dataset with a SQuAD JSON format.",
        metavar="SQUAD_in",
        default= os.path.join (SCRIPT_DIR,"../training/data/covid_QA_el_small/COVID-QA-el_small.json" )
    )
    parser.add_argument(
        "--qa_pairs_file",
        dest="qa_pairs_input_filename",
        help="A file with simple question-answer pairs.",
        metavar="QA_pairs in",
        default= os.path.join (SCRIPT_DIR,"../training/data/who_pairs.json" )
    )
    parser.add_argument(
        "--output_file",
        dest="dpr_output_filename",
        help="The name of the DPR JSON formatted output file",
        metavar="DPR_out",
        default= os.path.join (SCRIPT_DIR,"../training/data/dpr_full.json" )
    )
    parser.add_argument(
        "--num_hard_negative_ctxs",
        dest="num_hard_negative_ctxs",
        help="Number of hard negative contexts to use",
        metavar="num_hard_negative_ctxs",
        default=2
    )
    parser.add_argument(
        "--max_seq_len_passage",
        dest= "max_seq_len_passage",
        help= "Maximum token numbers of passages",
        default= 300
    )
    
    parser.add_argument(
        "--split_dataset",
        dest="split_dataset",
        action="store_true",
        help="Whether to split the created dataset or not (default: False)",
    )

    args = parser.parse_args()

    preprocessor = PreProcessor(
        split_length=300,
        split_overlap=0,
        clean_empty_lines=False,
        split_respect_sentence_boundary=False,
        clean_whitespace=False
    )
    squad_input_filename = Path(args.squad_input_filename)
    qa_pairs_input_filename = Path (args.qa_pairs_input_filename)
    dpr_output_filename = Path(args.dpr_output_filename)
    num_hard_negative_ctxs = args.num_hard_negative_ctxs
    split_dataset = args.split_dataset

    main(
        squad_input_filename=squad_input_filename,
        qa_pairs_input_filename= qa_pairs_input_filename,
        dpr_output_filename=dpr_output_filename,
        max_seq_len=args.max_seq_len_passage,
        num_hard_negative_ctxs=num_hard_negative_ctxs,
        split_dataset=split_dataset,
    )

    print (f"Successfully created DPR dataset with examples. Saved at {dpr_output_filename}")