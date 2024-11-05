import json
import random
import os
from tqdm import tqdm
import sys
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(SCRIPT_DIR, '../')))
from transformers import AutoTokenizer
from sentence_transformers import SentenceTransformer
from haystack.document_stores.elasticsearch import ElasticsearchDocumentStore
from haystack.nodes.retriever import EmbeddingRetriever
from haystack.nodes.retriever.sparse import BM25Retriever 
from haystack.nodes.label_generator import PseudoLabelGenerator
from haystack.nodes import PreProcessor
from utils.data_handling import DataExtractor

# Environment configuration
os.environ['TOKENIZERS_PARALLELISM'] = 'true'
# Define constants
BI_ENCODER = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
INPUT_SQUAD_DATASET = os.path.join (SCRIPT_DIR, "covid_QA_el_small/COVID-QA-el_small.json")
INPUT_JSON_FILE = os.path.join (SCRIPT_DIR, "data/who_pairs.json")
GPL_DATA_FILE =  os.path.join (SCRIPT_DIR, "data/gpl_data.json")
TRAINING_DATA_FILE = os.path.join (SCRIPT_DIR, "data/gpl_training_data.json" )
RETRIEVER_SAVE_DIR = os.path.join (SCRIPT_DIR,"/adapted_retriever")
INDEX_NAME = "gpl"

def initialize_models_and_tokenizer():
    """Initializes the sentence transformer and tokenizer models."""
    max_seq_len = SentenceTransformer(BI_ENCODER).max_seq_length
    tokenizer = AutoTokenizer.from_pretrained("nlpaueb/bert-base-greek-uncased-v1")
    return max_seq_len, tokenizer

def extract_data(tokenizer, max_seq_len):
    """Extracts and shuffles query-document pairs from SQuAD and JSON files."""
    data_extractor = DataExtractor(tokenizer, max_seq_len)
    squad_query_doc_pairs = data_extractor.get_query_doc_pairs_from_squad_file(INPUT_SQUAD_DATASET)
    who_query_doc_pairs = data_extractor.get_query_doc_pairs_from_json_file(INPUT_JSON_FILE)
    qd_pairs = squad_query_doc_pairs + who_query_doc_pairs
    random.shuffle(qd_pairs)
    return qd_pairs

def save_query_doc_pairs(qd_pairs, file_path):
    """Saves the extracted query-document pairs to a JSON file."""
    with open(file_path, "w") as fp:
        json.dump(qd_pairs, fp, ensure_ascii=False)

def create_corpus_from_pairs(qd_pairs):
    """Creates a corpus from the query-document pairs."""
    return [qd_pair["document"] for qd_pair in qd_pairs]

def initialize_document_store(index_name, embedding_dim=384):
    """Initializes the Elasticsearch document store."""
    return ElasticsearchDocumentStore(index=index_name, embedding_dim=embedding_dim, recreate_index=True)

def preprocess_and_store_docs(document_store, corpus):
    """Processes documents and writes them to the document store."""
    preprocessor = PreProcessor(clean_empty_lines=False, clean_whitespace=False, split_respect_sentence_boundary=False)
    docs = preprocessor.process([{"content": t} for t in corpus])
    document_store.write_documents(docs)

def generate_training_instances(qd_pairs, document_store):
    """Retrieved hard negative passages for the query-document pairs using BM25 retriever. Margin scores are also computed using a cross encoder (only used if margin_mse training loss is chosen) """
    bm25_retriever = BM25Retriever(document_store=document_store)
    psg = PseudoLabelGenerator(qd_pairs, bm25_retriever, cross_encoder_model_name_or_path="amberoad/bert-multilingual-passage-reranking-msmarco")
    output, _ = psg.run(documents=document_store.get_all_documents())
    return output

def save_training_instances(output, file_path):
    """Saves pseudo labels to a training data file."""
    with open(file_path, "w") as fp:
        fp.write(str(output["gpl_labels"]))

def initialize_and_train_retriever(document_store, max_seq_len):
    """Initializes and trains the embedding retriever with SBERT and MNRL loss."""
    retriever = EmbeddingRetriever(
        document_store=document_store,
        embedding_model=BI_ENCODER,
        model_format="sentence_transformers",
        max_seq_len=max_seq_len,
        progress_bar=True,
    )
    document_store.update_embeddings(retriever)
    return retriever

def main():
    # Step 1: Initialize models and tokenizer
    max_seq_len, tokenizer = initialize_models_and_tokenizer()

    # Step 2: Extract and save query-document pairs
    qd_pairs = extract_data(tokenizer, max_seq_len)
    save_query_doc_pairs(qd_pairs, GPL_DATA_FILE)

    # Step 3: Prepare corpus and initialize document store
    corpus = create_corpus_from_pairs(qd_pairs)
    document_store = initialize_document_store(INDEX_NAME)

    # Step 4: Process and store documents in document store
    preprocess_and_store_docs(document_store, corpus)

    # Step 5: Generate and save training instances (query, positive, negative passages)
    output = generate_training_instances(qd_pairs, document_store)
    save_training_instances(output, TRAINING_DATA_FILE)

    # Step 6: Initialize, train, and save retriever
    retriever = initialize_and_train_retriever(document_store, max_seq_len)
    retriever.train(output["gpl_labels"], train_loss="mnrl", n_epochs=40)
    retriever.save(save_dir=RETRIEVER_SAVE_DIR)

if __name__ == "__main__":
    main()