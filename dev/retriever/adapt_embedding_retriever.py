import json
import random
from tqdm.auto import tqdm
from transformers import AutoTokenizer
from sentence_transformers import SentenceTransformer
from haystack.document_stores.elasticsearch import ElasticsearchDocumentStore
from haystack.nodes.retriever import EmbeddingRetriever
from haystack.nodes.retriever.sparse import BM25Retriever 
from haystack.nodes.label_generator import PseudoLabelGenerator
from haystack.nodes import PreProcessor
from get_gpl_data import GPL_data
import os 

os.environ['TOKENIZERS_PARALLELISM'] = 'true'

# Define constants
CROSS_ENCODER = "amberoad/bert-multilingual-passage-reranking-msmarco"
BI_ENCODER = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
INPUT_SQUAD_DATASET = "../data/covid_QA_el_small/COVID-QA-el_small.json"
INPUT_JSON_FILE = "../data/who/who_pairs.json"
GPL_DATA_FILE = "gpl_data.json"
TRAINING_DATA_FILE = "./gpl_training_data.json"
RETRIEVER_SAVE_DIR = "./adapted_retriever"
INDEX_NAME = "gpl"

# Initialize models and tokenizer
max_seq_len = SentenceTransformer(BI_ENCODER).max_seq_length
tokenizer = AutoTokenizer.from_pretrained("nlpaueb/bert-base-greek-uncased-v1")

# Extract data
data_extractor = GPL_data(tokenizer, max_seq_len)
squad_query_doc_pairs = data_extractor.get_query_doc_pairs_from_squad_file(INPUT_SQUAD_DATASET)
who_query_doc_pairs = data_extractor.get_query_doc_pairs_from_file(INPUT_JSON_FILE)
qd_pairs = squad_query_doc_pairs + who_query_doc_pairs
random.shuffle(qd_pairs)

# Save extracted query-document pairs to file
with open(GPL_DATA_FILE, "w") as fp:
    json.dump(qd_pairs, fp, ensure_ascii=False)

# Create corpus from query-document pairs
corpus = [qd_pair["document"] for qd_pair in qd_pairs]

# Initialize document store and preprocessor
document_store = ElasticsearchDocumentStore(index=INDEX_NAME, embedding_dim= 384, recreate_index=True)
preprocessor = PreProcessor(clean_empty_lines=False, clean_whitespace=False, split_respect_sentence_boundary=False)
docs = preprocessor.process([{"content": t} for t in corpus])
document_store.write_documents(docs)

# Initialize BM25 retriever and pseudo label generator
bm25_retriever = BM25Retriever(document_store=document_store)
psg = PseudoLabelGenerator(qd_pairs, bm25_retriever, cross_encoder_model_name_or_path=CROSS_ENCODER)
output, _ = psg.run(documents=document_store.get_all_documents())

# Save pseudo labels to file
with open(TRAINING_DATA_FILE, "w") as fp:
    fp.write(str(output["gpl_labels"]))

# Initialize and train embedding retriever
retriever = EmbeddingRetriever(
    document_store=document_store,
    embedding_model=BI_ENCODER,
    model_format="sentence_transformers",
    max_seq_len=max_seq_len,
    progress_bar=True,
)
document_store.update_embeddings(retriever)
retriever.train(output["gpl_labels"], n_epochs=20)
retriever.save(save_dir=RETRIEVER_SAVE_DIR)
