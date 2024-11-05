from haystack.document_stores import InMemoryDocumentStore, ElasticsearchDocumentStore 
from haystack.nodes.retriever.dense import DensePassageRetriever
from pathlib import Path
import logging
import argparse

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()    
    parser.add_argument('-train_file', type=str, required = True)
    parser.add_argument('-doc_dir', type=str, required = True)
    parser.add_argument('-query_model', type =str, required=False, default = "nlpaueb/bert-base-greek-uncased-v1")
    parser.add_argument('-passage_model', type=str, required=False, default= "nlpaueb/bert-base-greek-uncased-v1")
    parser.add_argument('--save_dir', type=str, required=False, default =  Path.cwd(),  help='directory to save trained retriever or evaluation report')
    args = parser.parse_args()

    retriever = DensePassageRetriever(
    document_store=ElasticsearchDocumentStore(),
    query_embedding_model=args.query_model,
    passage_embedding_model=args.passage_model,
    max_seq_len_query=64,
    max_seq_len_passage=300,
)

    retriever.train(
        data_dir=args.doc_dir,
        train_filename=args.train_file,
        test_filename=args.dev_file,
        max_processes = 128,
        num_hard_negatives=2,
        num_positives=1,
        n_epochs=20,
        batch_size=4,
        evaluate_every=1000,
        learning_rate = 1e-6,
        epsilon  = 1e-08,
        weight_decay = 0.0,
        n_gpu=3,
        num_warmup_steps  = 100,
        grad_acc_steps = 8,
        optimizer_name = "AdamW",
        save_dir=args.save_dir
)
        
        
        