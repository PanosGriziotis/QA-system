import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

import logging

from haystack.nodes import FileTypeClassifier, JsonConverter, TextConverter, PDFToTextConverter, DocxToTextConverter, PreProcessor
from haystack.pipelines import Pipeline
from haystack.nodes import EmbeddingRetriever, PreProcessor
from transformers import AutoTokenizer
from custom_components.json_file_detector import JsonFileDetector
from document_store.initialize_document_store import document_store as DOCUMENT_STORE

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

if DOCUMENT_STORE is None:
    raise ValueError("the imported document_store is None. Please make sure that the Elasticsearch service is properly launched")

# Initialize Bi-encoder model through embedding retriever class to pre-compute document embeddings at indexing time
retriever = EmbeddingRetriever(embedding_model= "panosgriz/covid_el_paraphrase-multilingual-MiniLM-L12-v2", document_store=DOCUMENT_STORE)
# Initialize tokenizer for preprocessing stage
tokenizer = AutoTokenizer.from_pretrained("ilsp/Meltemi-7B-v1.5")

def convert_file_to_doc_pipeline (preprocessor:PreProcessor=None) -> Pipeline:

    """Two stage preprocessing pipeline method to prepare input text data for indexing in document store.
    
    Stage1 (Convert file to documents): routes an input data file (.txt, .json, .jsonl, .pdf, .docx) to the corresponding Converter module and creates haystack document objects (dictionaries consisting of text and metadata fields)
    Stage2 (Preprocess before indexing): preprocesses document objects from previous stage with a PreProcessor module. Preprocessing includes 1) splitting documents into smaller chunks b) cleaning any trailing whitespaces
    
    Args: preprocessor: custom PreProcessor instance
    """

    # initialize pipeline components
    if preprocessor is None:
        preprocessor = PreProcessor(
        clean_empty_lines=True,
        clean_whitespace=True,
        split_respect_sentence_boundary=True,
        split_by="token",
        split_length=128,
        tokenizer=tokenizer
        )
        
    file_type_classifier = FileTypeClassifier()
    text_converter = TextConverter(valid_languages=['el', 'en'])
    pdf_converter = PDFToTextConverter(valid_languages=['el', 'en'])
    docx_converter = DocxToTextConverter(valid_languages=['el', 'en'])
    json_converter =JsonConverter(valid_languages=["el", 'en'])    
    
    # Bring everything together in a pipeline
    p = Pipeline()
    p.add_node(component=JsonFileDetector(), name="JsonFileDetector", inputs=["File"])
    p.add_node(component=json_converter, name="JsonConverter", inputs=["JsonFileDetector.output_1"])
    p.add_node(component=file_type_classifier, name="FileTypeClassifier", inputs=["JsonFileDetector.output_2"])
    p.add_node(component=text_converter, name="TextConverter", inputs=["FileTypeClassifier.output_1"])
    p.add_node(component=pdf_converter, name="PdfConverter", inputs=["FileTypeClassifier.output_2"])
    p.add_node(component=docx_converter, name="DocxConverter", inputs=["FileTypeClassifier.output_4"])
    p.add_node(component=preprocessor, name="Preprocessor", inputs=["JsonConverter", "TextConverter", "PdfConverter", "DocxConverter"])

    return p

# Initialize preprocessing pipeline
indexing_pipeline = convert_file_to_doc_pipeline()
# Compute document embeddings
indexing_pipeline.add_node(component=retriever, name = "DenseRetriever", inputs=["Preprocessor"])
# Index document objects in DS
indexing_pipeline.add_node(component=DOCUMENT_STORE, name= "DocumentStore", inputs=["DenseRetriever"])

