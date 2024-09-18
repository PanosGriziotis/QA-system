# File classifier for: .txt, .pdf, .docx, .json, .jsonl files
from typing import Dict, List, Union
from haystack.schema import Document
from haystack.nodes import FileTypeClassifier, JsonConverter, TextConverter, PDFToTextConverter, DocxToTextConverter, PreProcessor
from haystack.pipelines import Pipeline
from haystack.nodes.base import BaseComponent
from pathlib import Path
import nltk
nltk.download('punkt_tab')

class JsonFileDetector (BaseComponent):
    """
    Detect and route json or jsonl input file in a Pipeline to JSONConverter
    Note: Only a single file path is allowed as input
    """
    outgoing_edges = 2
    def __init__(self):
        super().__init__()

    def _get_extension(self, file_paths: List[Path]) -> str:
        extension = file_paths[0].suffix.lower()
        return extension
    
    def run(self, file_paths:List[str]):
        
        paths = [Path(path) for path in file_paths]
        extension = self._get_extension(paths)
        output_index = 1 if extension in [".json", ".jsonl"] else 2

        output = {"file_paths": paths}

        return output, f'output_{output_index}'

    def run_batch(
        self,
        **kwargs):
         return

def init_file_to_doc_pipeline (custom_preprocessor:PreProcessor=None) -> Pipeline:
    """Pipeline to route file to corresponding converter and preprocess the resulting docs"""

    file_type_classifier = FileTypeClassifier()
    text_converter = TextConverter(valid_languages=['el', 'en'])
    pdf_converter = PDFToTextConverter(valid_languages=['el', 'en'])
    docx_converter = DocxToTextConverter(valid_languages=['el', 'en'])
    json_converter =JsonConverter(valid_languages=["el", 'en'])
    # initialize default preprocessor if not given in function arguments
    
    
    p = Pipeline()

    # Classify doc according to extension and routes it to corresponding converter
    p.add_node (component=JsonFileDetector(), name="JsonFileDetector", inputs=["File"])
    p.add_node(component=json_converter, name="JsonConverter", inputs=["JsonFileDetector.output_1"])
    p.add_node(component=file_type_classifier, name="FileTypeClassifier", inputs=["JsonFileDetector.output_2"])
    p.add_node(component=text_converter, name="TextConverter", inputs=["FileTypeClassifier.output_1"])
    p.add_node(component=pdf_converter, name="PdfConverter", inputs=["FileTypeClassifier.output_2"])
    p.add_node(component=docx_converter, name="DocxConverter", inputs=["FileTypeClassifier.output_4"])
    # Split, clean and convert document(s) to haystack Document object(s)
    p.add_node(component=custom_preprocessor, name="Preprocessor", inputs=["JsonConverter", "TextConverter", "PdfConverter", "DocxConverter"])

    return p
