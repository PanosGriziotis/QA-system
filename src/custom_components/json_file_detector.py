from haystack.nodes.base import BaseComponent
from pathlib import Path
from typing import List

class JsonFileDetector (BaseComponent):
    """
    Detect and route a json/jsonl input file in to JSONConverter module
    
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
        self):
         return