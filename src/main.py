from typing import List, Optional

import os 
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from fastapi import FastAPI, UploadFile, File
from pathlib import Path
import uuid
import logging
from numpy import ndarray
import json
import time
from schema import QueryRequest, QueryResponse

from document_store.initialize_document_store import document_store as DOCUMENT_STORE
from pipelines.query_pipelines import init_rag_pipeline, init_extractive_qa_pipeline
from pipelines.indexing_pipeline import indexing_pipeline
from utils.data_handling import flush_cuda_memory, convert_numpy_scalars

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="QA-subsystem API")
query_pipeline = init_rag_pipeline(use_gpu=False) # initialize model
FILE_UPLOAD_PATH = os.getenv("FILE_UPLOAD_PATH", str((Path(__file__).parent / "file-upload").absolute()))
Path(FILE_UPLOAD_PATH).mkdir(parents=True, exist_ok=True)

@app.get("/ready")
def check_status():
    """Check if the server is ready to take requests."""
    return True

@app.post("/file-upload")
def upload_files(
    files: List[UploadFile] = File(...),
    keep_files: Optional[bool] = False
    ):
    """
    Use this endpoint to upload files for indexing in Document Store.
    
    Params:
    Pass the `keep_files=true` parameter if you want to keep files in the file_upload folder after being indexed
    """

    file_paths = []
    
    for file_to_upload in files:
        file_path = Path(FILE_UPLOAD_PATH) / f"{uuid.uuid4().hex}_{file_to_upload.filename}"
        with file_path.open("wb") as fo:
            fo.write(file_to_upload.file.read())
        file_paths.append(file_path)
        file_to_upload.file.close()
        
    result = indexing_pipeline.run(file_paths=file_paths)
    
    # convert embeddings to lists of float numbers
    for document in result.get('documents', []):
        if isinstance(document.embedding, ndarray):
            document.embedding = document.embedding.tolist()

    if not keep_files:
        for p in file_paths:
            p.unlink()

    return result


@app.post("/extractive-query", response_model=QueryResponse)
async def ask_extractive_qa_pipeline(request: QueryRequest):
    """
    Use this endpoint to post queries as input to an Extractive QA pipeline (The output is a text span in the retrieved documents)
    """
    # Initialize pipeline
    query_pipeline = init_extractive_qa_pipeline(use_gpu = True)
    start_time = time.time()

    params = request.params or {}
    if DOCUMENT_STORE.get_document_count() == 0:
        raise ValueError("Document store index is empty. Esure documents are indexed in document store before runing the query pipeline")
    
    # Run query pipeline using input parameters
    result = query_pipeline.run(query=request.query, params=params)
    
    result = convert_numpy_scalars(result)
    # Ensure answers and documents and answers exist, even if they're empty lists
    if "documents" not in result:
        result["documents"] = []
    if not "answers" in result:
        result["answers"] = []
    
    flush_cuda_memory()

    logging.info(
        json.dumps({"request": request.dict(), "response": result, "time": f"{(time.time() - start_time):.2f}"}, default=str, ensure_ascii=False)
    )
    return result


@app.post("/rag-query")
def ask_rag_pipeline(request: QueryRequest):
    """
    Use this endpoint to post queries as input to a Retrieval Augmented Generation (RAG) pipeline (The output answer is generated answer given the retrieved documents)
    """
    start_time = time.time()
    params = request.params or {}

    if DOCUMENT_STORE.get_document_count() == 0:
        raise ValueError("Document store index is empty. Esure documents are indexed in document store before runing the query pipeline")
    
    result = query_pipeline.run(query=request.query, params=params)

    result = convert_numpy_scalars(result)
    # Ensure answers and documents exist, even if they're empty lists
    if not "documents" in result:
        result["documents"] = []
    if not "answers" in result:
        result["answers"] = []

    print(
        json.dumps({"request": request.dict(), "response": result, "time": f"{(time.time() - start_time):.2f}"}, default=str, ensure_ascii=False)
    )

    flush_cuda_memory()

    return result