import glob
import requests
import logging
import os

logging.basicConfig(level=logging.INFO)

HAYSTACK_SERVICE_HOST = os.environ["HAYSTACK_SERVICE_HOST"] if "HAYSTACK_SERVICE_HOST" in os.environ else "localhost"
HAYSTACK_SERVICE_PORT = int(
    os.environ['HAYSTACK_SERVICE_PORT']) if "HAYSTACK_SERVICE_PORT" in os.environ else 8001


def ingest_data():
    """
    Call the file-upload endpoint with all the files in the index data folder.
    """ 
    
    data_dir = f"{os.path.dirname(os.path.abspath(__file__))}/data/full_corpus"
    for file in glob.glob(f"{data_dir}/*"):
        logging.info(f"Indexing content in {file} to document store")
        with open(file, "rb") as f:
            requests.post(url=f"http://{HAYSTACK_SERVICE_HOST}:{HAYSTACK_SERVICE_PORT}/file-upload", files={"files": f})   

if __name__ == "__main__":
    ingest_data()
