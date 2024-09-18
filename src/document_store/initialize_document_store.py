import time
import requests
import os 

from haystack.document_stores import ElasticsearchDocumentStore
from haystack.utils import launch_es

DOCUMENTSTORE_PARAMS_HOST = os.environ["DOCUMENTSTORE_PARAMS_HOST"] if "DOCUMENTSTORE_PARAMS_HOST" in os.environ else "localhost"
DOCUMENTSTORE_PARAMS_PORT = int(
    os.environ['DOCUMENTSTORE_PARAMS_PORT']) if "DOCUMENTSTORE_PARAMS_PORT" in os.environ else 9200

def check_elasticsearch():
    """Check if Elasticsearch is up and running."""
    url = f"http://{DOCUMENTSTORE_PARAMS_HOST}:{DOCUMENTSTORE_PARAMS_PORT}/_cat/health"
    try:
        response = requests.get(url)
        if response.status_code == 200:
            return True
        else:
            return False
    except requests.exceptions.RequestException:
        return False

def initialize_document_store():
    """Initialize a Elasticsearch document store object."""
    return ElasticsearchDocumentStore(
        host=DOCUMENTSTORE_PARAMS_HOST,
        port=DOCUMENTSTORE_PARAMS_PORT,
        username="",
        password="",
        index="document",
        embedding_dim=384,
        duplicate_documents="overwrite"
    )
    
if not check_elasticsearch():    
    print ("Elasticsearch document store is not running. Please start the elasticsearch docker container")
    
document_store = initialize_document_store()
