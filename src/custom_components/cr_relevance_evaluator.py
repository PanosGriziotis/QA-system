
from utils.metrics import compute_context_relevance
from haystack.nodes.base import BaseComponent

class ContextRelevanceEvaluator(BaseComponent):
    """A node to evaluate weather the retrieved context (text documents) is relevant to the query. 
    
    output_edge_1: If threashold not applied or if cr_score is higher or equal to the threshold  
    output_edge_2: If threshold applied and cr_score is less or equal to the threshold"""
    
    outgoing_edges = 2

    def __init__(self):
        super().__init__()
    
    def run (self, query, documents, threshold:float=0.17):
        docs = [doc.content for doc in documents]
        cr_score = compute_context_relevance(query=query,documents=docs)
        output={
            "query": query,
            "documents": documents,
            "cr_score": cr_score
        }
        if threshold is not None and cr_score <= threshold:
            output_index = 2
        else:
            output_index = 1

        return output, f"output_{output_index}"
    
    def run_batch(self):
        return