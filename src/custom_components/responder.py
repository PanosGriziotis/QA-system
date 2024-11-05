from haystack.nodes.base import BaseComponent
from haystack.schema import Answer
from utils.metrics import compute_context_relevance


class Responder(BaseComponent):
    """A node to return the final response in a query pipeline"""

    outgoing_edges = 1

    def __init__(self):
        super().__init__()
    
    def run(self, query, documents, answers, threshold:float=None):
        """param threshold: a context_relevance score threshold ranging from 0 to 1. The Responder returns a fallback response message instead of the Reader's answer if context relevance score is below define threshold."""
        
        if answers[0].type == "extractive":
            doc_id = answers[0].document_ids[0]
            for doc in documents:
                if doc.id == doc_id:
                    docs = [doc.content]
        else:            
            docs = [doc.content for doc in documents]
        
        cr_score = compute_context_relevance(query=query,documents=docs)

        if threshold is None or threshold == 0.0 or cr_score >= threshold:
            answers[0].meta["context_relevance"] = cr_score
            output = {
                "query": query,
                "answers": answers,
                "documents": documents
            }
        else: 
            answer = Answer(
                answer="Συγγνώμη, αλλά δεν διαθέτω αρκετές πληροφορίες για να σου απαντήσω σε αυτήν την ερώτηση.",
                type="other",
                context=None,
                document_ids=None,
                meta={"context_relevance": cr_score}
            )
        
            output = {
                "query": query,
                "answers": [answer],
                "documents": documents,
            }

        return output, "output_1"
    
    def run_batch(self):
        return