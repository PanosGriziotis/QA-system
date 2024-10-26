from haystack.nodes.base import BaseComponent
from haystack.schema import Answer


class Responder(BaseComponent):
    """A node to return the final response in a query pipeline"""

    outgoing_edges = 1

    def __init__(self):
        super().__init__()
    
    def run(self, query, documents, cr_score, answers=None):
        if answers is None:
            answer = Answer(
                answer="Συγγνώμη, αλλά δεν διαθέτω αρκετές πληροφορίες για να σου απαντήσω σε αυτήν την ερώτηση.",
                type="other",
                context=None,
                document_ids=None,
                meta=None,
            )
            answers = [answer]

        output = {
            "query": query,
            "answers": answers,
            "documents": documents,
            "cr_score": cr_score
        }

        return output, "output_1"
    
    def run_batch(self):
        return