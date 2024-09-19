from query_pipelines import Generator
import pandas as pd
import torch
import gc

def rewrite_query (query):

    prompt_messages= [
            {"role": "system", "content": 'Επαναδιατύπωσε την ακόλουθη ερώτηση του χρήστη σε μια σαφή, συγκεκριμένη και επίσημη ερώτηση κατάλληλη για την ανάκτηση σχετικών πληροφοριών από μια βάση δεδομένων.'},
            {"role": "user", "content": f"{query}"}]
    
    generator = Generator(model_name="ilsp/Meltemi-7B-Instruct-v1", prompt_messages= prompt_messages)
    
    result, _ = generator.run(query=query, documents=[], max_new_tokens=100)
    return result

dir = "/other/users/panagri/thesis/QA-subsystem-thesis/dev/evaluation/nlu_test_data/final_dataset.csv"
df = pd.read_csv(dir)
list_of_queries= df ["query"].to_list()

new_queries = []
print (list_of_queries)
for query in list_of_queries:
    print (query)
    results = rewrite_query(query=query)
    new_query = results["answers"][0].answer
    print (new_query)
    new_queries.append (query)
    torch.cuda.empty_cache()
    gc.collect()

with open ("new_queries.txt", "w") as fp:
    for q in new_queries:

        fp.write (q + "\n")