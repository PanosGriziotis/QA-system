import os
import json
import string

import requests

dir = "final_results"
new_dir = "final_results_no_empty_answers"

def post_question_request (query:str, params:dict, endpoint:str):

    request_body = {
        "query": query,
        "params": params
    }
    response = requests.post(url= f"http://localhost:8000/{endpoint}", json=request_body)

    result = response.json()
    
    return result

for file in os.listdir(new_dir):
    counter = 0
    queries = []
    indices = []
    with open (os.path.join(new_dir, file), "r") as fp:
        data= json.load(fp)
        for d in data:
              answer = d["answers"][0]["answer"].strip()
              if answer == "Συγγνώμη, αλλά δεν μπορώ να δώσω μία απάντηση σε αυτήν την ερώτηση.":
                    counter +=1
                    queries.append (d["query"])
                    indices.append(data.index(d))
    print (f"Found {counter} answers with empty string or single punctuation in {file}")

    '''
    file_parts = file.split("_")
    if file_parts[0] == "extractive":
        top_k_ranker = int(file_parts[1])
        top_k_reader = int(file_parts[2])
        
        params = {"Retriever": {"top_k":20}, "Ranker": {"top_k": top_k_ranker}, "Reader": {"top_k": top_k_reader}}
        for query, index in zip(queries,indices):
             
            result = post_question_request(query=query, params=params,endpoint="extractive-query")
            result["answer_accuracy"] = None
            print (result["answers"][0]["answer"])
            data [index] = result
    with open (os.path.join(new_dir, file), "w") as fp:
         json.dump(data, fp, ensure_ascii=False, indent=4)
    '''