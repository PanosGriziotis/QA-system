# In this script we map the subintents of the 164  queries from the dataset with their corresponding response given by Theano.

from typing import Union, List
import yaml
import re
import argparse
import pandas as pd

def get_responses(yml_file, dataset_file):
    # Step 1: Read and parse the YAML file
    with open(yml_file, "r", encoding="utf-8") as fp:
        responses = yaml.safe_load(fp)["responses"]
        responses_intents = list(responses.keys())

    # Step 3: Prepare the data for the CSV
    data = []
    df = pd.read_csv(dataset_file)

    for query, intent_name in zip(df["query"], df["intent_name"]):

        # Initialize response as None
        response_text = None

        # Step 4: Map responses to intents (skip if intent is 'out_of_scope')

        for response_intent in responses_intents:
            if intent_name in response_intent:
                response_text = responses[response_intent][0]["text"]
                break
        
        # Append the row data
        data.append({"intent_name": intent_name, "query": query, "response": response_text})

    # Step 5: Save the data to a CSV file
    df = pd.DataFrame(data)
    df.to_csv(dataset_file, index=True, index_label="id")
        
def main ():
    parser = argparse.ArgumentParser()
    parser.add_argument("--yml_file", type=str,default="/home/pgriziotis/thesis/covid-va-chatbot/domain.yml", help="Path to responses data")
    
    args = parser.parse_args()

    get_responses(yml_file=args.yml_file, dataset_file= "./final_dataset.csv"  )
if __name__ == "__main__":
    main()
