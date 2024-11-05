import random
import collections
from typing import List
import yaml
import re
import argparse
import os
import pandas as pd

# Function to extract NLU examples from the YAML file
def get_examples_for_intent(filepath: str, intent_name: str) -> List[dict]:
    try:
        with open(filepath, "r", encoding="utf-8") as fp:
            data = yaml.safe_load(fp)
    except FileNotFoundError:
        print(f"File not found: {filepath}")
        return []
    except yaml.YAMLError as e:
        print(f"Error parsing YAML file: {e}")
        return []

    nlu_data = data.get("nlu", [])
    intent_data = []
    for intent_instance in nlu_data:
        try:
            intent = intent_instance['intent']
            if intent_name in intent:
                examples = intent_instance['examples'].strip().split('\n')
                examples = [example.strip('- ').strip() for example in examples]
                c_examples = clean_examples(examples)
                intent_instance["examples"] = c_examples
                intent_data.append(intent_instance)
        except KeyError:
            continue
    return intent_data 

def clean_examples(examples: List[str]):
    pattern_square_brackets = r'\[(.*?)\]'
    pattern_parentheses = r'\(.*?\)'
    cleaned_examples = []
    for example in examples:
        example_no_parentheses = re.sub(pattern_parentheses, '', example).strip()
        cleaned_example = re.sub(pattern_square_brackets, r'\1', example_no_parentheses).strip()
        cleaned_examples.append(cleaned_example)
    return cleaned_examples

# Function to generate the dataset
def generate_dataset(file_path: str, intents: List[str], save_path: str):
    # Step 1: Load examples for each intent 
    queries_faq = collections.defaultdict(list)
    queries_out_of_scope = collections.defaultdict(list)  # Store each sub-intent separately

    for intent in intents:
        nlu_data = get_examples_for_intent(filepath=file_path, intent_name=intent)
        for data in nlu_data:
            intent_name = data["intent"]
            examples = data["examples"]

            if "out_of_scope" in intent_name:
                # Collect out-of-scope examples by sub-intent
                for query in examples:
                    queries_out_of_scope[intent_name].append(query)
            else:
                # Collect in-scope examples by intent
                for query in examples:
                    queries_faq[intent_name].append(query)

    # Step 2: Select queries for each category
    final_dataset = []

    # Select out_of_scope queries: 1 per sub-intent + rest from out_of_scope/general
    general_out_of_scope = queries_out_of_scope.get("out_of_scope/general", [])
    other_out_of_scope = [
        (sub_intent, random.choice(examples)) for sub_intent, examples in queries_out_of_scope.items()
        if sub_intent != "out_of_scope/general"
    ]
    selected_out_of_scope = [(query, sub_intent) for sub_intent, query in other_out_of_scope]

    # Calculate how many additional general out-of-scope examples are needed to reach 87 total
    additional_out_of_scope_needed = 87 - len(selected_out_of_scope)
    selected_out_of_scope.extend(
        (query, "out_of_scope/general") for query in random.sample(general_out_of_scope, additional_out_of_scope_needed)
    )

    # Select in-scope queries: 35 from EODY_faq, 33 from Vaccines, 15 from mask_faq (1 per sub-intent)
    def select_queries(queries_dict, required_count):
        selected = [
            (random.choice(examples), sub_intent) for sub_intent, examples in queries_dict.items()
        ]
        return selected[:required_count]

    selected_eody = select_queries(
        {k: v for k, v in queries_faq.items() if k.startswith("EODY_faq")}, 35
    )
    selected_vaccines = select_queries(
        {k: v for k, v in queries_faq.items() if k.startswith("vaccines")}, 33
    )
    selected_mask_faq = select_queries(
        {k: v for k, v in queries_faq.items() if k.startswith("mask_faq")}, 15
    )

    # Step 3: Combine all selected queries and shuffle
    final_dataset = selected_out_of_scope + selected_eody + selected_vaccines + selected_mask_faq
    random.shuffle(final_dataset)

    # Step 4: Convert to DataFrame and save as CSV
    df = pd.DataFrame(final_dataset, columns=["query", "intent_name"])
    df.to_csv(save_path, index=False, encoding='utf-8')

# Map responses to intents and save to CSV
def get_responses(yml_file: str, dataset_file: str):
    with open(yml_file, "r", encoding="utf-8") as fp:
        responses = yaml.safe_load(fp)["responses"]

    responses_intents = list(responses.keys())
    data = []
    df = pd.read_csv(dataset_file)

    for query, intent_name in zip(df["query"], df["intent_name"]):
        response_text = None
        for response_intent in responses_intents:
            if intent_name in response_intent:
                response_text = responses[response_intent][0]["text"]
                break
        data.append({"intent_name": intent_name, "query": query, "response": response_text})

    final_df = pd.DataFrame(data)
    final_df.to_csv(dataset_file, index=True, index_label="id")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nlu_file", type=str, default="./nlu.yml" ,help="Path to NLU data (nlu.yml)")
    parser.add_argument("--domain_file", type=str, default="./domain.yml", help="Path to domain data (domain.yml)")
    parser.add_argument("--output_file", type=str, default="./test_dataset.csv", help="Output path for the final dataset (CSV)")
    args = parser.parse_args()

    # Generate initial dataset
    intents = ["out_of_scope", "mask_faq", "EODY_faq", "vaccines"]
    generate_dataset(file_path=args.nlu_file, intents=intents, save_path=args.output_file)
    
    # Map responses to intents and save final CSV
    get_responses(yml_file=args.domain_file, dataset_file=args.output_file)

if __name__ == "__main__":
    main()