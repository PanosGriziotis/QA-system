# Get nlu data examples from yaml file
from typing import Union, List
import yaml
import re
import argparse

def get_examples_for_intent(filepath: str, intent_name: str) -> List[dict]:
    """
    Extract example queries from a specified intent in the NLU training data.

    :param filepath: Filepath with NLU training data (YAML)
    :param intent_name: Name of the intent (supercategory) to extract example queries from
    :return: List of example queries for the specified intent
    """
    try:
        with open(filepath, "r", encoding="utf-8") as fp:
            data = yaml.safe_load(fp)
            
    except FileNotFoundError:
        print(f"File not found: {filepath}")
        return []
    except yaml.YAMLError as e:
        print(f"Error parsing YAML file: {e}")
        return []


    nlu_data = data["nlu"]
    # here we store any intent and subintent examples of the given inteent name
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


def clean_examples(examples:List[str]):
    """
    Remove entity tags e.g., (country). Remove the square brackets from entity text e.g., [Ελλάδα].
    """ 
    pattern_square_brackets = r'\[(.*?)\]'
    pattern_parentheses = r'\(.*?\)'
    cleaned_examples = []
    for example in examples:
        example_no_parentheses = re.sub(pattern_parentheses, '', example).strip()
        cleaned_example = re.sub(pattern_square_brackets, r'\1', example_no_parentheses).strip()
        cleaned_examples.append(cleaned_example)
    return cleaned_examples


def main ():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file_path", type=str, help="Path to nlu data")
    parser.add_argument("--save_dir", type=str, help="Directory to save extracted nlu data with intent label")
    args = parser.parse_args()

    intents = ["out_of_scope_general", "mask_faq", "EODY_faq", "vaccines"]


    for intent in intents:
        nlu_data = get_examples_for_intent(filepath=args.file_path, intent_name=intent)

        with open (f"{args.save_dir}/{intent}.txt", "w") as fp:
            for data in nlu_data:
                intent_name = data["intent"]
                examples = data["examples"]
                for query in examples:
                    fp.write(f"{query}\t{intent_name}\n")

if __name__ == "__main__":
    main()