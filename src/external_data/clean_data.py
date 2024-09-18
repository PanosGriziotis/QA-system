import json

import json
'''
def remove_duplicates_based_on_another_file(input_file, compare_file, output_file, unwanted_substrings):
    seen_content = set()
    
    # Load content from compare_file
    with open(compare_file, 'r', encoding='utf-8') as compare_infile:
        for line in compare_infile:
            try:
                compare_data = json.loads(line.strip())
                if 'content' in compare_data:
                    seen_content.add(compare_data['content'])
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON line from compare_file: {e}")

    # Process input_file and remove duplicates/unwanted substrings
    with open(input_file, 'r', encoding='utf-8') as infile, open(output_file, 'w', encoding='utf-8') as outfile:
        for line in infile:
            try:
                data = json.loads(line.strip())
                if 'content' in data:
                    content = data['content']
                    # Check for unwanted substrings
                    if any(substring in content for substring in unwanted_substrings):
                        continue  # Skip this entry

                    # Check for duplicate content
                    if content not in seen_content:
                        seen_content.add(content)
                        outfile.write(json.dumps(data, ensure_ascii=False) + '\n')
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON line from input_file: {e}")

# Example usage:
input_file = "data/crawled_docs_eody_psychology.jsonl"
compare_file = "data/crawled_eody_odigies.jsonl"
output_file = "data/output_psychology.jsonl"
unwanted_substrings_ecdc = ["Graphics Design by CIRCUS DESIGN STUDIO", "Διαγωνισμοί"]

remove_duplicates_based_on_another_file(input_file, compare_file, output_file, unwanted_substrings_ecdc)

# Example usage


unwanted_substrings_who = [
    "Français",
    "Русский",
    "Español",
    "WHO TEAM",
    "Community Readiness and Resilience (CRR)",
    "ItemListOrderAscending",
    "< Go back to all Coronavirus disease 2019 Q&As Coronavirus disease (COVID-19):",
    "Ukraine Latest Disease Outbreak News Situation reports Weekly Epidemiological",
    "function(a,e,b,f,g,c,d){a[b]=a[b]",
    "(function () )(); (function () )(); Subscribe to the WHO newslette",
    " @font-face @font-face @font-face",
    "ck to all Coronavirus disease 2019 Q&As Coronavirus disease (COVID-19):",
    "Manage cookies Cookies"
]

'''
import json

def correct_punctuation_spacing(text):
    import re
    # Regular expression to match punctuation followed by a non-whitespace character
    corrected_text = re.sub(r'([:;.,\?])([^\s])', r'\1 \2', text)
    return corrected_text

input_file = "data/gov_covid.jsonl"
output_file = "data/gov_covid_clean.jsonl"

with open(input_file, 'r', encoding='utf-8') as infile, open(output_file, 'w', encoding='utf-8') as outfile:
    for line in infile:
        data = json.loads(line)
        if 'content' in data:
            data['content'] = correct_punctuation_spacing(data['content'])
        outfile.write(json.dumps(data, ensure_ascii=False) + '\n')

print(f'Processed file saved as {output_file}')

# Example usage
'''
input_file = '/home/pgriziotis/thesis/covid-va-chatbot/qa-subsystem/src/external_data/crawled_docs_who_en.jsonl'
output_file = '/home/pgriziotis/thesis/covid-va-chatbot/qa-subsystem/src/external_data/crawled_doc_who_en_clean.jsonl'


remove_duplicates_and_unwanted(input_file, output_file, unwanted_substrings=unwanted_substrings)
print("Duplicate removal complete. Check the output file.")
'''