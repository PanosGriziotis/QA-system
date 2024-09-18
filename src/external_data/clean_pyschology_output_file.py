import json

def clean_text_symbols_in_place(output_file, symbols_to_remove):
    # Read the content of the output file
    with open(output_file, 'r', encoding='utf-8') as infile:
        lines = infile.readlines()
    
    # Process and clean the content
    cleaned_lines = []
    for line in lines:
        try:
            data = json.loads(line.strip())
            if 'content' in data:
                content = data['content']
                
                # Remove specified symbols from content
                for symbol in symbols_to_remove:
                    content = content.replace(symbol, '')

                # Update content after cleaning
                data['content'] = content
                
            cleaned_lines.append(json.dumps(data, ensure_ascii=False) + '\n')
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON line: {e}")

    # Write the cleaned content back to the same file
    with open(output_file, 'w', encoding='utf-8') as outfile:
        outfile.writelines(cleaned_lines)

# Example usage:
output_file = 'data/output_psychology.jsonl'
symbols_to_remove = ["•", ""]

clean_text_symbols_in_place(output_file, symbols_to_remove)