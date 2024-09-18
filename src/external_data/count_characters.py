import json

def count_characters_in_jsonl(file_path):
    total_characters = 0
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                # Load the JSON object from each line
                data = json.loads(line.strip())
                # Add the length of the 'content' field to the total count
                if 'content' in data:
                    total_characters += len(data['content'])
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON line: {e}")
    
    return total_characters

def count_characters_in_range(file_path, start_line, end_line):
    """
    Count characters in a text file from start_line to end_line (inclusive).
    
    Args:
        file_path (str): Path to the text file.
        start_line (int): The starting line number (1-based index).
        end_line (int): The ending line number (1-based index).
    
    Returns:
        int: Total number of characters in the specified range.
    """
    character_count = 0

    with open(file_path, 'r', encoding='utf-8') as file:
        for current_line_num, line in enumerate(file, start=1):
            if start_line <= current_line_num <= end_line:
                character_count += len(line)
            if current_line_num > end_line:
                break

    return character_count


# Example usage:
file_path = 'who_en.txt'  # Replace with your file path
start_line = 301  # Replace with the starting line number
end_line =  400  # Replace with the ending line number

count = count_characters_in_range(file_path, start_line, end_line)
print(f"Total characters from line {start_line} to line {end_line}: {count}")

# Example usage
#file_path = '/home/pgriziotis/thesis/covid-va-chatbot/qa-subsystem/src/external_data/crawled_docs_who_en_clean.jsonl'
#total_characters = count_characters_in_jsonl(file_path)
#print(f"Total characters in 'content' field: {total_characters}")