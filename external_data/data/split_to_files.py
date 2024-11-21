import os
import json
import pandas as pd

dirs = ["./ecdc", "./eody", "./who", "./gov_covid", "./wiki_articles"]
output_dir = "./processed_files"
os.makedirs(output_dir, exist_ok=True)

meta = []

for dir in dirs:
    if not os.path.exists(dir):
        print(f"Directory {dir} does not exist.")
        continue
    
    files = os.listdir(dir)
    
    for file in files:
        file_path = os.path.join(dir, file)
        
        with open(file_path, "r") as fp:
            counter = 0
            for line in fp:
                counter += 1
                
                try:
                    line = json.loads(line)
                except json.JSONDecodeError:
                    print(f"Error decoding JSON in file {file}, line skipped.")
                    continue
                
                # Write the content to a new file
                if "eody" in os.path.splitext(file)[0]:
                    filename = f"eody_{counter}.txt"

                else:
                    filename = f"{os.path.splitext(file)[0]}_{counter}.txt"
                output_path = os.path.join(output_dir, filename)
                with open(output_path, "w") as out_fp:
                    out_fp.write(line.get("content", ""))
                
                # Extract metadata
                title = line.get("meta", {}).get("title")
                
                if "eody" in os.path.splitext(file)[0]:
                    source = "Hellenic National Public Health Organization"
                elif "who" in os.path.splitext(file)[0]:
                    source = "World Health Organization"
                elif "ecdc" in os.path.splitext(file)[0]:
                    source = "European Centre for Disease Prevention and Control"
                elif "gov" in os.path.splitext(file)[0]:
                    source = "covid19.gov.gr"
                elif "wiki" in os.path.splitext(file)[0]:
                    source = "Wikipedia"
                
                url = line.get("meta", {}).get("filename")
                
                # Append metadata
                meta.append({"filename": filename, "source": source, "url": url, "title": title})

# Create a DataFrame and save metadata to CSV
df = pd.DataFrame(meta)
df.to_csv("files_metadata.csv", index=False)