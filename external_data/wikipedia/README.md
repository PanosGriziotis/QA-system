# Setup and Processing Greek Wikipedia Dump

## Step 1: Activate Virtual Environment
Activate the virtual environment where all necessary dependencies are installed:

```bash
source QA-subsystem/venv/bin/activate
```

## Step 2: Download and Process Wikipedia Dump
Make sure you have sufficient disk space as the dump is large.

To download the Wikipedia Dump and Extract, run the following commands in your terminal:

```bash

chmod +x download_wiki_dump.sh
chmod +x extract_and_clean_wiki_dump.sh

./download_wiki_dump.sh && ./extract_and_clean_wiki_dump.sh elwiki-20240501-pages-articles-multistream.xml.bz2
```

This downloads the dump dated 01/05/2024 containing articles from the Greek Wikipedia in a compressed .bz2 format.
Ensure adequate disk space; the uncompressed size is approximately 3 GB.

The script extract_and_clean_wiki_dump.sh also installs wikiextractor, which cleans and extracts text from the Wikipedia dump. It saves each document as a JSON object line in a .txt file.

## Step 3: Filter and Convert Articles to Haystack Documents


``` bash
python3 filter_and_convert_to_docs.py --preprocess --apply_filter
```
Options:

--preprocess: Cleans the dump further, removing disambiguation pages and structured articles like lists and categories.

--apply_filter: Filters documents based on specified keywords. By default, it keeps the top 200 most relevant documents related to the keywords "πανδημία covid-19" (pandemic covid-19).

Customize top_k and keywords list in keyword_filtering.py if needed.

Output:

The processed and filtered documents are saved as JSON objects in wiki_doc.json in the parent directory of the script.
These documents are ready for indexing into an Elasticsearch document store instance.