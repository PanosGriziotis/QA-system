# Description

This directory contains the code required to train the QA system's models and run evaluation experiments.

## Set up

Create a virtual environment and 

Create a virtual env: 

```bash
python3 -m venv venv
```

and then activate:

```bash
source venv/bin/activate
```

Make sure you have installed all required dependencies from the ``requirements.txt`` file contained in the  ``src`` directory in the repo. 

```bash
pip intall -r src/requirements.txt
```

**Note**: If you have CUDA version 11.8 run the following command to install the required dependencies:

```bash
pip install torch==2.3.1 --index-url https://download.pytorch.org/whl/cu118
```

Before running any evaluation or training script you must ensure that an elasticsearch instance is running on `http://localhost:9200`

## Evaluation

## Component-wise Evaluation

```bash
cd evaluation/component_evaluation
```
### Retrievers
To evaluate and compare different retrievers, run the following command. This will generate evaluation reports on the NPHO dataset. Alternatively, you can pass the `--xquad` flag to evaluate on the XQuAD dataset. The reports will include metrics such as Recall@k, MRR@k, NDCG@k, and MAP@k, across a range of k values from 1 to 20.

```bash
python3 run_experiments.py --retriever --npho
```


### Readers 

To evaluate different extractive reader models, first initialize an MLflow server on `localhost` by running the following command:


```bash
mlflow server --host 0.0.0.0 --port 5001

```

Then, run the following script:


```bash
./eval_reader_models_mlflow.sh
```

You can see detailed result reports on the MLFLOW UI. 

## End-to-end evaluation

```bash
cd evaluation/system_evaluation
```

To evaluate the QA system in an end-to-end setup with a dataset of user queries, you must first generate the dataset. To create a dataset with randomly selected queries from Theano's NLU training examples, run:

```bash
python3 test_data/generate_dataset.py

```

Next, to run both the generative and extractive QA pipelines on the dataset of queries with different top_k Reranker values, use:

```bash
./run_qa_pipelines_on_queries.sh

```

This will create JSON files in a results folder, with each file containing the full responses of the QA system on the query datasets for specific pipelines and top_k values.

To generate intuitive reports that include average scores of context relevance, answer relevance, and groundedness for each JSON file, first run the following script to compute metrics on generated results. Don't forget to pass the directory path of the saved results files as an argument:

```bash
python3 calculate_eval_metrics.py <RESULTS_DIR_PATH>

```

Then, generate the reports by running:

```bash
python3 get_eval_reports <RESULTS_DIR_PATH>

```

## Training



To train an SBERT Dense Retriever based on a multilingual Sentence-embedding model, using query-answer examples from covid-el-small and query_answer_pairs datasets, run the following command:

```bash
python3 training/adapt_sbert_retriever.py 
```

For generating a dataset in the Dense Passage Retriever (DPR) format run the following:

```bash
python3 utils/create_dpr_dataset.py

```
Then, run the training script with the appropriate flags, to train two GREEK-BERT encoders on generated data.


```bash
python3 training/train_dpr_retriever.py --train_file <TRAIN_FILE_PATH> --doc_dir <DIRECTORY_OF_TRAIN_FILE>
```

**Note:** You cannot train dpr model on a single GPU device

To train an extractive Reader model, run:

```bash
python3 training/train_reader.py

```

You can specify hyperparameters, such as batch size and number of training epochs, like this:

```bash
python3 training/train_reader.py -m <MODEL_NAME_OR_PATH> --batch_size <NUMBER_OF_BATCH_SIZE> --epochs <NUMBER_OF_TRAIN_EPOCHS>
```
