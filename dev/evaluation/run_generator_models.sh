#!/bin/bash

# List of top_k and max_new_tokens values
top_k_params=(10 15 20)  # Example values
max_new_tokens_params=(50 100 150)  # Example values

# Loop through top_k and max_new_tokens combinations
for top_k in "${top_k_params[@]}"
do
    for m_n_t in "${max_new_tokens_params[@]}"
    do
        # Execute Python script with the parameters
        python3 run_mlflow_experiment.py \
            --eval_type rag \
            --params "{\"Retriever\": {\"top_k\": 20}, \"Ranker\": {\"top_k\": $top_k}, \"Generator\": {\"max_new_tokens\": $m_n_t}}"
    done
done
