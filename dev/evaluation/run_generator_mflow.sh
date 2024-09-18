#!/bin/bash

# List of top_k and max_new_tokens values
top_k_params=(10 15 20)  # Add values, e.g., top_k_params=(10 20 30)
max_new_tokens_params=(50 100 150)  # Add values, e.g., max_new_tokens_params=(50 100 150)

# Loop through top_k and max_new_tokens combinations
for top_k in "${top_k_params[@]}"
do
    for m_n_t in "${max_new_tokens_params[@]}"
    do
        # Execute Python script with the parameters
        python3 run_mlflow_experiment.py \
            --pipeline rag \
            --params "{\"Retriever\": {\"top_k\": 20}, \"Ranker\": {\"top_k\": $top_k}, \"Generator\": {\"max_new_tokens\": $m_n_t}}" \
            --output_name "rag_${top_k}_${m_n_t}"
    done
done