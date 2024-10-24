#!/bin/bash

# List of top_k and max_new_tokens values
top_k_params=(2 4 6 8 10 12 14) 
max_new_tokens_params=(150)  

for top_k in "${top_k_params[@]}"
do
    for m_n_t in "${max_new_tokens_params[@]}"
    do
        # Execute Python script with the parameters
        python3 run_experiments_on_theano_data.py \
            --pipeline rag \
            --params "{\"BM25Retriever\": {\"top_k\": 10}, \"DenseRetriever\": {\"top_k\": 10}, \"Ranker\": {\"top_k\": $top_k}, \"Generator\": {\"max_new_tokens\": $m_n_t}}" \
            --output_name "rag_${top_k}_${m_n_t}"
    done
done