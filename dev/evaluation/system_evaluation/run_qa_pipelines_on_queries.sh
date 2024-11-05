#!/bin/bash

apply_cr_threshold=false

# Parse command-line arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --cr_threshold) apply_cr_threshold=true ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

top_k_params=(2)
m_n_t=150

# Run the first script (rag pipeline)
for top_k in "${top_k_params[@]}"
do
    if [ "$apply_cr_threshold" = true ]; then
        threshold_param="0.17"  # Passes threshold as `None` if flag is true
    else
        threshold_param="0.0"  # Default threshold when flag is false
    fi

    python3 generate_answers.py  \
        --pipeline rag \
        --params "{\"BM25Retriever\": {\"top_k\": 10}, \"DenseRetriever\": {\"top_k\": 10}, \"Reranker\": {\"top_k\": $top_k}, \"GenerativeReader\": {\"max_new_tokens\": $m_n_t}, \"Responder\": {\"threshold\": $threshold_param}}" \
        --output_name "rag_${top_k}_${m_n_t}"
done

# Run the second script (extractive pipeline)
for top_k in "${top_k_params[@]}"
do
    if [ "$apply_cr_threshold" = true ]; then
        threshold_param="0.17"  # Passes threshold as `None` if flag is true
    else
        threshold_param="0.0"  # Default threshold when flag is false
    fi
    
    python3 generate_answers.py  \
        --pipeline extractive \
        --params "{\"BM25Retriever\": {\"top_k\": 10}, \"DenseRetriever\": {\"top_k\": 10}, \"Reranker\": {\"top_k\": $top_k},  \"ExtractiveReader\": {\"top_k\": $top_k}, \"Responder\": {\"threshold\": $threshold_param}}" \
        --output_name "extractive_${top_k}_${top_k}"
done