#!/bin/bash

# List of top_k ranker and top_k values
top_k_ranker_params=(2 4 6 8 10 12 14)


# Loop through top_k_ranker and reader_top_k combinations
for top_k in "${top_k_ranker_params[@]}"

    do
            # Execute Python script with the parameters
            python3 run_experiments_on_theano_data.py \
                --pipeline extractive \
                --params "{\"BM25Retriever\": {\"top_k\": 10}, \"DenseRetriever\": {\"top_k\": 10}, \"Ranker\": {\"top_k\": $top_k}, \"Reader\": {\"top_k\": $top_k}}" \
                --output_name "extractive_${top_k}_${top_k}"
    done
done
