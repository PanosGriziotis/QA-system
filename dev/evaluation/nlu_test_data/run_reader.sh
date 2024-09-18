#!/bin/bash

# List of top_k and reader_top_k values
top_k_ranker_params=(15)  # Values for Ranker top_k
reader_top_k_values=(15)  # Values for Reader top_k

# Loop through top_k_ranker and reader_top_k combinations
for top_k in "${top_k_ranker_params[@]}"
do
    for reader_top_k in "${reader_top_k_values[@]}"
    do
        # Check if reader_top_k is less than or equal to top_k
        if [ "$reader_top_k" -le "$top_k" ]; then
            # Execute Python script with the parameters
            python3 run_experiments_on_theano_data.py \
                --pipeline extractive \
                --params "{\"Retriever\": {\"top_k\": 20}, \"Ranker\": {\"top_k\": $top_k}, \"Reader\": {\"top_k\": $reader_top_k}}" \
                --output_name "extractive_${top_k}_${reader_top_k}"
        fi
    done
done
