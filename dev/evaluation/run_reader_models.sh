#!/bin/bash

# List of model strings
model_strings=(
    "panosgriz/mdeberta-v3-base-squad2-covid-el" "panosgriz/mdeberta-v3-base-squad2-covid-el_small" "panosgriz/xlm-roberta-squad2-covid_el" "panosgriz/xlm-roberta-squad2-covid-el_small" "timpal0l/mdeberta-v3-base-squad2" "deepset/xlm-roberta-base-squad2")

for model_str in "${model_strings[@]}"
do
    python3 run_mlflow_experiment.py \
        --eval_type extractive \
        --reader_model "$model_str" \
        --params '{"Retriever": {"top_k":20}, "Ranker": {"top_k":10}, "Reader": {"top_k":10}}'
done