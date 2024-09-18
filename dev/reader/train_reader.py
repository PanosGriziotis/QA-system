from typing import List
import os
import logging
import argparse
import json
from pathlib import Path
import os
from haystack.nodes import FARMReader
from haystack.utils import EarlyStopping
from haystack.document_stores import InMemoryDocumentStore
from count_squad import get_squad_dataset_counts


#TODO FIX THE LOGGINGS. Either print the training loggings in a separate txt file via console or install haystack folder and in haystack/modeling/training/base.py turn the MLflow parameter to True
#TODO MAKE A JSON file that will give all configuration parameters to our function
#TODO see early stopping error again
#TODO find a better dev/train split methodology

def fine_tune_reader_model (config: dict):

    if config["early_stopping"] == True:
        config["early_stopping"] = EarlyStopping(metric='top_n_accuracy',
                                                 save_dir=config["save_dir"],
                                                 mode='max',
                                                 patience=10,
                                                 min_delta=0.001,
                                                 min_evals=0,
                                                )    
    else:
        config["early_stopping"] = None

    # load baseline model
    reader = FARMReader(config["model_name_or_path"])
    # run training method
    reader.train(
        data_dir = config["data_dir"],
        train_filename = config["train_filename"],
        dev_filename = config["dev_filename"],
        use_gpu = config ["use_gpu"],
        devices = config ["devices"],
        batch_size= config["batch_size"],
        n_epochs =config["n_epochs"],
        max_seq_len = config["max_seq_len"],
        dev_split = config["dev_split"],
        evaluate_every = config["evaluate_every"],
        save_dir=config["save_dir"],
        num_processes = config["num_processes"],
        grad_acc_steps = config ["grad_acc_steps"],
        early_stopping = config["early_stopping"]
        )


parser = argparse.ArgumentParser()
parser.add_argument("-m", '--model_name_or_path', type=str)
parser.add_argument("-d",'--data_dir', type=str)
parser.add_argument("-t",'--train_filename', type=str)
parser.add_argument('--batch_size', type=int, default= 8)
parser.add_argument('--n_epochs', type=int, default=3)
parser.add_argument('--learning_rate', type = float, default = 3e-5)
parser.add_argument('--dev_filename', type=str, default= None)
parser.add_argument('--use_gpu', type=bool, default= True)
parser.add_argument('--devices', type=list, default = None)
parser.add_argument('--max_seq_len', type=str, default=384)
parser.add_argument('--dev_split', type = float, default=0)
parser.add_argument('--evaluate_every', type = int, default=300)
parser.add_argument('--save_dir', type = str, default = os.path.join (os.getcwd(), "model"))
parser.add_argument('--num_processes', type = int, default = None)
parser.add_argument('--grad_acc_steps', type = int, default = 2)
parser.add_argument('--early_stopping', type = bool, default= False)

args = parser.parse_args()

if __name__ == '__main__': 
    args = parser.parse_args()

    config = vars(args)
    
    save_dir = config ["save_dir"]

    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    
    training_log_file = os.path.join(save_dir, 'training.log')

    with open (os.path.join(save_dir, "train_config.json"), "w") as file:
        info = [get_squad_dataset_counts(file) for file in [config["train_filename"], config["dev_filename"]]]
        info.append (config)
        json.dump(info, fp=file)

    logging.basicConfig(filename=training_log_file, filemode='w', format="%(levelname)s - %(name)s -  %(message)s", level=logging.WARNING)
    logging.getLogger("haystack").setLevel(logging.INFO)

    fine_tune_reader_model(vars(args))