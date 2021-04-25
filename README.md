# Euller

## Setup

1. environment:

+ conda env create -f environment.yml


2. Directory:

+ src/lm_core -> GPT2 Language model wrapper code.
+ src/insights -> source code for computing insights out of validation data.
+ experiments -> Jupyter notebooks containing experiments and outputs.
+ data -> contains raw and processed data.


3. Experiment:

+ experiments/1_data_preparation -> pre-process news-corpus data
+ experiments/2_experiment_topk_2020 -> topk experiment ran on 2020 new-corpus data
+ experiments/2_experiment_topk_2010 -> topk experiment ran on 2010 new-corpus data
