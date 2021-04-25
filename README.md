# Euller

## Setup

1. environment:

+ conda env create -f environment.yml


2. Directory:

+ [src/lm_core](https://github.com/Lawliet19189/Euller/tree/main/src/lm_core) -> GPT2 Language model wrapper code.
+ [src/insights](https://github.com/Lawliet19189/Euller/tree/main/src/insights) -> source code for computing insights out of validation data.
+ [experiments](https://github.com/Lawliet19189/Euller/tree/main/experiments) -> Jupyter notebooks containing experiments and outputs.
+ [TODO] data -> contains raw and processed data.


3. Experiment:

+ [experiments/1_data_preparation](https://github.com/Lawliet19189/Euller/blob/main/experiments/1_data_preparation.ipynb) -> pre-process news-corpus data
+ [experiments/2_experiment_topk_2020](https://github.com/Lawliet19189/Euller/blob/main/experiments/2_experiment_topk_2020.ipynb) -> topk experiment ran on 2020 new-corpus data
+ [experiments/2_experiment_topk_2010](https://github.com/Lawliet19189/Euller/blob/main/experiments/3_experiment_topk_2010.ipynb) -> topk experiment ran on 2010 new-corpus data
