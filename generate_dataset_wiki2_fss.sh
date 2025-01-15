#!/bin/bash

# This script should be called by `source` command with the correct conda environment.
python generate_dataset_wiki2.py -n 30 50 70 100 145 190 270 -m 40 60 80 110 155 200 280 -o dataset_wiki_fss
python generate_dataset_wiki2.py -n 30 50 70 100 145 190 270 -m 40 60 80 110 155 200 280 -o dataset_wiki_fss_decap --strip_headers