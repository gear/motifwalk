#!/usr/bin/bash

source activate network
python motifwalk.py -d cora -w undirected -l 80 -k 32 -t 1000000 -b 4096 -m skipgram -ws 5 -nn 15 -ns 2 -lr 0.001 --log_step 4000 --save_step 10000 --device="/cpu:0" --save_loc ./emb/
