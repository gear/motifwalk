#!/usr/bin/bash

source activate network
python motifwalk.py -d cora -w undirected -l 80 -k 32 -t 100000 -nw 10 -b 1024 -m skipgram -ws 5 -nn 15 -ns 2 -lr 0.0001 --log_step 4000 --save_step 10000 --device="/cpu:0" --save_loc ./emb/
python motifwalk.py -d cora -w undirected -l 80 -k 32 -t 100000 -nw 10 -b 1024 -m skipgram -ws 5 -nn 15 -ns 2 -lr 0.001 --log_step 4000 --save_step 10000 --device="/cpu:0" --save_loc ./emb/
python motifwalk.py -d cora -w undirected -l 80 -k 32 -t 100000 -nw 10 -b 1024 -m skipgram -ws 5 -nn 15 -ns 2 -lr 0.01 --log_step 4000 --save_step 10000 --device="/cpu:0" --save_loc ./emb/
python motifwalk.py -d cora -w undirected -l 80 -k 32 -t 100000 -nw 10 -b 1024 -m skipgram -ws 5 -nn 15 -ns 2 -lr 0.1 --log_step 4000 --save_step 10000 --device="/cpu:0" --save_loc ./emb/
python motifwalk.py -d cora -w undirected -l 80 -k 8 -t 100000 -nw 10 -b 1024 -m skipgram -ws 5 -nn 15 -ns 2 -lr 0.01 --log_step 4000 --save_step 10000 --device="/cpu:0" --save_loc ./emb/
python motifwalk.py -d cora -w undirected -l 80 -k 16 -t 100000 -nw 10 -b 1024 -m skipgram -ws 5 -nn 15 -ns 2 -lr 0.01 --log_step 4000 --save_step 10000 --device="/cpu:0" --save_loc ./emb/
python motifwalk.py -d cora -w undirected -l 80 -k 32 -t 100000 -nw 10 -b 1024 -m skipgram -ws 5 -nn 15 -ns 2 -lr 0.01 --log_step 4000 --save_step 10000 --device="/cpu:0" --save_loc ./emb/
python motifwalk.py -d cora -w undirected -l 80 -k 64 -t 100000 -nw 10 -b 1024 -m skipgram -ws 5 -nn 15 -ns 2 -lr 0.01 --log_step 4000 --save_step 10000 --device="/cpu:0" --save_loc ./emb/
python motifwalk.py -d cora -w undirected -l 80 -k 128 -t 100000 -nw 10 -b 1024 -m skipgram -ws 5 -nn 15 -ns 2 -lr 0.01 --log_step 4000 --save_step 10000 --device="/cpu:0" --save_loc ./emb/
python motifwalk.py -d cora -w undirected -l 80 -k 256 -t 100000 -nw 10 -b 1024 -m skipgram -ws 5 -nn 15 -ns 2 -lr 0.01 --log_step 4000 --save_step 10000 --device="/cpu:0" --save_loc ./emb/
python motifwalk.py -d cora -w undirected -l 80 -k 512 -t 100000 -nw 10 -b 1024 -m skipgram -ws 5 -nn 15 -ns 2 -lr 0.01 --log_step 4000 --save_step 10000 --device="/cpu:0" --save_loc ./emb/
python motifwalk.py -d cora -w undirected -l 80 -k 64 -t 10000 -nw 10 -b 1024 -m skipgram -ws 5 -nn 15 -ns 2 -lr 0.01 --log_step 4000 --save_step 10000 --device="/cpu:0" --save_loc ./emb/
python motifwalk.py -d cora -w undirected -l 80 -k 64 -t 20000 -nw 10 -b 1024 -m skipgram -ws 5 -nn 15 -ns 2 -lr 0.01 --log_step 4000 --save_step 10000 --device="/cpu:0" --save_loc ./emb/
python motifwalk.py -d cora -w undirected -l 80 -k 64 -t 40000 -nw 10 -b 1024 -m skipgram -ws 5 -nn 15 -ns 2 -lr 0.01 --log_step 4000 --save_step 10000 --device="/cpu:0" --save_loc ./emb/
python motifwalk.py -d cora -w undirected -l 80 -k 64 -t 100000 -nw 10 -b 1024 -m skipgram -ws 5 -nn 15 -ns 2 -lr 0.01 --log_step 4000 --save_step 10000 --device="/cpu:0" --save_loc ./emb/
python motifwalk.py -d cora -w undirected -l 80 -k 64 -t 1000000 -nw 10 -b 1024 -m skipgram -ws 5 -nn 15 -ns 2 -lr 0.01 --log_step 4000 --save_step 10000 --device="/cpu:0" --save_loc ./emb/
python motifwalk.py -d cora -w undirected -l 80 -k 64 -t 2000000 -nw 10 -b 1024 -m skipgram -ws 5 -nn 15 -ns 2 -lr 0.01 --log_step 4000 --save_step 10000 --device="/cpu:0" --save_loc ./emb/
python motifwalk.py -d cora -w undirected -l 80 -k 64 -t 20000000 -nw 10 -b 1024 -m skipgram -ws 5 -nn 15 -ns 2 -lr 0.01 --log_step 4000 --save_step 10000 --device="/cpu:0" --save_loc ./emb/
