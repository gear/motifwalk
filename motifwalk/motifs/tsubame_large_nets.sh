#!/bin/sh
#
# Compute z_score for large networks

cd /home/usr8/15M54097/motifwalk/motifwalk/motifs
source activate network

python z_score.py -d cora -f cora_zscore_ -k 4 -n 0 --dloc ./../../data

