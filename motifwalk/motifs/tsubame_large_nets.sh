#!/bin/sh
#
# Compute z_score for large networks

cd /home/usr8/15M54097/motifwalk/motifwalk/motifs
source activate network

python z_score.py -d amazon -f amazon_zscore_ -k 4 -n 10 --dloc ./../../data
python z_score.py -d youtube -f youtube_zscore_ -k 4 -n 10 --dloc ./../../data
python z_score.py -d blogcatalog -f blogcatalog_zscore_ -k 4 -n 10 --dloc ./../../data

