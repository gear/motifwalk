#!/bin/sh


cd /home/usr8/15M54097/motifwalk/motifwalk/motifs/tsubame
t2sub -q V -W group_list=t2g-crest-deep-mu -et 3 -l walltime=96:00:00 -l select=1:ncpus=16:mem=23gb -l place=scatter ./tsubame_large_nets.sh
