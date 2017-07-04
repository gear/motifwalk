#!/bin/sh


cd /home/usr8/15M54097/motifwalk/motifwalk/motifs
t2sub -q V -W group_list=t2g-crest-deep-mu -l select=16:mpiprocs=16:mem=20gb -l place=scatter ./tsubame_large_nets.sh
