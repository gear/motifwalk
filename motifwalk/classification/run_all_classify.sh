#!/bin/bash
FILES=/home/gear/Dropbox/CompletedProjects/motifwalk/emb/step_save/polblogs_1498394461.0723994
for f in $FILES/*
do
  echo $f
  python simple_clf.py -e $f -d polblogs -tr 0.05 --metric accuracy -c 'Logistic Regression CV' -rs 2
done

