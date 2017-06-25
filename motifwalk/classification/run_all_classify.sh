#!/bin/bash
FILES=/home/gear/BigFiles/step_save_2017_06_26/cora_1498325969.910316/
for f in $FILES/*
do
  echo $f
  python simple_clf.py -e $f -d cora -tr 0.05 --metric accuracy -c 'Logistic Regression CV' -rs 2
done

