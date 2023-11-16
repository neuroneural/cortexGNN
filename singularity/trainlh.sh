#!/bin/bash
echo "A"
. /opt/miniconda3/bin/activate pialnn
echo "B"
cd /pialnn
echo "C"
python train.py --data_path=/subj/ --hemisphere=lh
echo "D"
