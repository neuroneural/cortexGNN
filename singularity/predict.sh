#!/bin/bash
. /opt/miniconda3/bin/activate pialnn
cd /pialnn
#python predict.py --data_path=/data-pialnn/ --hemisphere=lh --save_mesh_eval=True
python predict.py --data_path=/subj/ --hemisphere=lh --save_mesh_eval=True

