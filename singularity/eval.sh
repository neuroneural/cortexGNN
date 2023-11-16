#!/bin/bash
. /opt/miniconda3/bin/activate pialnn
cd /pialnn
python eval.py --data_path=/data-pialnn/ --hemisphere=lh --save_mesh_eval=True

