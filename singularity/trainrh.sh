#!/bin/bash
echo "A"
. /opt/miniconda3/bin/activate pialnn
echo "B"
cd /pialnn
echo "C"
echo "rh training"

gnn="$1"
layers="$2"

# Printing the arguments
echo "gnn: $gnn"
echo "layers: $layers"

python train.py --data_path=/subj/ --hemisphere=rh --save_mesh_train=True --gnn_layers=$layers --gnnVersion=$gnn --cortexGNN=True
echo "D"
