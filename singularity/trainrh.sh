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

if [[ $gnn -eq -1 ]]; then
    echo "trainrh.sh model is pialnn"
    layers=0
    python train.py --data_path=/subj/ --hemisphere=rh --save_mesh_train=True --gnn_layers=$layers --gnnVersion=$gnn --n_epoch=200

else
    echo "trainrh.sh model is a pialgnn"

    python train.py --data_path=/subj/ --hemisphere=rh --save_mesh_train=True --gnn_layers=$layers --gnnVersion=$gnn --cortexGNN=True --n_epoch=200
fi
echo "D"
echo "D"
