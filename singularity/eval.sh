#!/bin/bash
. /opt/miniconda3/bin/activate pialnn
cd /pialnn

gnn="$1"
layers="$2"

# Printing the arguments
echo "gnn: $gnn"
echo "layers: $layers"

model_location="$3"
if [[ $gnn -eq -1 ]]; then
    echo "eval.sh model is pialnn"
    layers=0
    python eval.py --data_path=/data-pialnn/ --hemisphere=lh --gnn_layers=$layers --gnnVersion=$gnn --model_location=$model_location --save_mesh_eval=True #removed --cortexGNN to set False 
else
    echo "eval.sh model is a pialgnn"

    python eval.py --data_path=/data-pialnn/ --hemisphere=lh --gnn_layers=$layers --gnnVersion=$gnn --cortexGNN=True --model_location=$model_location --save_mesh_eval=True #remove boolean values to set false most likely (see config)
fi