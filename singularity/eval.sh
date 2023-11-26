#!/bin/bash
. /opt/miniconda3/bin/activate pialnn
cd /pialnn

gnn="$1"
layers="$2"

# Printing the arguments
echo "gnn: $gnn"
echo "layers: $layers"

model_location="$3"
python eval.py --data_path=/data-pialnn/ --hemisphere=lh --gnn_layers=$layers --gnnVersion=$gnn --cortexGNN=True --model_location=$model_location 

#--save_mesh_eval=False just remove to set false i think. 