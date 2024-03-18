#!/bin/bash
#SBATCH -n 1
#SBATCH -c 4
#SBATCH --mem=100g
#SBATCH -p qTRDGPU
#SBATCH --gres=gpu:RTX:1
#SBATCH -t 1-00:00
#SBATCH -J pgnevlh
#SBATCH -e jobs/error%A_%a.err
#SBATCH -o jobs/out%A_%a.out
#SBATCH -A psy53c17
#SBATCH --mail-type=ALL
#SBATCH --mail-user=washbee1@student.gsu.edu
#SBATCH --oversubscribe
#SBATCH --nodelist=arctrdagn007

sleep 5s
echo "aa"
source /usr/share/lmod/lmod/init/bash
module use /application/ubuntumodules/localmodules
module load singularity/3.10.2
echo "bb"

gnn=(0 0 0 0 0 0 -1)
layers=(2 3 4 5 6 7 -1)

# Compute the combination from the SLURM_ARRAY_TASK_ID

# Fetch the actual values from the arrays
gnn=${gnn[$SLURM_ARRAY_TASK_ID]}
layers=${layers[$SLURM_ARRAY_TASK_ID]}

# Now use value1 and value2 as needed
echo "Combination: $gnn, $layers"
# Add your code here that uses value1 and value2
#base="/data/users2/washbee/websurf/cortexGNN/ckpts/model/lh/"
base="/data/users2/washbee/websurf/cortexGNN/ckpts/model/rh/"
model=""

if [[ $gnn -eq 0 && $layers -eq 2 ]]; then
    model="PialGCN_GNNlayers2_mse_whitein_full_model_rh_best.pt"
elif [[ $gnn -eq 0 && $layers -eq 3 ]]; then
    model="PialGCN_GNNlayers3_mse_whitein_full_model_rh_best.pt"
elif [[ $gnn -eq 0 && $layers -eq 4 ]]; then
    model="PialGCN_GNNlayers4_mse_whitein_full_model_rh_best.pt"
elif [[ $gnn -eq 0 && $layers -eq 5 ]]; then
    model="PialGCN_GNNlayers5_mse_whitein_full_model_rh_best.pt"
elif [[ $gnn -eq 0 && $layers -eq 6 ]]; then
    model="PialGCN_GNNlayers6_mse_whitein_full_model_rh_best.pt"
elif [[ $gnn -eq 0 && $layers -eq 7 ]]; then
    model="PialGCN_GNNlayers7_mse_whitein_full_model_rh_best.pt"
elif [[ $gnn -eq -1 ]]; then
    model="pialnn_model_rh_200epochs.pt"
fi

model_location="${base}${model}"
echo "Model location: $model_location"
sleep 10s
#/data/users2/washbee/websurf/pialnn_web_trsmall
#singularity exec --nv --bind /data,/data/users2/washbee/speedrun/hcp-plis-subj-pialnn-rp:/data-pialnn/,/data/users2/washbee/websurf/cortexGNN:/pialnn, /data/users2/washbee/containers/speedrun/pialnn_sr.sif /pialnn/singularity/benchmarks/bm.eval.sh $gnn $layers $model_location
singularity exec --nv --bind /data,/data/users2/washbee/websurf/pialnn_onesub:/data-pialnn/,/data/users2/washbee/websurf/cortexGNN:/pialnn, /data/users2/washbee/containers/speedrun/pialnn_sr.sif /pialnn/singularity/benchmarks/bm.eval.sh $gnn $layers $model_location
singularity exec --nv --bind /data,/data/users2/washbee/websurf/pialnn_onesub:/data-pialnn/,/data/users2/washbee/websurf/cortexGNN:/pialnn, /data/users2/washbee/containers/speedrun/pialnn_sr.sif /pialnn/singularity/benchmarks/bm.eval.sh $gnn $layers $model_location
singularity exec --nv --bind /data,/data/users2/washbee/websurf/pialnn_onesub:/data-pialnn/,/data/users2/washbee/websurf/cortexGNN:/pialnn, /data/users2/washbee/containers/speedrun/pialnn_sr.sif /pialnn/singularity/benchmarks/bm.eval.sh $gnn $layers $model_location
singularity exec --nv --bind /data,/data/users2/washbee/websurf/pialnn_onesub:/data-pialnn/,/data/users2/washbee/websurf/cortexGNN:/pialnn, /data/users2/washbee/containers/speedrun/pialnn_sr.sif /pialnn/singularity/benchmarks/bm.eval.sh $gnn $layers $model_location
singularity exec --nv --bind /data,/data/users2/washbee/websurf/pialnn_onesub:/data-pialnn/,/data/users2/washbee/websurf/cortexGNN:/pialnn, /data/users2/washbee/containers/speedrun/pialnn_sr.sif /pialnn/singularity/benchmarks/bm.eval.sh $gnn $layers $model_location


wait
echo "CC"

sleep 10s

