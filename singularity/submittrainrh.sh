#!/bin/bash
#SBATCH -n 1
#SBATCH -c 4
#SBATCH --mem=50g
#SBATCH -p qTRDGPU
#SBATCH --gres=gpu:RTX:1
#SBATCH -t 4-00:00
#SBATCH -J cortNNR
#SBATCH -e jobs/error%A_%a.err
#SBATCH -o jobs/out%A_%a.out
#SBATCH -A psy53c17
#SBATCH --mail-type=ALL
#SBATCH --mail-user=washbee1@student.gsu.edu
#SBATCH --oversubscribe


sleep 5s
echo "aa"
module load singularity/3.10.2
echo "bb"
gnn=(0 1)
layers=(2 3 4)

# Compute the combination from the SLURM_ARRAY_TASK_ID
index1=$(($SLURM_ARRAY_TASK_ID / 3))
index2=$(($SLURM_ARRAY_TASK_ID % 3))

# Fetch the actual values from the arrays
gnn=${gnn[$index1]}
layers=${layers[$index2]}

# Now use value1 and value2 as needed
echo "Combination: $gnn, $layers"
# Add your code here that uses value1 and value2

#/data/users2/washbee/speedrun/pialnn_trainval
#/data/users2/washbee/websurf/pialnn_web_trsmall/
#singularity exec --nv --bind /data:/data/,/data/users2/washbee/websurf/cortexGNN:/pialnn,/data/users2/washbee/websurf/pialnn_web_trsmall/:/subj /data/users2/washbee/containers/speedrun/pialnn_sr.sif /pialnn/singularity/trainrh.sh $gnn $layers
singularity exec --nv --bind /data:/data/,/data/users2/washbee/websurf/cortexGNN:/pialnn,/data/users2/washbee/speedrun/pialnn_trainval/:/subj /data/users2/washbee/containers/speedrun/pialnn_sr.sif /pialnn/singularity/trainrh.sh $gnn $layers
echo "CC"

sleep 5s

