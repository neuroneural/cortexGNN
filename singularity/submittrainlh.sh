#!/bin/bash
#SBATCH -n 1
#SBATCH -c 4
#SBATCH --mem=30g
#SBATCH -p qTRDGPUH
#SBATCH --gres=gpu:V100:1
#SBATCH -t 3-00:00
#SBATCH -J pialnnl
#SBATCH -e /data/users2/washbee/pialnn/jobs/error%A.err
#SBATCH -o /data/users2/washbee/pialnn/jobs/out%A.out
#SBATCH -A psy53c17
#SBATCH --mail-type=ALL
#SBATCH --mail-user=washbee1@student.gsu.edu
#SBATCH --oversubscribe
#SBATCH --exclude=arctrdgn002


sleep 5s
echo "aa"
module load singularity/3.10.2
echo "bb"
singularity exec --nv --bind /data:/data/,/data/users2/washbee/speedrun/PialNN_fork:/pialnn,/data/users2/washbee/hcp-plis-subj-pialnn/:/subj /data/users2/washbee/containers/speedrun/pialnn_sr.sif /pialnn/singularity/trainlh.sh &
echo "CC"
wait

sleep 10s

