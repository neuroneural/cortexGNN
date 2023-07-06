#!/bin/bash
#SBATCH -n 1
#SBATCH -c 4
#SBATCH --mem=60g
#SBATCH -p qTRDGPU
#SBATCH --gres=gpu:RTX:1
#SBATCH -t 1-00:00
#SBATCH -J pialnnl
#SBATCH -e jobs/error%A.err
#SBATCH -o jobs/out%A.out
#SBATCH -A psy53c17
#SBATCH --mail-type=ALL
#SBATCH --mail-user=washbee1@student.gsu.edu
#SBATCH --oversubscribe
#SBATCH --exclude=arctrdgn002,arctrddgx001


sleep 5s
echo "aa"
source /usr/share/lmod/lmod/init/bash
module use /application/ubuntumodules/localmodules
module load singularity/3.10.2
echo "bb"
singularity exec --nv --bind /data,/data/users2/washbee/speedrun/hcp-plis-subj-pialnn-201818:/data-pialnn/,/data/users2/washbee/speedrun/PialNN_fork:/pialnn,/data/users2/washbee/speedrun/hcp-plis-subj-pialnn-rp:/subj/ /data/users2/washbee/containers/speedrun/pialnn_bm2_sandbox/ /pialnn/singularity/benchmarks/bm.eval.sh &
echo "CC"
wait

sleep 10s

