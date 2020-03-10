#!/bin/bash
#SBATCH --ntasks=6
#SBATCH --mem=14G
#SBATCH --gres=gpu:2
#SBATCH -t 24:00:00
#SBATCH -o /media/data/smartens/output/lobe_BL_batch_out_%j.log
#SBATCH -e /media/data/smartens/output/lobe_BL_batch_error_%j.log

module load python/3.6.7
module load tensorflow/1.12.0

source /media/data/smartens/env/bin/activate

# copy data to temp job dir
ROI=Hammers
MY_TMP_DIR=/slurmtmp/${SLURM_JOB_USER}.${SLURM_JOB_ID}
cp -r /media/data/smartens/data/datasize176_208_160/${ROI}/* ${MY_TMP_DIR}

# run python script
python /media/data/smartens/code/MainIterativeLobes3.py Baseline ${MY_TMP_DIR} ${SLURM_JOB_ID}

#cp /media/data/smartens/output/lobe_BL_batch_out_${SLURM_JOB_ID}.log /media/data/smartens/results/${SLURM_JOB_ID}*/
#cp /media/data/smartens/output/lobe_BL_batch_error_${SLURM_JOB_ID}.log /media/data/smartens/results/${SLURM_JOB_ID}*/

#chown -R 1020:1014 /media/data/smartens/results/${SLURM_JOB_ID}*/
#chown -R 1020:1014 /media/data/smartens/data/datasize176_208_160/hyperparametertuning/Baseline/${SLURM_JOB_ID}*/

deactivate
