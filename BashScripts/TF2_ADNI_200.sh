#!/bin/bash
#SBATCH --ntasks=6
#SBATCH --mem=14G
#SBATCH --gres=gpu:1
#SBATCH -t 300:00:00
#SBATCH -o /media/data/smartens/output/TF2_ADNI_200_out_%j.log
#SBATCH -e /media/data/smartens/output/TF2_ADNI_200_error_%j.log

module load python/3.6.7
module load tensorflow/1.12.0

source /media/data/smartens/env/bin/activate

# copy data to temp job dir
ROI_test=Hammers
ROI_train=ADNI_200
MY_TMP_DIR=/slurmtmp/${SLURM_JOB_USER}.${SLURM_JOB_ID}
cp -r /media/data/smartens/data/datasize176_208_160/${ROI_test}/* ${MY_TMP_DIR}
cp -r /media/data/smartens/data/datasize176_208_160/ADNI/${ROI_train}/* ${MY_TMP_DIR}

# run python script
python /media/data/smartens/code/TransferLearning2.py ADNI_200 Temporal Frontal ${MY_TMP_DIR} ${SLURM_JOB_ID}

#cp /media/data/smartens/output/TF2_ADNI_200_out_${SLURM_JOB_ID}.log /media/data/smartens/results/${SLURM_JOB_ID}*/
#cp /media/data/smartens/output/TF2_ADNI_200_error_${SLURM_JOB_ID}.log /media/data/smartens/results/${SLURM_JOB_ID}*/

#chown -R 1020:1014 /media/data/smartens/results/${SLURM_JOB_ID}*/
#chown -R 1020:1014 /media/data/smartens/data/datasize176_208_160/hyperparametertuning/ADNI_200/${SLURM_JOB_ID}*/

deactivate
