#!/bin/bash
#SBATCH --ntasks=6
#SBATCH --mem=14G
#SBATCH --gres=gpu:1
#SBATCH -t 6:00:00
#SBATCH -o /media/data/smartens/output/out_%j.log
#SBATCH -e /media/data/smartens/output/error_%j.log

module load python/3.6.7

source /media/data/smartens/env/bin/activate

# copy data to temp job dir
ROI=OriginalData
MY_TMP_DIR=/slurmtmp/${SLURM_JOB_USER}.${SLURM_JOB_ID}
cp -r /media/data/smartens/data/Hammers_n30r83unzipped/${ROI}/* ${MY_TMP_DIR}

# run python script
python /media/data/smartens/code/Cropping/ReferenceCropHammers.py ${MY_TMP_DIR} ${SLURM_JOB_ID}

#cp /media/data/smartens/output/crop_out_${SLURM_JOB_ID}.log /media/data/smartens/results/${SLURM_JOB_ID}*/
#cp /media/data/smartens/output/crop_error_${SLURM_JOB_ID}.log /media/data/smartens/results/${SLURM_JOB_ID}*/

#chown smartens /media/data/smartens/results/${SLURM_JOB_ID}*/

deactivate
