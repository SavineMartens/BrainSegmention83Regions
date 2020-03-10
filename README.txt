This document explains the code I've produced during my internship. All results are saved in the folder
hyperparametertuning and sectioned per lobe. Training Baseline (BL) means only including 22 Hammers in training. Baseline measurements means m0 (first scan). LR = learning rate.

FILES
3months_list.txt		the list of ADNI that were scanned at m0 and m3
ADNI_list.npy			All ADNI samples at baseline (m0) 
adni_list_full_labels.npy	All ADNI samples that had 83 labels after Esther's pipeline
adni_test_set.npy		test set that only contains healthy participants who have been measured 				at m0 and m3 in case we want to look at consistency (N=100)
adni_trainN00_set.npy		list of IDs for adding N adni samples to training
adni_train_set.npy		train list of ADNI samples measured at baseline (N=1611)
AllSubjectsDiagnosis.csv	file from which I extracted which subjects were healthy
ListTraining.npy		list of augmented Hammers samples, 25 per original training (550 in 					total) 
ListTraining.npy		list of augmented Hammers samples, 10 per original
requirements.txt		requirements for code (packages + versions)


PYTHON CODE
ADNI_list.py			the file with which I made the ADNI selections with N random indices 
ADNI_sample_visualisation.py	check to see if cropped and rotated ADNI was performed well
Augmentation.py			creates augmented Hammers samples with various transformation parameters
CheckLabels.py			code that produces adni_list_full_labels.npy. Has version for cluster and 					local computer, but need to indicate location!
ConcatenationLobes2Loss.py	
CountHealthyDuplicates.py	create ADNI test set and train set. uses AllSubjectsDiagnosis.csv and 
				adni_list_full_labels.npy
MainAllLobes.py			whole brain classification into 8 lobes. Uses the background in the loss 					function
MainLobes2.py			used to evaluate which trainingtype was most promising: BL, DA_ADNI, 					DA_Hammers, BL_ADNI, BL_ADNI_wo_test, BL_Skull, ADNIskull_200, ADNI_200etc
MainLobes3.py			used to train with final training type for all lobes and to assess which 					loss function can overcome issues with class imbalance
MakeTestDir.py			makes a directory with all training samples from a predefined list 
Metrics.py			Has functions to calculate several metrics: binary Dice, probabalistic 					Dice, Dice of just foreground, AVD
OverlapDecision.py		Two possible options for overlapping segmentations: majority vote and by 					using whole brain classification into 8 regions (MainAllLobes.py)
PlotLearningEpochsFromCSV.py	Makes a plot of 'train_history.csv', present in every Model directory
SegmentationComparison.py	code to visualize predicted segmentations and compare metrics
Shortlist.py			collection of random snippets of code I made for labels mostly
Statistics.py			code to statistically compare the three types of training types(box plots)
Testing.py			code to load pretrained network and test in on a specific dataset
TF_Testing.py			code I tried to write to test performance of pretrained network after 					doing transfer learning but I had issues with saving the transfer learned 					model
TransferLearning.py		Nested version of a transfer learning model, does not save model 					properly, frozen after first up-sampling layer
TransferLearning1.py		Nested version, frozen after last conv layer bottle neck
TransferLearning1a.py		NOT nested version, frozen after last conv layer bottle neck
TransferLearning2.py		Nested version, frozen before last conv layer bottle neck
TransferLearning2a.py		NOT nested version, frozen before last conv layer bottle
UNet3D.py			Bo's U-Net code with several loss functions and dice accuracies I tried
Visualisation.py		several codes to depict MRI scans and segmentations
Visualisation2.py		code from previous python file turned to classes
VolumeRegionComparison.py	code I used to assess which areas are smallest of a lobe and what the 					areas specific relative volumetric ratio was

BASH SCRIPTS FOR GPU CLUSTER
ADNI_N00.sh 			run MainLobes2.npy to add N ADNI samples to training with LR in title 
ADNIskull_N00.sh		same as ADNI_N00.sh but with skull stripped data. At 300 samples a letter 					is added because it was part of a cross validation
Appendix.sh			run final version MainLobes3.py on Appendices lobe with 300 ADNI
BL_0.01.sh			used this to assess if using only Hammers (Baseline) with varying LR
BL_ADNI_0.001.sh		training using 22 Hammers (BL) AND 1711 ADNI samples 
BL_ADNI_wo_test_0.001.sh	training using 22 Hammers (BL) AND 1611 ADNI samples
BL_batch_size.sh		assessed if using 2 vs 1 batch size would improve much on baseline
BL_Hammers10_001.sh		training using 22 Hammers AND 10 augmented hammers samples per original
BL_skull_0.01.sh		training BL on skullstripped scans with varying LR
CentralLeft.sh			run final version MainLobes3.py on central left lobe with 300 ADNI
CentralRight.sh			run final version MainLobes3.py on central right lobe with 300 ADNI
CheckLabels.sh			runs CheckLabels.py to produce adni_list_full_labels.npy
ConcatenationLobes.sh		runs ConcatenationLobes2Loss.py, you need to tell which dataset it should 					use
DA_ADNI_0.01.sh			uses MainLobes2.npy to train on only 1711 ADNI samples with varying LR
DA_Hammers_0.01.sh		uses MainLobes2.npy to train on only augmented Hammers samples (25 per 					orginal) with varying LR
FrontalLeft.sh			run final version MainLobes3.py on frontal left lobe with 300 ADNI
FrontalLeftDL.sh		same as FrontalLeft.sh but with Dice loss (whereas that one has -Dice as 					loss function)
FrontalLeftWDL.sh		FrontalLeft.sh with weighted dice loss = largest volume/individual volumes
FrontalLeftWDL2.sh		FrontalLeft.sh with weights = 1/individual volumes
FrontalLeftWDL3.sh		FrontalLeft.sh with weights = background volume / individual volumes
FrontalRight.sh			run final version MainLobes3.py on frontal right lobe with 300 ADNI
make_list.sh			runs ADNI_list.py
OccipitalParietal.sh		run final version MainLobes3.py on occipital and parietal lobes with 300 					ADNI samples
run_augmentation.sh		run Augmentation.py on cluster
TemporalLeft.sh			run final version MainLobes3.py on temporal left lobe with 300 ADNI
TemporalRight.sh		run final version MainLobes3.py on temporal right lobe with 300 ADNI
Test_xxxx.sh			runs Testing.py with the trained networks listed in title
Test_Dir.sh			runs MakeTestDir.py on cluster
Test_TF_ADNI.sh			runs TF_Testing on cluster
TFN_ADNI_200.sh			runs a version of transfer learning python 
WholeBrainskull_200.sh		runs MainAllLobes.py with 200 ADNI samples
WholeBrainskull_300.sh		runs MainAllLobes.py with 300 ADNI samples



