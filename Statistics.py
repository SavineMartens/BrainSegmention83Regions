import os
import numpy as np
import time
from Shortlist import *
import matplotlib.pyplot as plt
from datetime import datetime
from Visualisation import *
from Configuration2 import *
import sys
import pandas as pd
import seaborn as sns
from scipy import stats

######################################################################################################
# Boxplots with dots
training_types = ['Baseline', 'BL_skull', 'BL_ADNI_wo_test', 'BL_Hammers']
location = 'local'
data_collected = {}
if location == 'cluster':
    lobe = 'Temporal'
    lateralisation = 'Left'
    it_val = 0.01
    hyper_parameter_to_evaluate = 'Initial_LR'
    save_path = '/media/data/smartens/data/datasize176_208_160/hyperparametertuning'
    test_list = np.load('/media/data/smartens/data/datasize176_208_160/ADNI/adni_test_set.npy')
if location == 'local':
    path = '/home/jara/PycharmProjects/GPUcluster/DiceCoefficientResultsforADNI_'
    test_list = np.load('/AdditionalFiles/adni_test_set.npy')

df = pd.DataFrame({'ID': test_list})

dc = []
labels = []

for training_type in training_types:
    if location == 'cluster':
        parameter_save = os.path.join(save_path, training_type, lobe, lateralisation +
                              hyper_parameter_to_evaluate + 'E' + str(n_epoch))
        pred_seg_path = os.path.join(parameter_save, 'PredictedSegmentation' + str(training_type) + str(it_val))
        data = np.load(os.path.join(pred_seg_path, 'DiceCoefficientResultsforADNI.npy'))
    if location == 'local':
        data = np.load(path + training_type + '.npy')
        label = len(test_list) * [training_type]
    df[training_type] = data
    it = np.arange(len(data))
    for (t, d, l) in zip(it, data, label):
        # print(t, d, l)
        dc.append(d)
        labels.append(l)

df2 = pd.DataFrame({'training_type': labels, 'Dice': dc})

# paired t-test BL with ADNI
BL_Ham = stats.ttest_rel(df['Baseline'], df['BL_Hammers'])
SS_ADNI = stats.ttest_rel(df['BL_skull'], df['BL_ADNI_wo_test'])
ADNI_Ham = stats.ttest_rel(df['BL_Hammers'], df['BL_ADNI_wo_test'])
SS_BL = stats.ttest_rel(df['Baseline'], df['BL_skull'])

# boxplot with dots
ax = sns.boxplot(x="training_type", y="Dice", data=df2, showfliers = False)
ax = sns.swarmplot(x="training_type", y="Dice", data=df2, color=".25")

# statistical annotation to figure
x1, x2, x3, x4 = 0, 1, 2, 3
# BL + SS
y, h, col = max(df['BL_skull'].max(), df['Baseline'].max()) + 0.02, 0.05, 'k'
plt.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, c=col)
plt.text((x1+x2)*.5, y+h, "p = " + '{:.2e}'.format(SS_BL[1]), ha='center', va='bottom', color=col)
# # BL + ADNI
# y, h, col = max(df['BL_ADNI_wo_test'].max(), df['Baseline'].max()) + 0.04, 0.1, 'k'
# plt.plot([x1, x1, x3, x3], [y, y+h, y+h, y], lw=1.5, c=col)
# plt.text((x1+x3)*.5, y+h, "p = " + '{:.2e}'.format(AD[1]), ha='center', va='bottom', color=col)
# ADNI + SS
y, h, col = max(df['BL_ADNI_wo_test'].max(), df['BL_skull'].max()) + 0.01, 0.025, 'k'
plt.plot([x2, x2, x3, x3], [y, y+h, y+h, y], lw=1.5, c=col)
plt.text((x2+x3)*.5, y+h, "p = " + str(round(SS_ADNI[1],3)), ha='center', va='bottom', color=col)
# # ADNI + HAmmers
# y, h, col = max(df['BL_ADNI_wo_test'].max(), df['BL_Hammers'].max()) - 0.08, 0.025, 'k'
# plt.plot([x3, x3, x4, x4], [y, y+h, y+h, y], lw=1.5, c=col)
# plt.text((x3+x4)*.5, y+h, "p = " + '{:.2e}'.format(ADNI_Ham[1]), ha='center', va='bottom', color=col)

# above chance level
chance_BL = np.sum(np.where(df['Baseline'] > 0.5, 1, np.zeros((1, 100))))
chance_ADNI = np.sum(np.where(df['BL_ADNI_wo_test'] > 0.5, 1, np.zeros((1, 100))))
chance_Hammers = np.sum(np.where(df['BL_Hammers'] > 0.5, 1, np.zeros((1, 100))))
chance_skull = np.sum(np.where(df['BL_skull'] > 0.5, 1, np.zeros((1, 100))))

# labels
plt.title('Boxplots with dots for training types')
plt.xlabel('Training types with percentage above 50%')
plt.ylabel('Dice coefficient on ADNI test set')
plt.xticks([0, 1, 2, 3], ('Baseline ('+str(chance_BL) + '%)', 'BL SS ('+str(chance_skull) + '%)', 'ADNI (' + str(chance_ADNI) + '%)', 'DA Hammers (' +
                       str(chance_Hammers) + '%)'))
plt.show()

##################################################################################################################
# making list of n best and worst

# n = 5
#
# list2 = test_list
# list1 = df['Baseline']
#
# list1, list2 = (list(t) for t in zip(*sorted(zip(list1, list2))))
# worst_BL = list2[-n-1:-1]
# best_BL = list2[0:n]
#
# np.save('/home/jara/PycharmProjects/GPUcluster/worst_BL', worst_BL, allow_pickle=True)
# np.save('/home/jara/PycharmProjects/GPUcluster/best_BL', best_BL, allow_pickle=True)
#
# list2 = test_list
# list1 = df['BL_ADNI_wo_test']
#
# list1, list2 = (list(t) for t in zip(*sorted(zip(list1, list2))))
# worst_ADNI = list2[-n-1:-1]
# best_ADNI = list2[0:n]
#
# np.save('/home/jara/PycharmProjects/GPUcluster/worst_ADNI', worst_ADNI, allow_pickle=True)
# np.save('/home/jara/PycharmProjects/GPUcluster/best_ADNI', best_ADNI, allow_pickle=True)
#
# list2 = test_list
# list1 = df['BL_Hammers']
#
# list1, list2 = (list(t) for t in zip(*sorted(zip(list1, list2))))
# worst_Ham = list2[-n-1:-1]
# best_Ham = list2[0:n]
#
# np.save('/home/jara/PycharmProjects/GPUcluster/worst_Ham', worst_Ham, allow_pickle=True)
# np.save('/home/jara/PycharmProjects/GPUcluster/best_Ham', best_Ham, allow_pickle=True)
#
# list2 = test_list
# list1 = df['BL_skull']
#
# list1, list2 = (list(t) for t in zip(*sorted(zip(list1, list2))))
# worst_skull = list2[-n-1:-1]
# best_skull = list2[0:n]
#
# np.save('/home/jara/PycharmProjects/GPUcluster/worst_skull', worst_skull, allow_pickle=True)
# np.save('/home/jara/PycharmProjects/GPUcluster/best_skull', best_skull, allow_pickle=True)
#
#############################################################################################
# ADNI number figures
import numpy as np
import matplotlib.pyplot as plt

## Need to add of ADNI_300
x1 = [0, 100, 200, 300, 400, 600, 1611, 1711]
y1 = [0.834, 0.611, 0.84, 0.843, 0.846, 0.845, 0.842, 0.844]
y2 = [0.829, 0.618, 0.82, 0.828, 0.836, 0.829, 0.811, 0.828]
y3 = [0.769, 0.642, 0.839, 0.872, 0.859, 0.875, 0.856]
std2 = [0, 0.017, 0.009, 0.010, 0.008, 0.014, 0.014, 0.013]
std3 = [0.084, 0.029, 0.168, 0.118, 0.135, 0.119, 0.16]

plt.plot(x1[0], y1[0], 'C1', marker='.', markersize=10)
plt.plot(x1[1:], y1[1:], 'C1', marker='.', linestyle='dashed', linewidth=2, markersize=10)
plt.legend(['Validation Hammers'], loc=4)
plt.xlim((x1[0], x1[-1]+10))
plt.ylim((0.6, 1))
plt.xticks(x1)
plt.xlabel('Added number of ADNI samples to training')
plt.ylabel('Dice coefficient')
plt.title('Dice score with SD on validation Hammers (N=4)')
plt.show()

plt.plot(x1[0], y2[0], 'C2', marker='.', markersize=10)
plt.errorbar(x1[1:], y2[1:], yerr=std2[1:], color='C2', marker='.', linestyle='dashed', linewidth=2, markersize=10)
plt.legend([ 'Test Hammers'], loc=4)
plt.xlim((x1[0], x1[-1]+10))
plt.xticks(x1)
plt.ylim((0.6, 1))
plt.xlabel('Added number of ADNI samples to training')
plt.ylabel('Dice coefficient')
plt.title('Dice score with SD on test Hammers (N=4)')
plt.show()

plt.errorbar(x1[0], y3[0], yerr=std3[0], color='C3', marker='.', markersize=10)
plt.errorbar(x1[1:-1], y3[1:], yerr=std3[1:], color='C3', marker='.', linestyle='dashed', linewidth=2, markersize=10)
plt.legend([ 'Test ADNI'], loc=4)
plt.xlim((x1[0]-1, x1[-1]+10))
plt.ylim((0.6, 1))
plt.xticks(x1)
plt.xlabel('Added number of ADNI samples to training')
plt.ylabel('Dice coefficient')
plt.title('Dice score with SD on test set ADNI (N=100)')
plt.show()

y4 = [81, 540, 900, 1300, 1700, 2500, 7300, 7000]
plt.plot(x1[0], y4[0], 'C1', marker='.', markersize=10)
plt.plot(x1[1:], y4[1:], 'C1', marker='.', linestyle='dashed', linewidth=2, markersize=10)
# plt.legend(['Training time per epoch'], loc=4)
plt.xlim((x1[0], x1[-1]+10))
# plt.ylim((0.6, 1))
plt.xticks(x1)
plt.xlabel('Added number of ADNI samples to training')
plt.ylabel('Training time (s)')
plt.title('Training time per epoch')
plt.show()


######################################################################################################################
# Comparison ADNI_200 and ADNIskull_200

# Boxplots with dots
training_types = ['ADNI_200', 'ADNIskull_200', 'BL_ADNI_wo_test']
location = 'local'
data_collected = {}
if location == 'cluster':
    lobe = 'Temporal'
    lateralisation = 'Left'
    it_val = 0.001
    hyper_parameter_to_evaluate = 'Initial_LR'
    save_path = '/media/data/smartens/data/datasize176_208_160/hyperparametertuning'
    test_list = np.load('/media/data/smartens/data/datasize176_208_160/ADNI/adni_test_set.npy')
if location == 'local':
    path = '/home/jara/PycharmProjects/GPUcluster/DiceCoefficientResultsforADNI_'
    test_list = np.load('/AdditionalFiles/adni_test_set.npy')

df = pd.DataFrame({'ID': test_list})

dc = []
labels = []

for training_type in training_types:
    if location == 'cluster':
        parameter_save = os.path.join(save_path, training_type, lobe, lateralisation +
                              hyper_parameter_to_evaluate + 'E' + str(n_epoch))
        pred_seg_path = os.path.join(parameter_save, 'PredictedSegmentation' + str(training_type) + str(it_val))
        data = np.load(os.path.join(pred_seg_path, 'DiceCoefficientResultsforADNI.npy'))
    if location == 'local':
        data = np.load(path + training_type + '.npy')
        label = len(test_list) * [training_type]
    df[training_type] = data
    it = np.arange(len(data))
    for (t, d, l) in zip(it, data, label):
        # print(t, d, l)
        dc.append(d)
        labels.append(l)

df2 = pd.DataFrame({'training_type': labels, 'Dice': dc})

# paired t-test BL with ADNI
SS_BL = stats.ttest_rel(df['ADNI_200'], df['ADNIskull_200'])
SS_ADNI_full = stats.ttest_rel(df['BL_ADNI_wo_test'], df['ADNIskull_200'])

# boxplot with dots
ax = sns.boxplot(x="training_type", y="Dice", data=df2, showfliers = False)
ax = sns.swarmplot(x="training_type", y="Dice", data=df2, color=".25")

# statistical annotation to figure
x1, x2, x3 = 0, 1, 2
# BL + SS
y, h, col = max(df['ADNIskull_200'].max(), df['ADNI_200'].max()) + 0.02, 0.05, 'k'
plt.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, c=col)
plt.text((x1+x2)*.5, y+h, "p = " + str(round(SS_BL[1], 4)), ha='center', va='bottom', color=col)
# ADNI_full + ADNI_200_SS
y, h, col = max(df['ADNIskull_200'].max(), df['BL_ADNI_wo_test'].max()) + 0.02, 0.05, 'k'
plt.plot([x2, x2, x3, x3], [y, y+h, y+h, y], lw=1.5, c=col)
plt.text((x2+x3)*.5, y+h, "p = " + str(round(SS_ADNI_full[1], 4)), ha='center', va='bottom', color=col)

# above chance level
chance_ADNI = np.sum(np.where(df['ADNI_200'] > 0.5, 1, np.zeros((1, 100))))
chance_skull = np.sum(np.where(df['ADNIskull_200'] > 0.5, 1, np.zeros((1, 100))))
chance_ADNI_full = np.sum(np.where(df['BL_ADNI_wo_test'] > 0.5, 1, np.zeros((1, 100))))

# labels
plt.title('Boxplots with dots for training types')
plt.xlabel('Training types with percentage above 50%')
plt.ylabel('Dice coefficient on ADNI test set')
plt.xticks([0, 1, 2, 3], ('ADNI 200 ('+str(chance_ADNI) + '%)\nDice:' + str(round(np.mean(df['ADNI_200']), 3)),
                          'Skull stripped ('+str(chance_skull) + '%)\nDice:' + str(round(np.mean(df['ADNIskull_200']), 3)),
                          'ADNI 1611 ('+str(chance_ADNI_full) + '%)\nDice' + str(round(np.mean(df['BL_ADNI_wo_test']), 3))))
plt.show()

n = 5

for training_type in training_types:
    list2 = test_list
    list1 = df[training_type]

    list1, list2 = (list(t) for t in zip(*sorted(zip(list1, list2))))
    worst = list2[-n-1:-1]
    best = list2[0:n]

    np.save('/home/jara/PycharmProjects/GPUcluster/worst_' + training_type, worst, allow_pickle=True)
    np.save('/home/jara/PycharmProjects/GPUcluster/best_' + training_type, best, allow_pickle=True)


""""
Repeated measures ANOVA 300 skull stripped Temporal lobe
"""""
import numpy as np
import pandas as pd
import scipy.stats as stats
import pingouin as pg
import statsmodels.api as sm
from statsmodels.formula.api import ols


def ANOVA_300_SS_Temporal(test_type):
    '''
    :param test_type: 'ADNI' or 'Hammers'
    :return: results of Repeated Measures ANOVA
    '''

    id = ['a', 'b', 'c', 'd', 'e']
    # val_Ham = [0.855, 0.859, 0.849, 0.849, 0.857]
    # num_epoch = [79, 82, 57, 83, 90]
    # test_Ham = [0.824, 0.832, 0.824, 0.828, 0.836]
    # sd_Ham = [0.01, 0.015, 0.014, 0.024, 0.012]
    # test_ADNI = [0.911, 0.914, 0.908, 0.911, 0.916]
    # sd_ADNI = [0.014, 0.013, 0.015, 0.016, 0.012]
    # num = [1, 2, 3, 4, 5]
    test_type = 'Hammers'
    results_all = []
    label = []
    idx = []
    for id_i in id:
        results = np.load('/home/jara/Savine/datasize176_208_160/hyperparametertuning/ADNIskull_300' + id_i +
                          '/Temporal/LeftInitial_LRE200/PredictedSegmentationADNIskull_300' + id_i +
                          '0.001/DiceCoefficientResultsfor' + test_type + '.npy')
        print(results)
        for i, item in enumerate(results):
            results_all.append(item)
            label.append(id_i)
            idx.append(i+1)

    # idx = np.arange(len(label))
    df = pd.DataFrame({'id': idx, 'group': label, 'Dice': results_all})

    rp.summary_cont(df['Dice'])

    rp.summary_cont(df['Dice'].groupby(df['group']))

    res = stats.f_oneway(df['Dice'][df['group'] == 'a'],
                         df['Dice'][df['group'] == 'b'],
                         df['Dice'][df['group'] == 'c'],
                         df['Dice'][df['group'] == 'd'],
                         df['Dice'][df['group'] == 'e'])
    print(res)

    return res