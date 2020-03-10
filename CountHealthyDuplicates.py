import numpy as np
import csv
import pandas as pd
import random
f = open('AdditionalFiles/3months_list.txt', "r")

# list with all IDs that were measured baseline and after 3 months
duplicate_list = []

f1 = f.readlines()
for x in f1:
    for word in x.split():
        ID = word.replace('_m03.nii.gz', '')
        duplicate_list.append(ID)

ADNI_list = np.load('AdditionalFiles/adni_list_full_labels.npy')
ADNI_list2 = []
for item in ADNI_list:
    ID = item.replace('_bl', '')
    ADNI_list2.append(ID)
ADNI_list = ADNI_list2

df = pd.read_csv('AdditionalFiles/AllSubjectsDiagnosis.csv')
df.set_index('ID', inplace=True)

# find healthy subjects == 'CN'
control_list = []
for item in ADNI_list:
    # print(item)
    try:
        diagnosis = df.loc[item, 'Dx']
        if diagnosis == 'CN':
            # print(item, diagnosis)
            control_list.append(item)
    except KeyError:
        print('item is probably not in diagnosis list, ID =', item)

# select only ones that are control and 3 month measurements as well
control_duplicates_list = []
for item in control_list:   # healthy list
    if item in duplicate_list:  # list of people measured at m0 and m3
        control_duplicates_list.append(item)

num_3mo_control = len(control_duplicates_list)
rand_id = random.sample(range(0, num_3mo_control-1), 100) # np.random.randint(0, num_3mo_control-1, 100)

test_set = np.asarray(control_duplicates_list)[rand_id]

# remove test set from adni list
for item in ADNI_list:
    if item in test_set:
        ADNI_list.remove(item)

train_set = ADNI_list

np.save('/AdditionalFiles/adni_test_set.npy', test_set, allow_pickle=True)
np.save('/AdditionalFiles/adni_train_set.npy', train_set, allow_pickle=True)



