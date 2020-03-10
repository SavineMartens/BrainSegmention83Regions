import numpy as np
import nibabel as nib
import os
from Shortlist import *
import pandas as pd
from sklearn.utils.class_weight import compute_class_weight

test_path = '/home/jara/Savine/datasize176_208_160/Hammers_skull'
lobe_list = ['TemporalLeft', 'TemporalRight', 'CentralLeft', 'CentralRight',
             'FrontalLeft', 'FrontalRight', 'Appendix', 'OccipitalParietal']

store_volumes = np.zeros((30, 84), dtype=int)
store_order = np.zeros((30, 84), dtype=int)

for idx in np.arange(30):
    ID = str(idx + 1)
    ID = ID.zfill(2)
    Y_i = nib.load("".join(test_path + '/a' + ID + '-seg.nii.gz'))
    Y_true_all = np.asarray(Y_i.get_data())

    n_elements, counts = np.unique(Y_true_all, return_counts=True)
    store_volumes[idx, :] = counts.astype(int)
    sorted_counts, sorted_elements = (list(element) for element in zip(*sorted(zip(counts, n_elements))))
    store_order[idx] = sorted_elements
    # print(sorted_elements)

    # areas = lobe_selection('FrontalLeft')
    # Y_i = remove_areas(Y_true_all, areas)
    # Y_i = encode1hot(Y_i)
    #
    # y_integers = np.argmax(Y_i, axis=-1)
    # class_weights = compute_class_weight('balanced', np.unique(y_integers), y_integers)
    # d_class_weights = dict(enumerate(class_weights))

    print('Next volume')

# relative volumes
total_volume = 176*208*160
relative_total = store_volumes/total_volume

# volume_per_lobe
for lobe in ['Appendix']:
    areas = lobe_selection(lobe)
    classes_incl_BG = [0, ] + areas
    print(lobe, areas)
    selection_vol_lobe = store_volumes[:, areas]
    # selection_vol_lobe_incl_BG = store_volumes[:,classes_incl_BG]
    total_per_person_per_lobe = np.sum(selection_vol_lobe, axis=1)
    rel_selection_lobe = np.zeros((30, len(areas)))
    for a, area in enumerate(areas):
        for n in np.arange(30):
            rel_selection_lobe[n, a] = selection_vol_lobe[n,a]/total_per_person_per_lobe[n]
        string_label = areas_to_str(area)
    df = pd.DataFrame(data=rel_selection_lobe, columns=areas)
    min_area = np.median(rel_selection_lobe.argmin(axis=1))
    print('Median lowest area is', areas[int(min_area)], areas_to_str(area))

    print('Average relative volume to lobe:', round(np.mean(rel_selection_lobe[:, int(min_area)])*100, 3), '% of volume')

    print('Averaged relative volume over all in lobe (%):', np.mean(rel_selection_lobe,
                                                            axis=0)*100)
    # rel_selection_vol = np.mean(selection_vol_lobe_incl_BG /total_volume, axis=0)
    rel_selection_vol = np.mean(relative_total[:,classes_incl_BG], axis=0)
    print('Averaged relative volume for whole volume', rel_selection_vol)
    rel_avg_weights = rel_selection_vol.max()/rel_selection_vol
    print('weights:', rel_avg_weights)
    print('Next lobe')




