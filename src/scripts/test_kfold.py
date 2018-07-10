import numpy as np
from ..utils import excel
from ..dataset import data_utils

excel_file = "/home/juanjo/U/2017/Tesis/Experimentos/data/extras/ihc_slides/HER2.xlsx"

ids, labels = excel.get_valid_slides_ids(excel_file)

ids = np.array(ids)
labels = np.array(labels)
negative = ids[np.where((labels == '0') | (labels == '1'))]
equivocal = ids[np.where(labels == '2')]
positive = ids[np.where(labels == '3')]

train_dir = "/home/juanjo/U/2017/Tesis/Experimentos/data/processed/ihc_patches_x40/"
test_dir = "/home/juanjo/U/2017/Tesis/Experimentos/data/processed/ihc_all_patches_x40/"

data_utils.generate_kfold(negative, equivocal, positive,
                          5, train_dir, test_dir)
