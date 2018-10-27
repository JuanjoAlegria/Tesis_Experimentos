import os
import glob
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix
from ...utils import outputs

OUTPUT_FILE_PATTERN = "{}_predictions.txt"


def get_single_experiment_results(experiment_folder):
    accuracies = {}
    matrices = {}
    for dataset in ["train", "validation"]:
        result_path = os.path.join(experiment_folder,
                                   OUTPUT_FILE_PATTERN.format(dataset))
        with open(result_path) as file:
            output_lines = file.readlines()
        _, real, predicted = outputs.transform_to_lists(output_lines)
        accuracies[dataset] = accuracy_score(real, predicted)
        matrices[dataset] = confusion_matrix(real, predicted)
    return accuracies, matrices


def main(results_folder, experiment_prefix):
    pattern = os.path.join(results_folder, experiment_prefix + "_[0-9]")
    experiments_folders = glob.glob(pattern)
    experiments_folders.sort()
    # validation
    accuracies = []
    matrices = []
    for folder in experiments_folders:
        val_result_path = os.path.join(folder, "validation_predictions.txt")
        with open(val_result_path) as file:
            output_lines = file.readlines()
        _, real, predicted = outputs.transform_to_lists(output_lines)
        accuracy = accuracy_score(real, predicted)
        print(accuracy)
        accuracies.append(accuracy)
    accuracies = np.array(accuracies)
    mean_accuracy = accuracies.mean()
    std_accuracy = accuracies.std()
    print(mean_accuracy, "±", std_accuracy)

results_folder = "/home/juanjo/U/2017/Tesis/Experimentos/results/"
experiment_prefix = "ihc_patches_kfold_fixed_ids_x40_experiment"

main(results_folder, experiment_prefix)
