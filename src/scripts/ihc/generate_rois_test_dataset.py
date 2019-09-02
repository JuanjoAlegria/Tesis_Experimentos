import os
import json
import argparse
import numpy as np


def main(ids_partition_json, dataset_dict_folder):
    with open(ids_partition_json) as file:
        partitions = json.load(file)
    negative_part = partitions["negative_part"]
    equivocal_part = partitions["equivocal_part"]
    positive_part = partitions["positive_part"]

    for idx in range(len(negative_part)):
        current_biopsies = negative_part[idx] + \
            equivocal_part[idx] + \
            positive_part[idx]
        # Abriremos el diccionario siguiente, ya que allí
        # las imágenes serán de entrenamiento
        idx_dictionary = ((idx + 1) % 5) + 1
        dataset_name = 'dataset_dict_fold_{}.json'.format(idx_dictionary)
        with open(os.path.join(dataset_dict_folder, dataset_name)) as file:
            dataset_json = json.load(file)
        train_features = dataset_json["train_features"]
        train_labels = dataset_json["train_labels"]
        validation_features = dataset_json["validation_features"]
        validation_labels = dataset_json["validation_labels"]

        features = train_features + validation_features
        labels = train_labels + validation_labels
        assert len(features) == len(labels)

        indexes = []
        for jdx in range(len(features)):
            feature = features[jdx]
            biopsy_id = feature.split("/")[1].split("_")[0]
            if biopsy_id in current_biopsies:
                indexes.append(jdx)

        features = np.array(features)
        labels = np.array(labels)

        current_features = features[indexes]
        current_labels = labels[indexes]

        sorted_indexes = np.argsort(current_features)
        current_features = current_features[sorted_indexes]
        current_labels = current_labels[sorted_indexes]

        current_features = current_features.tolist()
        current_labels = current_labels.tolist()

        output_dataset = {"test_rois_features": current_features,
                          "test_rois_labels": current_labels}
        output_filename = "test_rois_dataset_fold_{}.json".format(idx)

        output_path = os.path.join(dataset_dict_folder, output_filename)
        assert not os.path.exists(output_path)
        with open(output_path, "w") as output_file:
            json.dump(output_dataset, output_file)

if __name__ == "__main__":
    PARSER = argparse.ArgumentParser()
    PARSER.add_argument(
        '--ids_partition_json',
        type=str,
        required=True,
        help="""\
        Ruta al archivo json con la partición de ids generada previamente.
        """
    )
    PARSER.add_argument(
        '--dataset_dict_folder',
        type=str,
        required=True,
        help="""
        Directorio donde se encuentran las particiones de kfold en formato json.
        """
    )
    FLAGS = PARSER.parse_args()
    main(FLAGS.ids_partition_json, FLAGS.dataset_dict_folder)
