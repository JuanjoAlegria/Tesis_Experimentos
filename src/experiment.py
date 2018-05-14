import os
import sys
import numpy as np
from tensorflow import app, train
from train import dataset, hub_module_estimator


def main(_):
    np.random.seed(FLAGS.random_seed)
    # Obtenemos los conjuntos de entrenamiento y prueba
    train_filenames, train_labels, labels_map = dataset.get_filenames_and_labels(
        FLAGS.images_dir, "train", labels_to_indexes=FLAGS.convert_labels_to_indexes,
        max_files=FLAGS.max_files_train)
    test_filenames, test_labels, _ = dataset.get_filenames_and_labels(
        FLAGS.images_dir, "test", labels_map=labels_map,
        max_files=FLAGS.max_files_test)
    n_classes = len(labels_map)

    hub_experiment = hub_module_estimator.HubModelExperiment(FLAGS, n_classes)
    if FLAGS.output_labels == "":
        output_labels_file = os.path.join(hub_experiment.results_dir,
                                          "output_labels.txt")
    else:
        output_labels_file = FLAGS.output_labels

    dataset.write_labels_map(labels_map, output_labels_file)

    # Configuramos el hook para loggear tensores
    tensors_to_log = {"filenames": "IteratorGetNext:0",
                      "loss": "loss:0",
                      "has_prev_bottlenecks": "has_prev_bottlenecks:0"}
    #"write_bottlenecks": "write_bottlenecks:0"

    logging_hook = train.LoggingTensorHook(
        tensors=tensors_to_log, every_n_iter=1)

    hub_experiment.train(train_filenames, train_labels, [logging_hook])
    hub_experiment.test(test_filenames, test_labels)
    hub_experiment.export_graph()

if __name__ == '__main__':
    PARSER = hub_module_estimator.get_parser()
    PARSER.add_argument(
        '--convert_labels_to_indexes',
        type=bool,
        default=False,
        help="""\
        Indica si es que se debe o no generar un mapeo de etiquetas a índices.
        En caso de ser False, se asume que las etiquetas ya son enteros en el
        rango [0, ... , n-1]; en caso contrario, se producirán errores.
        """,
    )
    PARSER.add_argument(
        '--output_labels',
        type=str,
        default="",
        help="""\
        Ubicación donde guardar el mapeo de etiquetas producido por el
        entrenamiento.'\
        """
    )
    PARSER.add_argument(
        '--max_files_train',
        type=int,
        default=-1,
        help="""\
        Cantidad máxima de archivos a usar para entrenar. Útil para depuración.
        """,
    )
    PARSER.add_argument(
        '--max_files_test',
        type=int,
        default=-1,
        help="""\
        Cantidad máxima de archivos a usar para evaluar. Útil para depuración.
        """,
    )

    FLAGS, UNPARSED = PARSER.parse_known_args()
    app.run(main=main, argv=[sys.argv[0]] + UNPARSED)
