ENV = cpu
PYTHON_BIN = python3
RANDOM_SEED = 1234
MODEL_NAME_TEST = inception_v3
TEST_EXPERIMENT_NAME = mnist_test_experiment

######################### TEST EXPERIMENT ####################################
data/raw/mnist:
	$(PYTHON_BIN) -m src.scripts.download_mnist --save_dir $@

data/partitions_json/mnist/dataset_dict.json: data/raw/mnist
ifeq ($(ENV), cpu)
	$(PYTHON_BIN) -m src.scripts.mnist_dataset_to_json --images_dir $< \
		--dataset_path $@ --random_seed $(RANDOM_SEED) \
		--max_files_train 200 --max_files_test 200
else
	$(PYTHON_BIN) -m src.scripts.mnist_dataset_to_json --images_dir $< \
		--dataset_path $@ --random_seed $(RANDOM_SEED) 
endif

test_experiment: data/partitions_json/mnist/dataset_dict.json
ifeq ($(ENV), cpu)
	$(PYTHON_BIN) -m src.experiments.hub_module_experiment \
		--experiment_name $(TEST_EXPERIMENT_NAME) \
		--images_dir data/raw/mnist \
		--random_seed $(RANDOM_SEED) \
		--dataset_json $< \
		--num_epochs 20 \
		--model_name $(MODEL_NAME_TEST) \
		--remove_prev_ckpts_and_logs \
		--evaluate_every_n_seconds 30 \
		--tensors_to_log_train loss global_step \
		--save_checkpoints_steps 10
else
	$(PYTHON_BIN) -m src.experiments.hub_module_experiment \
		--experiment_name $(TEST_EXPERIMENT_NAME) \
		--images_dir data/raw/mnist \
		--random_seed $(RANDOM_SEED) \
		--dataset_json $< \
		--num_epochs 1000 \
		--model_name $(MODEL_NAME_TEST) \
		--remove_prev_ckpts_and_logs \
		--evaluate_every_n_seconds 600 \
		--tensors_to_log_train loss global_step \
		--save_checkpoints_steps 100
endif

clear_test:
	rm -r data/raw/mnist || true
	rm -r data/partitions_json/mnist || true
	rm -r logs_and_checkpoints/$(TEST_EXPERIMENT_NAME) || true
	rm -r results/$(TEST_EXPERIMENT_NAME)
	rm -r saved_models/$(TEST_EXPERIMENT_NAME) || true

clear_test_with_bottlenecks: clear_test
	rm -r data/bottlenecks/$(MODEL_NAME_TEST)/mnist || true

####################### GENERATE DATASET ####################################
MAGNIFICATION = "x40"
DASASET_SLIDES = ihc_slides
DATASET_ROIS = ihc_rois_$(MAGNIFICATION)
DATASET_PATCHES = ihc_patches_$(MAGNIFICATION)

data/extras/$(DASASET_SLIDES)/annotations:
# Pedirá que ingrese Usuario y contraseña para ndp.microscopiavirtual.com
	$(PYTHON_BIN) -m src.scripts.get_annotations \
		--folder_id 10590 \
		--annotations_dir $@

data/extras/$(DASASET_SLIDES)/serialized_annotations: data/extras/$(DASASET_SLIDES)/annotations
	$(PYTHON_BIN) -m src.scripts.get_csv_and_serialize_annotations \
		--annotations_dir $< \
		--save_dir $@ \
		--csv_path $@/summary.csv

data/interim/$(DATASET_ROIS): data/extras/$(DASASET_SLIDES)/serialized_annotations
	$(PYTHON_BIN) -m src.scripts.extract_rois_from_slides \
		--serialized_annotations_dir $< \
		--slides_dir data/raw/$(DASASET_SLIDES) \
		--rois_dir $@ \
		--magnification $(MAGNIFICATION)

data/processed/$(DATASET_PATCHES): data/interim/$(DATASET_ROIS)
	$(PYTHON_BIN) -m src.scripts.extract_patches_from_rois \
		--excel_file data/extras/$(DASASET_SLIDES)/HER2.xlsx \
		--rois_dir $< \
		--patches_dir $@ \
		--valid_owners UI.Patologo2 \
		--patches_height 300 \
		--patches_width 300 \
		--stride_rows 50 \
		--stride_columns 50
	$(PYTHON_BIN) -m src.scripts.remove_useless_patches \
		--patches_dir $@ \
		--kib_min_size 10.7




clear_rois:
	rm -r data/extras/$(DASASET_SLIDES)/annotations  || true
	rm -r data/extras/$(DASASET_SLIDES)/serialized_annotations  || true
	rm -r data/interim/$(DATASET_ROIS) || true
	rm -r data/processed/$(DATASET_PATCHES) || true


extract_patches: data/processed/$(DATASET_PATCHES)
	echo "hola"

	
