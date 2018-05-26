ENV = cpu
PYTHON_BIN = python3
RANDOM_SEED = 1234
MODEL_NAME = inception_v3
TEST_EXPERIMENT_NAME = mnist_test_experiment

######################### TEST EXPERIMENT ####################################
data/raw/mnist:
	$(PYTHON_BIN) -m src.scripts.mnist.download_mnist --save_dir $@

data/partitions_json/mnist/dataset_dict.json: data/raw/mnist
ifeq ($(ENV), cpu)
	$(PYTHON_BIN) -m src.scripts.mnist.generate_dataset --images_dir $< \
		--dataset_path $@ --random_seed $(RANDOM_SEED) \
		--max_files_train 200 --max_files_test 200
else
	$(PYTHON_BIN) -m src.scripts.generate_dataset --images_dir $< \
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
		--model_name $(MODEL_NAME) \
		--remove_prev_ckpts_and_logs \
		--tensors_to_log_train global_step loss filenames \
		--save_checkpoints_steps 5 \
		--eval_frequency 5
else
	$(PYTHON_BIN) -m src.experiments.hub_module_experiment \
		--experiment_name $(TEST_EXPERIMENT_NAME) \
		--images_dir data/raw/mnist \
		--random_seed $(RANDOM_SEED) \
		--dataset_json $< \
		--num_epochs 1000 \
		--model_name $(MODEL_NAME) \
		--remove_prev_ckpts_and_logs \
		--tensors_to_log_train global_step loss filenames \
		--save_checkpoints_steps 50 \
		--eval_frequency 50
endif

clear_test:
	rm -r data/raw/mnist || true
	rm -r data/partitions_json/mnist || true
	rm -r logs_and_checkpoints/$(TEST_EXPERIMENT_NAME) || true
	rm -r results/$(TEST_EXPERIMENT_NAME) || true
	rm -r saved_models/$(TEST_EXPERIMENT_NAME) || true

clear_test_with_bottlenecks: clear_test
	rm -r data/bottlenecks/$(MODEL_NAME)/mnist || true

####################### PATCHES EXPERIMENT ####################################
MAGNIFICATION = x40
DASASET_SLIDES = ihc_slides
DATASET_ROIS = ihc_rois_$(MAGNIFICATION)
DATASET_PATCHES = ihc_patches_$(MAGNIFICATION)
PATCHES_EXPEIMENT_NAME = ihc_patches_experiment_2
data/extras/$(DASASET_SLIDES)/annotations:
# Pedirá que ingrese Usuario y contraseña para ndp.microscopiavirtual.com
	$(PYTHON_BIN) -m src.scripts.ihc.get_annotations \
		--folder_id 10590 \
		--annotations_dir $@

data/extras/$(DASASET_SLIDES)/serialized_annotations: \
data/extras/$(DASASET_SLIDES)/annotations
	$(PYTHON_BIN) -m src.scripts.ihc.get_csv_and_serialize_annotations \
		--annotations_dir $< \
		--save_dir $@ \
		--csv_path $@/summary.csv

data/interim/$(DATASET_ROIS): \
data/extras/$(DASASET_SLIDES)/serialized_annotations
	$(PYTHON_BIN) -m src.scripts.ihc.extract_rois_from_slides \
		--serialized_annotations_dir $< \
		--slides_dir data/raw/$(DASASET_SLIDES) \
		--rois_dir $@ \
		--magnification $(MAGNIFICATION)

data/processed/$(DATASET_PATCHES): data/interim/$(DATASET_ROIS)
	$(PYTHON_BIN) -m src.scripts.ihc.extract_patches_from_rois \
		--excel_file data/extras/$(DASASET_SLIDES)/HER2.xlsx \
		--rois_dir $< \
		--patches_dir $@ \
		--valid_owners UI.Patologo2 \
		--patches_height 300 \
		--patches_width 300 \
		--stride_rows 250 \
		--stride_columns 250
	$(PYTHON_BIN) -m src.scripts.ihc.remove_useless_patches \
		--patches_dir $@ \
		--kib_min_size 10.7


data/partitions_json/$(DATASET_PATCHES)/dataset_dict.json: \
data/processed/$(DATASET_PATCHES)
	$(PYTHON_BIN) -m src.scripts.ihc.generate_dataset \
		--images_dir $< \
		--dataset_path $@ \
		--validation_percentage 10 \
		--test_percentage 10 \
		--random_seed $(RANDOM_SEED)

clear_patches_experiment:
	rm -r data/extras/$(DASASET_SLIDES)/annotations  || true
	rm -r data/extras/$(DASASET_SLIDES)/serialized_annotations  || true
	rm -r data/interim/$(DATASET_ROIS) || true
	rm -r data/processed/$(DATASET_PATCHES) || true
	rm -r data/partitions_json/$(DATASET_PATCHES) || true

patches_experiment: data/partitions_json/$(DATASET_PATCHES)/dataset_dict.json
	$(PYTHON_BIN) -m src.experiments.hub_module_experiment \
		--experiment_name $(PATCHES_EXPEIMENT_NAME) \
		--images_dir data/processed/$(DATASET_PATCHES) \
		--random_seed $(RANDOM_SEED) \
		--dataset_json $< \
		--num_epochs 4000 \
		--model_name $(MODEL_NAME) \
		--remove_prev_ckpts_and_logs \
		--tensors_to_log_train loss global_step \
		--save_checkpoints_steps 100 \
		--eval_frequency 100
