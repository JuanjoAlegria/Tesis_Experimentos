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
	$(PYTHON_BIN) -m src.scripts.mnist.generate_dataset --images_dir $< \
		--dataset_path $@ --random_seed $(RANDOM_SEED) 
endif

test_experiment: data/partitions_json/mnist/dataset_dict.json
ifeq ($(ENV), cpu)
	$(PYTHON_BIN) -m src.scripts.ihc.train_model \
		--experiment_name $(TEST_EXPERIMENT_NAME) \
		--train_images_dir data/raw/mnist \
		--validation_images_dir data/raw/mnist \
		--test_images_dir data/raw/mnist \
		--random_seed $(RANDOM_SEED) \
		--dataset_json $< \
		--num_epochs 20 \
		--model_name $(MODEL_NAME) \
		--remove_prev_ckpts_and_logs \
		--tensors_to_log_train global_step loss \
		--save_checkpoints_steps 5 \
		--eval_frequency 5 
else
	$(PYTHON_BIN) -m src.scripts.ihc.train_model \
		--experiment_name $(TEST_EXPERIMENT_NAME) \
		--train_images_dir data/raw/mnist \
		--validation_images_dir data/raw/mnist \
		--test_images_dir data/raw/mnist \
		--random_seed $(RANDOM_SEED) \
		--dataset_json $< \
		--num_epochs 100 \
		--model_name $(MODEL_NAME) \
		--remove_prev_ckpts_and_logs \
		--tensors_to_log_train global_step loss \
		--save_checkpoints_steps 50 \
		--eval_frequency 50 \
		# --fine_tuning
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
NUM_EPOCHS = 5000
FINE_TUNING = True

SLIDES_DIR = ihc_slides
ROIS_DIR = ihc_rois_$(MAGNIFICATION)
PATCHES_FROM_ROIS_DIR = ihc_patches_from_rois_$(MAGNIFICATION)

PATCHES_HEIGHT = 300
PATCHES_WIDTH = 300


data/extras/$(SLIDES_DIR)/annotations:
# Pedirá que ingrese Usuario y contraseña para ndp.microscopiavirtual.com
	$(PYTHON_BIN) -m src.scripts.ihc.get_annotations \
		--folder_id 10590 \
		--annotations_dir $@

data/extras/$(SLIDES_DIR)/serialized_annotations: \
data/extras/$(SLIDES_DIR)/annotations
	$(PYTHON_BIN) -m src.scripts.ihc.get_csv_and_serialize_annotations \
		--annotations_dir $< \
		--save_dir $@ \
		--csv_path $@/summary.csv

data/interim/$(ROIS_DIR): \
data/extras/$(SLIDES_DIR)/serialized_annotations
	$(PYTHON_BIN) -m src.scripts.ihc.extract_rois_from_slides \
		--serialized_annotations_dir $< \
		--slides_dir data/raw/$(SLIDES_DIR) \
		--rois_dir $@ \
		--magnification $(MAGNIFICATION)

data/processed/$(PATCHES_FROM_ROIS_DIR): data/interim/$(ROIS_DIR)
	$(PYTHON_BIN) -m src.scripts.ihc.extract_patches_from_rois \
		--excel_file data/extras/$(SLIDES_DIR)/HER2.xlsx \
		--rois_dir $< \
		--patches_dir $@ \
		--valid_owners UI.Patologo2 \
		--patches_height $(PATCHES_HEIGHT) \
		--patches_width $(PATCHES_WIDTH) \
		--stride_rows 250 \
		--stride_columns 250 \
		--threshold_gray_pixels 0.9


data/partitions_json/$(PATCHES_FROM_ROIS_DIR)/dataset_dict.json: \
data/processed/$(PATCHES_FROM_ROIS_DIR)
	$(PYTHON_BIN) -m src.scripts.ihc.generate_dataset \
		--images_dir $< \
		--dataset_path $@ \
		--validation_percentage 10 \
		--test_percentage 10 \
		--random_seed $(RANDOM_SEED)

clear_patches_experiment:
	rm -r data/extras/$(SLIDES_DIR)/annotations  || true
	rm -r data/extras/$(SLIDES_DIR)/serialized_annotations  || true
	rm -r data/interim/$(ROIS_DIR) || true
	rm -r data/processed/$(PATCHES_FROM_ROIS_DIR) || true
	rm -r data/partitions_json/$(PATCHES_FROM_ROIS_DIR) || true

patches_experiment: data/partitions_json/$(PATCHES_FROM_ROIS_DIR)/dataset_dict.json
	$(PYTHON_BIN) -m src.scripts.ihc.train_model \
		--experiment_name $(PATCHES_FROM_ROIS_DIR)_experiment \
		--images_dir data/processed/$(PATCHES_FROM_ROIS_DIR) \
		--random_seed $(RANDOM_SEED) \
		--dataset_json $< \
		--num_epochs $(NUM_EPOCHS) \
		--model_name $(MODEL_NAME) \
		--tensors_to_log_train loss global_step \
		--save_checkpoints_steps 100 \
		--eval_frequency 100

patches_random_experiment: \
data/partitions_json/$(PATCHES_FROM_ROIS_DIR)/dataset_dict.json
	$(PYTHON_BIN) -m src.scripts.ihc.train_model \
		--experiment_name $(PATCHES_FROM_ROIS_DIR)_random_experiment \
		--images_dir data/processed/$(PATCHES_FROM_ROIS_DIR) \
		--random_seed $(RANDOM_SEED) \
		--dataset_json $< \
		--num_epochs $(NUM_EPOCHS) \
		--model_name $(MODEL_NAME) \
		--tensors_to_log_train loss global_step \
		--save_checkpoints_steps 100 \
		--eval_frequency 100 \
		--flip_left_right \
		--random_crop 20 \
		--random_scale 20 \
		--random_brightness 5

patches_fine_tuning_experiment: \
data/partitions_json/$(PATCHES_FROM_ROIS_DIR)/dataset_dict.json
	$(PYTHON_BIN) -m src.scripts.ihc.train_model \
		--experiment_name $(PATCHES_FROM_ROIS_DIR)_fine_tuning_experiment \
		--images_dir data/processed/$(PATCHES_FROM_ROIS_DIR) \
		--random_seed $(RANDOM_SEED) \
		--dataset_json $< \
		--num_epochs $(NUM_EPOCHS) \
		--model_name $(MODEL_NAME) \
		--tensors_to_log_train loss global_step \
		--save_checkpoints_steps 100 \
		--eval_frequency 100 \
		--fine_tuning

patches_random_fine_tuning_experiment: \
data/partitions_json/$(PATCHES_FROM_ROIS_DIR)/dataset_dict.json
	$(PYTHON_BIN) -m src.scripts.ihc.train_model \
		--experiment_name $(PATCHES_FROM_ROIS_DIR)_random_fine_tuning_experiment \
		--images_dir data/processed/$(PATCHES_FROM_ROIS_DIR) \
		--random_seed $(RANDOM_SEED) \
		--dataset_json $< \
		--num_epochs $(NUM_EPOCHS) \
		--model_name $(MODEL_NAME) \
		--tensors_to_log_train loss global_step \
		--save_checkpoints_steps 100 \
		--eval_frequency 100 \
		--flip_left_right \
		--random_crop 20 \
		--random_scale 20 \
		--random_brightness 5 \
		--fine_tuning


###################### SLIDE CLASSIFICATION EXPERIMENT #######################

ALL_PATCHES_DIR = ihc_all_patches_$(MAGNIFICATION)
N_FOLDS = 5
PROPORTION_THRESHOLD = 0.1
data/processed/$(ALL_PATCHES_DIR): 
	$(PYTHON_BIN) -m src.scripts.ihc.extract_all_patches \
		--excel_file data/extras/$(SLIDES_DIR)/HER2.xlsx \
		--slides_dir data/raw/$(SLIDES_DIR) \
		--patches_dir $@ \
		--patches_height $(PATCHES_HEIGHT) \
		--patches_width $(PATCHES_WIDTH) \
		--magnification $(MAGNIFICATION)



data/extras/ihc_slides/tissue_proportion_$(ALL_PATCHES_DIR).json: \
data/processed/$(ALL_PATCHES_DIR)
	$(PYTHON_BIN) -m src.scripts.ihc.calculate_tissue_proportion \
		--patches_dir $< \
		--json_path $@

data/extras/ihc_slides/tissue_proportion_$(PATCHES_FROM_ROIS_DIR).json: \
data/processed/$(PATCHES_FROM_ROIS_DIR)
	$(PYTHON_BIN) -m src.scripts.ihc.calculate_tissue_proportion \
		--patches_dir $< \
		--json_path $@


data/partitions_json/ihc_patches_$(MAGNIFICATION)/k_fold: \
data/extras/ihc_slides/tissue_proportion_$(PATCHES_FROM_ROIS_DIR).json \
data/extras/ihc_slides/tissue_proportion_$(ALL_PATCHES_DIR).json 
	$(PYTHON_BIN) -m src.scripts.ihc.generate_kfold_dataset \
		--excel_file data/extras/$(SLIDES_DIR)/HER2.xlsx \
		--n_folds $(N_FOLDS) \
		--train_dir data/processed/$(PATCHES_FROM_ROIS_DIR) \
		--test_dir data/processed/$(ALL_PATCHES_DIR) \
		--dataset_dst_dir $@ \
		--train_proportions_json $< \
		--test_proportions_json $(word 2,$^) \
		--proportion_threshold $(PROPORTION_THRESHOLD)

calculate_all_tissue_proportions: \
data/extras/ihc_slides/tissue_proportion_$(PATCHES_FROM_ROIS_DIR).json \
data/extras/ihc_slides/tissue_proportion_$(ALL_PATCHES_DIR).json
	echo 'hola'

generate_kfold_dataset: \
data/partitions_json/ihc_patches_$(MAGNIFICATION)/k_fold
	echo 'listo'
