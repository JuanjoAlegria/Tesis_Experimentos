test_cpu: create_mnist_dataset_cpu train_mnist_cpu

create_mnist_dataset_cpu:
	python3 -m src.scripts.download_mnist --save_dir data/raw/mnist
	python3 -m src.scripts.mnist_dataset_to_json --images_dir data/raw/mnist \
		--random_seed 1234 --max_files_train 200 --max_files_test 200

train_mnist_cpu:
	python3 -m src.experiments.hub_module_experiment \
		--experiment_name mnist_test_cpu \
		--images_dir data/raw/mnist \
		--random_seed 1234 \
		--dataset_json data/partitions_json/mnist/dataset_dict.json \
		--num_epochs 20 \
		--model_name inception_v3 \
		--remove_prev_ckpts_and_logs \
		--evaluate_every_n_seconds 30 \
		--tensors_to_log_train loss global_step \
		--save_checkpoints_steps 10


test_gpu: create_mnist_dataset_gpu train_mnist_gpu

create_mnist_dataset_gpu:
	python3 -m src.scripts.download_mnist --save_dir data/raw/mnist
	python3 -m src.scripts.mnist_dataset_to_json --images_dir data/raw/mnist \
		--random_seed 1234

train_mnist_gpu:
	python3 -m src.experiments.hub_module_experiment \
		--experiment_name mnist_test_cpu \
		--images_dir data/raw/mnist \
		--random_seed 1234 \
		--dataset_json data/partitions_json/mnist/dataset_dict.json \
		--num_epochs 1000 \
		--model_name inception_v3 \
		--remove_prev_ckpts_and_logs \
		--evaluate_every_n_seconds 600 \
		--tensors_to_log_train loss global_step \
		--save_checkpoints_steps 100

clear_test:
	rm -r data/raw/mnist || true
	rm -r data/partitions_json/mnist || true
	rm -r logs_and_checkpoints/mnist_test_cpu || true
	rm -r saved_models/mnist_test_cpu || true

clear_test_with_bottlenecks: clear_test
	rm -r data/bottlenecks/inception_v3/mnist || true