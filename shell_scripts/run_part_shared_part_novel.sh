# CIFAR100

# vanilla_cnn
# N=10
python src/la/scripts/run_part_shared_part_novel.py "nn.dataset_name=cifar100" "nn.num_shared_classes=80" "nn.num_novel_classes_per_task=10"
python src/la/scripts/run_part_shared_part_novel.py "nn.dataset_name=cifar100" "nn.num_shared_classes=60" "nn.num_novel_classes_per_task=10"
python src/la/scripts/run_part_shared_part_novel.py "nn.dataset_name=cifar100" "nn.num_shared_classes=40" "nn.num_novel_classes_per_task=10"
python src/la/scripts/run_part_shared_part_novel.py "nn.dataset_name=cifar100" "nn.num_shared_classes=20" "nn.num_novel_classes_per_task=10"
# N=5
python src/la/scripts/run_part_shared_part_novel.py "nn.dataset_name=cifar100" "nn.num_shared_classes=80" "nn.num_novel_classes_per_task=5"
python src/la/scripts/run_part_shared_part_novel.py "nn.dataset_name=cifar100" "nn.num_shared_classes=60" "nn.num_novel_classes_per_task=5"
python src/la/scripts/run_part_shared_part_novel.py "nn.dataset_name=cifar100" "nn.num_shared_classes=40" "nn.num_novel_classes_per_task=5"
python src/la/scripts/run_part_shared_part_novel.py "nn.dataset_name=cifar100" "nn.num_shared_classes=20" "nn.num_novel_classes_per_task=5"

# efficient_net
# N=10
python src/la/scripts/run_part_shared_part_novel.py "nn.task_models=['efficient_net']" "nn.dataset_name=cifar100" "nn.num_shared_classes=80" "nn.num_novel_classes_per_task=10"
./shell_scripts/clean_hf_cache.sh ./data/
python src/la/scripts/run_part_shared_part_novel.py "nn.task_models=['efficient_net']" "nn.dataset_name=cifar100" "nn.num_shared_classes=60" "nn.num_novel_classes_per_task=10"
./shell_scripts/clean_hf_cache.sh ./data/
python src/la/scripts/run_part_shared_part_novel.py "nn.task_models=['efficient_net']" "nn.dataset_name=cifar100" "nn.num_shared_classes=40" "nn.num_novel_classes_per_task=10"
./shell_scripts/clean_hf_cache.sh ./data/
python src/la/scripts/run_part_shared_part_novel.py "nn.task_models=['efficient_net']" "nn.dataset_name=cifar100" "nn.num_shared_classes=20" "nn.num_novel_classes_per_task=10"
# N=5
python src/la/scripts/run_part_shared_part_novel.py "nn.task_models=['efficient_net']" "nn.dataset_name=cifar100" "nn.num_shared_classes=80" "nn.num_novel_classes_per_task=5"
./shell_scripts/clean_hf_cache.sh ./data/
python src/la/scripts/run_part_shared_part_novel.py "nn.task_models=['efficient_net']" "nn.dataset_name=cifar100" "nn.num_shared_classes=60" "nn.num_novel_classes_per_task=5"
./shell_scripts/clean_hf_cache.sh ./data/
python src/la/scripts/run_part_shared_part_novel.py "nn.task_models=['efficient_net']" "nn.dataset_name=cifar100" "nn.num_shared_classes=40" "nn.num_novel_classes_per_task=5"
./shell_scripts/clean_hf_cache.sh ./data/
python src/la/scripts/run_part_shared_part_novel.py "nn.task_models=['efficient_net']" "nn.dataset_name=cifar100" "nn.num_shared_classes=20" "nn.num_novel_classes_per_task=5"

# TINY_IMAGENET
# vanilla_cnn
python src/la/scripts/run_part_shared_part_novel.py "nn.task_models=['vanilla_cnn']" "nn.dataset_name=tiny_imagenet" "nn.num_shared_classes=100" "nn.num_novel_classes_per_task=25"
python src/la/scripts/run_part_shared_part_novel.py "nn.task_models=['vanilla_cnn']" "nn.dataset_name=tiny_imagenet" "nn.num_shared_classes=50" "nn.num_novel_classes_per_task=25"

# efficient_net
python src/la/scripts/run_part_shared_part_novel.py "nn.task_models=['efficient_net']" "nn.dataset_name=tiny_imagenet" "nn.num_shared_classes=100" "nn.num_novel_classes_per_task=25"
python src/la/scripts/run_part_shared_part_novel.py "nn.task_models=['efficient_net']" "nn.dataset_name=tiny_imagenet" "nn.num_shared_classes=50" "nn.num_novel_classes_per_task=25"
python src/la/scripts/analyze_part_shared_part_novel.py
