# CIFAR100
# N=10
python src/la/scripts/prepare_data_part_shared_part_novel.py -m "dataset=cifar100" "num_shared_classes=80" "num_novel_classes_per_task=10"
python src/la/scripts/prepare_data_part_shared_part_novel.py -m "dataset=cifar100" "num_shared_classes=60" "num_novel_classes_per_task=10"
python src/la/scripts/prepare_data_part_shared_part_novel.py -m "dataset=cifar100" "num_shared_classes=40" "num_novel_classes_per_task=10"
python src/la/scripts/prepare_data_part_shared_part_novel.py -m "dataset=cifar100" "num_shared_classes=20" "num_novel_classes_per_task=10"
# N=5
python src/la/scripts/prepare_data_part_shared_part_novel.py -m "dataset=cifar100" "num_shared_classes=80" "num_novel_classes_per_task=5"
python src/la/scripts/prepare_data_part_shared_part_novel.py -m "dataset=cifar100" "num_shared_classes=60" "num_novel_classes_per_task=5"
python src/la/scripts/prepare_data_part_shared_part_novel.py -m "dataset=cifar100" "num_shared_classes=40" "num_novel_classes_per_task=5"
python src/la/scripts/prepare_data_part_shared_part_novel.py -m "dataset=cifar100" "num_shared_classes=20" "num_novel_classes_per_task=5"
#
# TINY_IMAGENET
python src/la/scripts/prepare_data_part_shared_part_novel.py -m "dataset=tiny_imagenet" "num_shared_classes=100" "num_novel_classes_per_task=25"
python src/la/scripts/prepare_data_part_shared_part_novel.py -m "dataset=tiny_imagenet" "num_shared_classes=50" "num_novel_classes_per_task=25"
