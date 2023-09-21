# python src/la/scripts/run_same_classes_disj_samples.py "nn.dataset_name=cifar100" "nn/model=vanilla_cnn"
# python src/la/scripts/run_same_classes_disj_samples.py "nn.dataset_name=cifar100" "nn/model=resnet"
# python src/la/scripts/run_same_classes_disj_samples.py "nn.dataset_name=cifar100" "nn/model=shake_resnet"
# python src/la/scripts/run_same_classes_disj_samples.py "nn.dataset_name=cifar100" "nn/model=efficient_net"
# python src/la/scripts/run_same_classes_disj_samples.py "nn.dataset_name=tiny_imagenet" "nn/model=resnet"
# python src/la/scripts/run_same_classes_disj_samples.py "nn.dataset_name=tiny_imagenet" "nn/model=shake_resnet"


python src/la/scripts/run_same_classes_disj_samples.py "nn.dataset_name=cifar100" "nn/model=vanilla_cnn"
python src/la/scripts/run_same_classes_disj_samples.py "nn.dataset_name=cifar100" "nn/model=efficient_net"
python src/la/scripts/run_same_classes_disj_samples.py "nn.dataset_name=tiny_imagenet" "nn/model=efficient_net"
python src/la/scripts/run_same_classes_disj_samples.py "nn.dataset_name=tiny_imagenet" "nn/model=vanilla_cnn"
