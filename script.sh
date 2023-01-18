data="--data_dir ./data --dataset cifar10 --verbose"
model="--model NaiveCNN "
training="--num_rounds 500 --num_workers 4 --lr 5e-3 --momentum 0.0 --local_epoch 3"
client="-N 100 -C 5 --label_distribution Dirichlet --label_dirichlet 0.3"
seeds="--seeds 0 1"
consts="${data} ${model} ${training} ${client} ${seeds}"

python main.py --postfix test --device 0 \
    ${consts} --active_algorithm FedCor \
    1> outs/fmnist_test.out 2> outs/fmnist_test.err &
