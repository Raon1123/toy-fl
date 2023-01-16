data="--data_dir /home/ayp/datasets --dataset ${dataset} --verbose"
model="--model MLP --mlp_layers 784 200 200 10"
training="--num_rounds 1 --num_workers 0 --lr 0.01 --momentum 0.9 --local_epoch 1"
client="-N 100 -C 5 --label_distribution Dirichlet --label_dirichlet 0.1"
seeds="--seeds 0 1 2 3 4 5 6 7 8 9"
consts="${data} ${model} ${training} ${client} ${seeds}"

python main.py --postfix test --device 0 \
    ${1} --active_algorithm FedCor \
    1> outs/fmnist_test.out 2> outs/fmnist_test.err &
