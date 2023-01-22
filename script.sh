dataset=cifar10
data="--data_dir /home/ayp/datasets --dataset ${dataset} --verbose"
model="--model NaiveCNN "
training="--num_rounds 2000 --num_workers 0 --lr 0.01 --momentum 0.9 --local_epoch 1"
client="-N 100 -C 5 --label_distribution Dirichlet --label_dirichlet 0.3"
seeds="--seeds 0 1 2 3 4 5 6 7 8 9"
postf="dir03"

expstr="${dataset}/FedCor_${postf}"

python main.py --postfix ${postf} --device 0 \
        ${data} ${model} ${training} ${client} ${seeds} \
        --active_algorithm FedCor \
        1> outs/${expstr}.out 2> outs/${expstr}.err &