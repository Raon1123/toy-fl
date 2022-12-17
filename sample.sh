#!/bin/bash

dataset=fmnist
data="--data_dir /home/ayp/datasets --dataset ${dataset} --verbose"
model="--model MLP --mlp_layers 784 200 200 10"
training="--num_rounds 1 --num_workers 0 --lr 0.01 --momentum 0.9 --local_epoch 1"
client="-N 100 -C 5 --label_distribution Dirichlet --label_dirichlet 0.1"
seeds="--seeds 0 1 2"

expfunc() {
        expstr="${dataset}/Random_${2}"
        python main.py --postfix ${2} --device 0 \
                ${1} --active_algorithm Random \
                1> outs/${expstr}.out 2> outs/${expstr}.err &

        expstr=${dataset}/LossSampling_${2}
        python main.py --postfix ${2} --device 1 \
                ${1} --active_algorithm LossSampling  \
                1> outs/${expstr}.out 2> outs/${expstr}.err &

        expstr=${dataset}/powd_${2}
        python main.py --postfix ${2}2 --device 2 \
                ${1} --active_algorithm powd --powd 10  \
                1> outs/${expstr}2.out 2> outs/${expstr}2.err &

        expstr=${dataset}/powd_${2}
        python main.py --postfix ${2}3 --device 3 \
                ${1} \
                --active_algorithm powd --powd 15  \
                1> outs/${expstr}3.out 2> outs/${expstr}3.err &

        expstr=${dataset}/powd_${2}
        python main.py --postfix ${2}5 --device 0 \
                ${1} \
                --active_algorithm powd --powd 25  \
                1> outs/${expstr}5.out 2> outs/${expstr}5.err &

        expstr=${dataset}/GradientBADGE_${2}
        python main.py --postfix ${2} --device 1 \
                ${1} --active_algorithm GradientBADGE \
                1> outs/${expstr}.out 2> outs/${expstr}.err &
}


postf="tst1"
client="-N 100 -C 5 --label_distribution Dirichlet --label_dirichlet 0.1"
consts="${data} ${model} ${training} ${client} ${seeds}"

expfunc "${consts}" ${postf}

postf="tst3"
client="-N 100 -C 5 --label_distribution Dirichlet --label_dirichlet 0.3"
consts="${data} ${model} ${training} ${client} ${seeds}"

expfunc "${consts}" ${postf}

postf="tst5"
client="-N 100 -C 5 --label_distribution Dirichlet --label_dirichlet 0.5"
consts="${data} ${model} ${training} ${client} ${seeds}"

expfunc "${consts}" ${postf}