#!/bin/bash

dataset=cifar10

data="--data_dir /home/ayp/datasets --dataset ${dataset} --verbose"
model="--model NaiveCNN "
client="-N 100 -C 5 --label_distribution Dirichlet --label_dirichlet 0.3"
seeds="--seeds 0 1 2 3 4 5 6 7 8 9"

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


postf="cifar10-dir03-lee1-cnn"
training="--num_rounds 500 --num_workers 0 --local_epoch 1 --lr 0.1 --momentum 0.9"
consts="${data} ${model} ${training} ${client} ${seeds}"

expfunc "${consts}" ${postf}
