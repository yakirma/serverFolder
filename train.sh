#!/bin/bash

rm -r results
mkdir results

mkdir results/Hand
th train-cifar100.lua --hier Hand --loadFrom ./model.model --dataRoot ../cifar-100-batches-t7/ --saveTo results/Hand --device 1 > results/Hand/log.txt --epochs 20 &

mkdir results/Visual
th train-cifar100.lua --hier Visual --loadFrom ./model.model --dataRoot ../cifar-100-batches-t7/ --saveTo results/Visual --device 2 > results/Visual/log.txt --epochs 20 &

mkdir results/Imgnt
th train-cifar100.lua --hier Imgnt --loadFrom ./model.model --dataRoot ../cifar-100-batches-t7/ --saveTo results/Imgnt --device 3 > results/Imgnt/log.txt --epochs 20 &

mkdir results/Rand
th train-cifar100.lua --hier Rand --loadFrom ./model.model --dataRoot ../cifar-100-batches-t7/ --saveTo results/Rand --device 4 > results/Rand/log.txt --epochs 20 &

wait
echo "test finished. results at ./results directory"
