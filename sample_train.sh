#!/bin/bash

testDiscript="test_if_dropbox_sync_works"

rm -r results
mkdir results

mkdir -p results/logs/Hand
mkdir -p results/objects/Hand
th train-cifar100.lua --hier Hand --loadFrom ./model.model --dataRoot ~/cifar-100-batches-t7/ --saveTo results/objects/Hand --device 1 > results/logs/Hand/log.txt --epochs 1 &

mkdir -p results/logs/Visual
mkdir -p results/objects/Visual
th train-cifar100.lua --hier Visual --loadFrom ./model.model --dataRoot ~/cifar-100-batches-t7/ --saveTo results/objects/Visual --device 2 > results/logs/Visual/log.txt --epochs 1 &

mkdir -p results/logs/Imgnt
mkdir -p results/objects/Imgnt
th train-cifar100.lua --hier Imgnt --loadFrom ./model.model --dataRoot ~/cifar-100-batches-t7/ --saveTo results/objects/Imgnt --device 3 > results/logs/Imgnt/log.txt --epochs 1 &

mkdir -p results/logs/Rand
mkdir -p results/objects/Rand
th train-cifar100.lua --hier Rand --loadFrom ./model.model --dataRoot ~/cifar-100-batches-t7/ --saveTo results/objects/Rand --device 4 > results/logs/Rand/log.txt --epochs 1 &

wait

echo "test finished. results products at ./results directory"

testDirName="results_"`date | tr " :" _`"_"$testDiscript
mkdir testResults/$testDirName
cp -r results/* testResults/$testDirName
echo "results copied to testResults/"$testDirName

dropbox exclude add `find -name \*.model`
