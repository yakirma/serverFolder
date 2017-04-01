#!/bin/bash

testDiscript="test_final_config1"

rm -r results
mkdir results


echo "Hand test started"
mkdir -p results/logs/Hand
mkdir -p results/objects/Hand
th train-cifar100.lua --hier Hand --loadFrom ./model.model --dataRoot ~/cifar-100-batches-t7/ --device 1 --epochs 100 --saveTo results/objects/Hand  --precNighboursNum 10 --doValid 1 --doTrain 1 > results/logs/Hand/log.txt &

echo "Visual test started"
mkdir -p results/logs/Visual
mkdir -p results/objects/Visual
th train-cifar100.lua --hier Visual --loadFrom ./model.model --dataRoot ~/cifar-100-batches-t7 --device 2 --epochs 100 --saveTo results/objects/Visual --precNighboursNum 10 --doValid 1 --doTrain 1 > results/logs/Visual/log.txt &

echo "Imgnt test started"
mkdir -p results/logs/Imgnt
mkdir -p results/objects/Imgnt
th train-cifar100.lua --hier Imgnt --loadFrom ./model.model --dataRoot ~/cifar-100-batches-t7/ --device 1 --epochs 100 --saveTo results/objects/Imgnt --precNighboursNum 10 --doValid 1 --doTrain 1 > results/logs/Imgnt/log.txt &

echo "Rand test started"
mkdir -p results/logs/Rand
mkdir -p results/objects/Rand
th train-cifar100.lua --hier Rand --loadFrom ./model.model --dataRoot ~/cifar-100-batches-t7/ --device 1 --epochs 100 --saveTo results/objects/Rand  --precNighboursNum 10 --doValid 1 --doTrain 1 > results/logs/Rand/log.txt &

echo "None test started"
mkdir -p results/logs/None
mkdir -p results/objects/None
th train-cifar100.lua --hier None --loadFrom ./model.model --dataRoot ~/cifar-100-batches-t7/ --device 2 --epochs 100 --saveTo results/objects/None  --precNighboursNum 10 --doValid 1 --doTrain 1 > results/logs/None/log.txt &

echo "Waiting for tests to end ..."
wait

echo "test finished. results products at ./results directory"

testDirName="results_"`date | tr " :" _`"_"$testDiscript
mkdir testResults/$testDirName
cp -r results/* testResults/$testDirName
echo "results copied to testResults/"$testDirName

dropbox exclude add `find -name \*.model`
