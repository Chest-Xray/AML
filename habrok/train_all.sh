#!/bin/bash

for model in baseline densenet161 densenet201; do
    if [ "$item" == "baseline" ]; then
        for training in pretrained scratch; do
            sbatch ./train_one.sh $model $training
        done
    else
        sbatch ./train_one.sh $model
    fi
done
