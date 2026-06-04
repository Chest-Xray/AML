#!/bin/bash

for model in vgg16 densenet161 densenet201; do
    if [ "$item" != "vgg16" ]; then
        for training in pretrained scratch; do
            sbatch ./habrok/train_one.sh $model $training
	    echo "training $model $training"
        done
    else
        sbatch ./habrok/train_one.sh $model
	echo "training $model"
    fi
done
