#!/usr/bin/env bash

cd ..

python train_cifar.py \
--arch=SeWideResNet_28_10 \
--dataset=cifar10 \
--epochs=200 \
--start_epoch=0 \
--batch_size=128 \
--lr=0.1 \
--lr_schedule=0 \
--momentum=0.9 \
--nesterov \
--weight_decay=0.0005 \
--ckpt_path=''

python train_cifar.py \
--arch=SeWideResNet_28_10 \
--dataset=cifar100 \
--epochs=200 \
--start_epoch=0 \
--batch_size=128 \
--lr=0.1 \
--lr_schedule=0 \
--momentum=0.9 \
--nesterov \
--weight_decay=0.0005 \
--ckpt_path=''