#!/usr/bin/env bash

cd ..

python train_cifar.py \
--arch=SeResNet164 \
--dataset=cifar10 \
--epochs=160 \
--start_epoch=0 \
--batch_size=128 \
--lr=0.1 \
--lr_schedule=2 \
--momentum=0.9 \
--weight_decay=0.0002 \
--ckpt_path=''

python train_cifar.py \
--arch=SeResNet164 \
--dataset=cifar100 \
--epochs=160 \
--start_epoch=0 \
--batch_size=128 \
--lr=0.1 \
--lr_schedule=2 \
--momentum=0.9 \
--weight_decay=0.0002 \
--ckpt_path=''