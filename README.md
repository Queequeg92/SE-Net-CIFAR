SE-Net Incorporates with ResNet and WideResnet on CIFAR-10/100 Dataset
=============
----------

This is a SE-Net implementation based on "Squeeze-and-Excitation Networks" [3] on CVPR 2017 "Beyond Imagenet" workshop.  
We combine SE Module with ResNet-164 and WideResnet28-10 to construct SeResNet-164 and SeWideResNet28-10 respectively. For details of ResNet-164 and WideResNet28-10, please refers to the original papers [1] and [2].  
We evaluate SeResNet-164 and SeWideResNet28-10 on cifar-10 and cifar-100 datasets. 
For details of the hyperparameters and training processes, please refer to the /scripts folder.

## SeResNet-164 VS ResNet-164 on cifar-10:
Accuracy: 95.12 vs 94.92 (94.54 reported by [1])  
![image](/doc/p1.png)

## SeResNet-164 VS ResNet-164 on cifar-100:
Accuracy: 78.09 vs 76.53 (75.67 reported by [1])  
![image](/doc/p2.png)

## SeWideResNet28-10 VS WideResNet28-10 on cifar-10:  
Accuracy: both around 96.10 (96.00 reported by [2])  
![image](/doc/p3.png)

## SeWideResNet28-10 VS WideResNet28-10 on cifar-100: 
Accuracy: both around 81.2 (80.75 reported by [2])  
![image](/doc/p4.png)

## Coarse Conclusion:
SE Module seems to work better with thin networks than wide networks on CIFAR-10 and CIFAR-100 datasets. 

## To-Do:
More networks with SE Module.  
**Welcome to make contributions!**

## Pre-requisites:
pytorch http://pytorch.org/  
tensorboard https://www.tensorflow.org/get_started/summaries_and_tensorboard  
tensorboard-pytorch https://github.com/lanpa/tensorboard-pytorch

## How to Run:
```shell
# cd to the /scripts folder.
cd /path-to-this-repository/scripts  
# run the shells.
sh resnet164.sh
```
## References:
[1] He, Kaiming, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. "Identity
    mappings in deep residual networks." In European Conference on
    Computer Vision, pp. 630-645. Springer International Publishing, 2016.  
[2] Zagoruyko, Sergey, and Nikos Komodakis. "Wide residual networks." arXiv
    preprint arXiv:1605.07146 (2016).  
[3] Hu, Jie, Li Shen, and Gang Sun. "Squeeze-and-Excitation Networks." arXiv preprint arXiv:1709.01507 (2017).
