# Relay-SGD
This repository implements the Relay-SGD algorithm proposed in "RelaySum for Decentralized Deep Learning on Heterogeneous Data" [1]


# Available Models
* ResNet
* VGG-11
* MobileNet-V2
* LeNet-5

# Available Datasets
* CIFAR-10
* CIFAR-100
* Fashion MNIST
* Imagenette
* Imagenet

# Available Graph Topologies
* Chain Topology

# Requirements
* found in environment.yml file

# Hyper-parameters
* --world_size     = total number of agents
* --arch           = model to train
* --normtype       = type of normalization layer
* --dataset        = dataset to train; options: [cifar10, cifar100, fmnist, imagenette, imagenet]
* --batch_size     = batch size for training (batch_size = batch_size per agent x world_size)
* --epochs         = total number of training epochs
* --lr             = learning rate
* --momentum       = momentum coefficient
* --weight_decay   = weight decay
* --nesterov       = sets nesterov momentum as true
* --skew           = amount of skew in the data distribution (alpha parameter of Dirichlet Distribution); 0.001 = completely non-IID 
* --partition_type = random or non-iid-dirichlet


# How to run?

ResNet-20 with 10 agents ring topology:
```
python trainer.py --lr=0.1 --skew=0.1 --epochs=200 --arch=resnet --momentum=0.9 --seed=1234 --batch-size=512 --world_size=16 --momentum=0.9 --depth=20 --dataset=cifar10 --classes=10 --devices=4 --seed=123

```

# Reference
[1] Vogels, Thijs, et al. "Relaysum for decentralized deep learning on heterogeneous data." Advances in Neural Information Processing Systems 34 (2021): 28004-28015.

```
@article{vogels2021relaysum,
  title={Relaysum for decentralized deep learning on heterogeneous data},
  author={Vogels, Thijs and He, Lie and Koloskova, Anastasiia and Karimireddy, Sai Praneeth and Lin, Tao and Stich, Sebastian U and Jaggi, Martin},
  journal={Advances in Neural Information Processing Systems},
  volume={34},
  pages={28004--28015},
  year={2021}
}
```