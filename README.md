# PLMR

**Title:** Partial Label Learning via Mutual Information Representation Learning

```
@ARTICLE{11203238,
  author={Feng, Jian and Huang, Linqing and Li, JiangNan and Ren, Kangrui and Fan, JinFu and Bu, QingKai and Gan, Min and Philip Chen, C. L.},
  journal={IEEE Transactions on Circuits and Systems for Video Technology}, 
  title={Partial Label Learning via Mutual Information Representation Learning}, 
  year={2025},
  volume={},
  number={},
  pages={1-1},
  keywords={Phase locked loops;Training;Optimization;Accuracy;Estimation;Videos;Representation learning;Noise;Mutual information;Fans;Partial label learning;mutual information estimation;cluster centers;representation learning},
  doi={10.1109/TCSVT.2025.3621307}
}
```

## Running PLMR

You need to download CIFAR-10 , MNIST, Kuzushiji-MNIST and Fashion-MNIST datasets into './data/'.

Please note that when partial_rate ∈ {0.1, 0.3, 0.5}, the corresponding same_weight ∈ {5, 3, 2}.

**Run cifar10**

```shell
python -u train.py --exp-dir experiment/CIFAR-10 --dataset cifar10 --partial_rate 0.5 \
--same_weight 2
```

**Run mnist**

```shell
python -u train.py --exp-dir experiment/MNIST-10 --dataset mnist --partial_rate 0.5 \
--same_weight 2
```

**Run kmnist**

```shell
python -u train.py --exp-dir experiment/KMNIST-10 --dataset kmnist --partial_rate 0.5 \
--same_weight 2
```

**Run kmnist**

```shell
python -u train.py --exp-dir experiment/KMNIST-10 --dataset kmnist --partial_rate 0.5 \
--same_weight 2
```
