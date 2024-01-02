# Model-Heterogeneous Semi-Supervised Federated Learning for Medical Image Segmentation

## Introduction

This is an official release of the paper **Model-Heterogeneous Semi-Supervised Federated Learning for Medical Image Segmentation**, including the network implementation and the training scripts.

> [**Model-Heterogeneous Semi-Supervised Federated Learning for Medical Image Segmentation**](https://ieeexplore.ieee.org/document/10379169),   <br/>
> Yuxi Ma, Jiacheng Wang, Jing Yang, Liansheng Wang <br/>
> Published in: IEEE Transactions on Medical Imaging (TMI) <br/>

<div align="center" border=> <img src=fig/fig1.png width="700" > </div>
<div align="center" border=> <img src=fig/fig2.png width="700" > </div>


## Code List

- [x] Network
- [x] Training Codes
- [x] Models Weights


## Usage

### Dataset
In this paper, we perform the experiments using two imaging modalities, including the polyp images ([Kvasir](https://datasets.simula.no/kvasir-seg/), [CVC-ClinicDB](https://polyp.grand-challenge.org/CVCClinicDB/), [CVC-ColonDB](https://ieeexplore.ieee.org/document/7294676), [CVC-300](https://arxiv.org/abs/1612.00799), [EndoTectETIS](https://link.springer.com/article/10.1007/s11548-013-0926-3)), and [ISIC-2018](https://challenge.isic-archive.com/data/) images.

### Training 
Run the train script `$ python train.py`.

### Testing
Please download the pre-trained weights from Baidu Disk (https://pan.baidu.com/s/1MHkadVGC3UVCVchR2AtAVg?pwd=8m7y, 8m7y) and put them in the project directory.

Rename the log_dir as `./checkpoints`.

Run the test script `$ python test.py`.

### Result
The test **Dice** scores and **HD95** scores on the Polyp dataset are:

<div align="center" border=> <img src=fig/results.png width="700" > </div>

## Citation
If you find HSSF useful in your research, please consider citing:
```
@ARTICLE{Ma2023Model,
  author={Ma, Yuxi and Wang, Jiacheng and Yang, Jing and Wang, Liansheng},
  journal={IEEE Transactions on Medical Imaging}, 
  title={Model-Heterogeneous Semi-Supervised Federated Learning for Medical Image Segmentation}, 
  year={2023},
  volume={},
  number={},
  pages={1-1},
  doi={10.1109/TMI.2023.3348982}}
```