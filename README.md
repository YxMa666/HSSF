# Model-Heterogeneous Semi-Supervised Federated Learning for Medical Image Segmentation

## Introduction

This is an official release of the paper **Model-Heterogeneous Semi-Supervised Federated Learning for Medical Image Segmentation**, including the network implementation and the training scripts.

> [**Model-Heterogeneous Semi-Supervised Federated Learning for Medical Image Segmentation**](https://arxiv.org/abs/2207.04655),   <br/>
> Yuxi Ma, Jiacheng Wang, Jing Yang, Liansheng Wang <br/>
<!-- > Published in: IEEE Transactions on Medical Imaging (TMI) <br/> -->
<!-- > [[arXiv](https://arxiv.org/abs/2207.04655)][[Bibetex](https://github.com/jcwang123、FedLC#Citation)][[Supp](https://jcwang123.github.io/assets/pdfs/eccv22/supp.pdf)] -->

<div align="center" border=> <img src=fig/fig1.png width="700" > </div>
<div align="center" border=> <img src=fig/fig2.png width="700" > </div>


## News
- **[3/1 2023] Codes for Head Calibration are tuned.**
- **[7/12 2022] We have released the training codes.**
- **[7/25 2022] We have uploaded the test scripts.**
- **[7/12 2022] We have released the pre-print manuscript.**
- **[7/11 2022] We have released the pre-trained weights on the polyp segmentation.**
- **[9/20 2023] We have created this repo.**

## Code List

- [x] Network
- [x] Training Codes
- [x] Pretrained Weights

For more details or any questions, please feel easy to contact us by email (jiachengw@stu.xmu.edu.cn).

## Usage

### Dataset
In this paper, we perform the experiments using two imaging modalities, including the polyp images ([Kvasir](https://datasets.simula.no/kvasir-seg/), [CVC-ClinicDB](https://polyp.grand-challenge.org/CVCClinicDB/), [CVC-ColonDB](https://ieeexplore.ieee.org/document/7294676), [CVC-300](https://arxiv.org/abs/1612.00799), [EndoTectETIS](https://link.springer.com/article/10.1007/s11548-013-0926-3)), and [ISIC-2018](https://challenge.isic-archive.com/data/) images.

### Training 
Run the train script `$ python train.py`.

### Testing
Please download the pre-trained weights from Baidu Disk (https://pan.baidu.com/s/10HkQ90xeFcHMaNgfIyT0iw, a1sm) and put them in the project directory.

Rename the directory as `logs/{dataset}/{exp_name}/model/`.

Run the test script `$ python test.py`.

### Result
The test **Dice** scores and **HD95** scores on the Polyp dataset are:

<div align="center" border=> <img src=fig/results.png width="700" > </div>

## Citation
If you find HSSF useful in your research, please consider citing:
```
@inproceedings{wang2022personalizing,
  title={Personalizing Federated Medical Image Segmentation via Local Calibration},
  author={Wang, Jiacheng and Jin, Yueming and Wang, Liansheng},
  booktitle={European Conference on Computer Vision},
  pages={456--472},
  year={2022},
  organization={Springer}
}
```