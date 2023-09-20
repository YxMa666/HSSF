# Model-Heterogeneous Semi-Supervised Federated Learning for Medical Image Segmentation

## Introduction

This is an official release of the paper **Model-Heterogeneous Semi-Supervised Federated Learning for Medical Image Segmentation**, including the network implementation and the training scripts.

> [**Model-Heterogeneous Semi-Supervised Federated Learning for Medical Image Segmentation**](https://arxiv.org/abs/2207.04655),   <br/>
> Yuxi Ma, Jiacheng Wang, Jing Yang, Liansheng Wang <br/>
> Published in: IEEE Transactions on Medical Imaging (TMI) <br/>
<!-- > [[arXiv](https://arxiv.org/abs/2207.04655)][[Bibetex](https://github.com/jcwang123、FedLC#Citation)][[Supp](https://jcwang123.github.io/assets/pdfs/eccv22/supp.pdf)] -->

<div align="center" border=> <img src=fig/fig1.png width="700" > </div>
<div align="center" border=> <img src=fig/fig2.png width="700" > </div>


## News
- **[3/1 2023] Codes for Head Calibration are tuned.**
- **[7/12 2022] We have released the training codes.**
- **[7/25 2022] We have uploaded the test scripts.**
- **[7/12 2022] We have released the pre-print manuscript.**
- **[7/11 2022] We have released the pre-trained weights on the polyp segmentation.**
- **[7/4 2022] We have released the pre-processing scripts.**
- **[7/4 2022] We have created this repo.**

## Code List

- [x] Network
- [x] Pre-processing
- [x] Training Codes
- [x] Pretrained Weights

For more details or any questions, please feel easy to contact us by email (jiachengw@stu.xmu.edu.cn).

## Usage

### Dataset
In this paper, we perform the experiments using three imaging modalities, including the polyp images, fundus images, and prostate MR images. They could be downloaded from the public websites, or copied from [FedDG](https://github.com/liuquande/FedDG-ELCFS) and [PraNet](https://github.com/DengPingFan/PraNet).

### Pre-processing
After downloading the data resources, please run the file `utils/prepare_dataset.py`. Note that the file directory should be replaced with yours.

### Training 
Run the train script `$ python scripts/train_lcfed.py`.

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