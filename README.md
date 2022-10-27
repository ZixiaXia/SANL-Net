# SANL-Net

## Structure-aware dehazing of sewer inspection images based on monocular depth cues 
[[Paper]](https://onlinelibrary.wiley.com/doi/abs/10.1111/mice.12900)

![SANL-Net](https://user-images.githubusercontent.com/44375942/197808050-5b3aac17-b6df-453b-97b8-5c6019f64d6a.png)

## Table of contents
* [Prerequisites](#prerequisites)
* [Getting Started](#getting-started)
    * [Installation](#installation)
    * [Dataset Preparation](#dataset-preparation)
    * [Testing](#testing)
    * [Training](#training)
* [Citation](#citation)

## Prerequisites

- Python 3.6
- NumPy
- PyTorch 1.0+

## Getting Started
### Installation
```
https://github.com/ZixiaXia/SANL-Net.git
```

### Train
```
python train.py
```
The trained model is saved in the `./ckpt`.
```
├── ckpt/
│   ├── .pth                           /* model file
|   ├── .txt                           /* log file  
```

### Testing
```
python infer.py
```
The inferred results are saved in the `./ckpt/(SANLNet) prediction_40000`.
```
├── ckpt/
│   ├── .pth                           /* model file
|   ├── .txt                           /* log file
│   ├── (SANLNet) prediction_40000/    /* saved results    
```

## Citation
Please cite this paper if it helps your research:
```bibtex
@article{xia2022structure,
  title={Structure-aware dehazing of sewer inspection images based on monocular depth cues},
  author={Xia, Zixia and Guo, Shuai and Sun, Di and Lv, Yaozhi and Li, Honglie and Pan, Gang},
  journal={Computer-Aided Civil and Infrastructure Engineering},
  year={2022},
  publisher={Wiley Online Library}
}
```

