# Video Motion Prompts

<p align="center">
    
<p>


<p align="center">
  <img src="./fig/pipeline.png" width="600"/>
  <br>
  &nbsp&nbsp 🌺 <a href="https://q1xiangchen.github.io/motion-prompts/">Project Website</a> ｜ 📑 <a href="https://arxiv.org/abs/2407.03179">Paper</a> &nbsp&nbsp
</p>

This is an official pytorch implementation of our ACML 2024 paper [Motion meets Attention: Video Motion Prompts](https://arxiv.org/abs/2407.03179). In this repository, we provide PyTorch code for training and testing our video motion prompts (VMPs) layer with TimeSformer model. 



# 🌹Getting Started

## 🪛 Installation
1. Clone the repository:
```
git clone git@github.com:q1xiangchen/VMPs.git
cd VMPs
```
2. Install the required dependencies below (or refer to [environment.yml](environment.yml)):

- torchvision: `pip install torchvision` or `conda install torchvision -c pytorch`
- [fvcore](https://github.com/facebookresearch/fvcore/): `pip install 'git+https://github.com/facebookresearch/fvcore'`
- simplejson: `pip install simplejson`
- einops: `pip install einops`
- timm: `pip install timm`
- PyAV: `conda install av -c conda-forge`
- psutil: `pip install psutil`
- scikit-learn: `pip install scikit-learn`
- OpenCV: `pip install opencv-python`
- tensorboard: `pip install tensorboard`
- wandb: `pip install wandb`
- h5py: `pip install h5py`

3. Lastly, build the TimeSformer codebase by running:
  ```
  python setup.py build develop
  ```

## 🍀 Model Weights
| Model | Pretrained dataset | Fine-tuned dataset | Link |
| --- | --- | --- | --- |
| TimeSformer | Kinetics-600 | - | [Download](link_here) |
| TimeSformer | Kinetics-600 | MPII-Cooking-2 | [Download](link_here) |
| **VMPs** + TimeSformer | Kinetics-600 | MPII-Cooking-2 | [Download](link_here) |

## 🗂️ Dataset Preparation
Please use the dataset preparation instructions provided in [DATASET.md](timesformer/datasets/DATASET.md).



## 🚀 Training and Testing
To train and test the VMPs layer with TimeSformer model, please refer to the instructions in [TRAIN.md](./configs/TRAIN.md).


# 📎 License

Our code is released under the MIT license. See [LICENSE](LICENSE) for more information.
The portions of the TimeSformer codebase are released under CC-BY-NC 4.0 license. See [LICENSE_TIMESFORMER](https://github.com/facebookresearch/TimeSformer?tab=readme-ov-file#license) for more information.

# ✏️ Citation
If you find VMPs useful in your research, please consider 📝 citing our paper or ⭐️ star our repo:

```BibTeX
@inproceedings{
chen2024motion,
title={Motion meets Attention: Video Motion Prompts},
author={Qixiang Chen and Lei Wang and Piotr Koniusz and Tom Gedeon},
booktitle={The 16th Asian Conference on Machine Learning (Conference Track)},
year={2024},
url={https://openreview.net/forum?id=nIDAT99Vhb}
}
```


# 📣 Acknowledgment

Qixiang Chen conducted this research under the supervision of Lei Wang for his final year honors research project at ANU. He is a recipient of research sponsorship from Active Intelligence Australia Pty Ltd in Perth, Western Australia, including The Active Intelligence Research Challenge Award. This work was also supported by the NCI Adapter Scheme Q4 2023, the NCI National AI Flagship Merit Allocation Scheme, and the National Computational Merit Allocation Scheme 2024 (NCMAS 2024), with computational resources provided by NCI Australia, an NCRIS-enabled capability supported by the Australian Government.

This codebase is built on top of [TimeSformer](https://github.com/facebookresearch/TimeSformer) of facebookresearch, and we thank the authors for their work. 