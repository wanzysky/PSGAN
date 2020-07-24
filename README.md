# PSGAN

This project is forked from the official implementation of PSGAN: "[PSGAN: Pose and Expression Robust Spatial-Aware GAN for Customizable Makeup Transfer](https://arxiv.org/abs/1909.06956)". 

I have added GPU inference, multiple-gpu support and optimized training and inference speed.

![](psgan_framework.png)

## Checklist
- [x] more results 
- [ ] video demos
- [ ] partial makeup transfer example
- [ ] interpolated makeup transfer example
- [x] inference on GPU
- [x] training code


## Requirements
   The code was tested on Ubuntu 16.04, with Python 3.6 and PyTorch 1.5.

## Test

1. `python3 demo.py` or `python3 demo.py --device cuda` for gpu inference.

NOTE: You need dlib gpu support for fully gpu inference.

## Train
1. Download dataset from [here](https://1drv.ms/u/s!AgqNJZCiLRDCgaYWgH5Pe5ppH3qc4w?e=jCnods).
2. Check config.py to modify `default.data_path` to your data path.
  
## More Results

#### MT-Dataset (frontal face images with neutral expression)

![](MT-results.png)


#### MWild-Dataset (images with different poses and expressions)

![](MWild-results.png)

#### Video Makeup Transfer (by simply applying PSGAN on each frame)

![](Video_MT.png)

## Citation
Please consider citing this project in your publications if it helps your research. The following is a BibTeX reference. The BibTeX entry requires the url LaTeX package.

~~~
@inproceedings{jiang2019psgan,
  title={PSGAN: Pose and Expression Robust Spatial-Aware GAN for Customizable Makeup Transfer},
  author={Jiang, Wentao and Liu, Si and Gao, Chen and Cao, Jie and He, Ran and Feng, Jiashi and Yan, Shuicheng},
  booktitle={CVPR},
  year={2020}
}
~~~

## Acknowledge
Some of the codes are built upon [face-parsing.PyTorch](https://github.com/zllrunning/face-parsing.PyTorch) and [BeautyGAN](https://github.com/wtjiang98/BeautyGAN_pytorch). 
