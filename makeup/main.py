#!/usr/bin/python
# -*- encoding: utf-8 -*-
import os.path as osp
pwd = osp.split(osp.realpath(__file__))[0]
import sys
import time
sys.path.append(pwd + '/..')

import numpy as np
from PIL import Image
import torch
from torch.autograd import Variable
import torch.nn.functional as F
from torch.backends import cudnn
from torchvision import transforms
import cv2
from neupeak.utils import webcv2
from smart_path import smart_path
from fire import Fire
from tqdm import tqdm

import faceutils as futils
from makeup.solver_makeup import Solver_makeupGAN
from makeup.preprocess import preprocess

cudnn.benchmark = True
solver = Solver_makeupGAN()

def load_image(path: smart_path):
    with path.open("rb") as reader:
        data = np.fromstring(reader.read(), dtype=np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if img is None:
            return
        img = img[..., ::-1]
    return img

def main(
    makeup_dir="faces/makeup",
    non_makeup_dir="faces/no_makeup",
    img_size=256,
    speed=False):
    makeup_dir = smart_path(makeup_dir)
    non_makeup_dir = smart_path(non_makeup_dir)

    makeup_paths = list(makeup_dir.glob("*"))
    non_makeup_paths = list(non_makeup_dir.glob("*"))

    assert len(makeup_paths) > 0
    assert len(non_makeup_paths) > 0

    random = np.random.RandomState(seed=0)
    while True:
        makeup_path = random.choice(makeup_paths)
        ant_makeup_path = random.choice(makeup_paths)
        non_makeup_path = random.choice(non_makeup_paths)
        
        makeup_image = load_image(makeup_path)
        non_makeup_image = load_image(non_makeup_path)
        if makeup_image is None or non_makeup_image is None:
            continue

        transferred_image = solver.test(
            *preprocess(Image.fromarray(non_makeup_image)),
            *preprocess(Image.fromarray(makeup_image)))
        if speed:
            input_1 = preprocess(Image.fromarray(non_makeup_image))
            input_2 = preprocess(Image.fromarray(makeup_image))
            start = time.time()
            for _ in tqdm(range(100)):
                transferred_image = solver.test(*input_1, *input_2)

            print("inference time", time.time() - start)

        webcv2.imshow("source", non_makeup_image[..., ::-1])
        webcv2.imshow("reference", makeup_image[..., ::-1])
        webcv2.imshow("result", np.array(transferred_image)[..., ::-1])
        webcv2.waitKey()


if __name__ == '__main__':
    # source image, reference image
    Fire(main)
