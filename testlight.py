import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
import random
from scipy.stats import skewnorm, norm, expon
from color_model import *


filelist = 'light1.txt'
images, filepath = read_image_from_filelist(filelist)


# 전처리
# 0. to gray scale
# 1. smoothing
# 2. histogram equalization

# 히스토그램의 변화
# 전 프레임과 histogram 이 얼마나 차이나는가?

# color difference 의 변화
