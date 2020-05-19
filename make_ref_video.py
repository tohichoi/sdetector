import cv2
import numpy as np
import math
import statistics 
import datetime
import re
import os
import sys


for dirpath, dirnames, filenames in os.walk(sys.argv[1]):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')    
    vcap_out = cv2.VideoWriter(sys.argv[2], fourcc, 20.0, (1280, 720))
    for filename in filenames:
        filepath=os.path.join(dirpath, filename)
        frame=cv2.imread(filepath)
        # frame_gray=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # frame_gray=cv2.GaussianBlur(frame_gray, (3, 3), 0)
        vcap_out.write(frame)
    vcap_out.release()

