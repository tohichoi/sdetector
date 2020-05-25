import cv2
import numpy as np
from collections import deque
from sdetector import *
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import threading
import logging
import time
import datetime
import os

# matplotlib.use('GTK3Agg')

plt.rcParams['interactive']

read_queue=deque()
stop_event=threading.Event()
stop_event.clear()
capture_started_event=threading.Event()
capture_started_event.clear()
Config.video_src='act3.avi'
os.environ['DISPLAY'] = ':0'

capture_thread=CaptureThread(name='CaptureThread', 
    args=(read_queue, stop_event, capture_started_event))
capture_thread.start()

logging.info('Waiting for capture starting')
capture_started_event.wait()

# fig, ax = plt.subplots(1, 2)

frame=None
prev_frame=None
while True:

    # check if capture thread is running
    if not capture_thread.is_alive():
        logging.info(f'{capture_thread.name} is not running')
        break

    # try:
    if len(read_queue) < 1:
        time.sleep(0.01)
        continue

    vframe=read_queue.popleft()

    prev_frame=frame
    frame=vframe.frame

    # ax[0].imshow(frame)
    # ax[1].imshow(frame)
    plt.imshow(frame)
    plt.show()

        # if not show_frame(org=frame, waitKey=False):
        #     logging.info('Stopping capture')
        #     stop_event.set()
        #     continue

    # except IndexError:
    #     time.sleep(0.01)
    #     continue



