import shutil

import cv2
import numpy as np
from collections import deque
from sdetector import *
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import cv2
import threading
import logging
import time
import datetime
import os
from threading import BoundedSemaphore

# matplotlib.use('GTK3Agg')

# plt.rcParams['interactive']
logging.basicConfig(level=logging.INFO,
                    #    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
                    format='%(asctime)s : %(funcName)s : %(message)s')

logger = logging.getLogger()
logger.setLevel(logging.INFO)

read_queue = deque()
stop_event = threading.Event()
stop_event.clear()
capture_started_event = threading.Event()
capture_started_event.clear()
Config.video_src = 'data/act4.avi'
Config.send_video = False
pool_sema = BoundedSemaphore(value=1)

# os.environ['DISPLAY'] = ':0'

capture_thread = CaptureThread(name='CaptureThread',
                               args=(read_queue, stop_event, capture_started_event, 0, False))
capture_thread.start()

logging.info('Waiting for capture starting')
capture_started_event.wait()

# fig, ax = plt.subplots(1, 2)

dname, ext = os.path.splitext(Config.video_src)
if not os.path.exists(dname):
    # shutil.rmtree(dname)
    os.mkdir(dname)

frame = None
prev_frame = None
count = 0
while True:

    # check if capture thread is running
    if not capture_thread.is_alive():
        logging.info(f'{capture_thread.name} is not running')
        break

    # try:
    if len(read_queue) < 1:
        time.sleep(0.01)
        continue

    vframe = read_queue.popleft()

    prev_frame = frame
    frame = vframe.frame

    fn0 = f'frame-{count:04d}.jpg'
    fn = os.path.join(dname, fn0)
    # fn=fn.replace('\\', '/')
    fn=fn.replace('/', '\\')
    logging.info(f'Writing : {fn}')
    cv2.imwrite(fn, frame)
    count += 1

    # ax[0].imshow(frame)
    # ax[1].imshow(frame)
    # plt.imshow(frame)
    # plt.show()

    # if not show_frame(org=frame, waitKey=False):
    #     logging.info('Stopping capture')
    #     stop_event.set()
    #     continue

    # except IndexError:
    #     time.sleep(0.01)
    #     continue
