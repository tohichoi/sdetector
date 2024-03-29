# encoding: utf-8
# import cProfile
import datetime
import os
import re
import threading
import time
from collections import deque
from queue import Queue
from threading import BoundedSemaphore

import matplotlib
import numpy as np

from scenemodel import SceneModel

matplotlib.use('Agg')
import imutils
import cv2
import collections
from telegram import Bot
import json
# from pomegranate import *
import sys
import logging
from scipy.ndimage import interpolation
from color_model import is_color


class Config():
    FPS = 20
    WRITE_FPS = 10
    SET_FRAME_SCALE = 0.5
    # 20 frames/sec * 30sec
    MAX_CAPTURE_BUFFER = FPS * 60
    MAX_TIMELINE = 3 * FPS
    MAX_LEARN_FRAMES = FPS * 2
    MAX_NHISTORY = FPS * 5
    # 1 minute
    ROI = [404, 0, 1006, 680]
    VIDEO_ORG_WIDTH = -1
    VIDEO_ORG_HEIGHT = -1
    VIDEO_WIDTH = -1
    VIDEO_HEIGHT = -1
    MAX_COLOR_SAMPLES = 100
    WAITKEY_MS = 50
    TG_CHAT_ID = None
    TG_TOKEN = None
    MAX_IOU = 0.02
    SKIPFRAME = 2
    RESIZEFRAME = True
    SCENECHANGE_VALUE = 0.08
    send_video = False
    statemodel = None
    imagemodel = None
    video_src = None

    tg_video_q = Queue()
    tg_bot = None


class State():
    UNKNOWN = 0
    ABSENT = 1
    ENTERED = 2
    PRESENT = 3
    LEFT = 4


class FrameWriteMode():
    IGNORE = 0
    SINGLE = 1
    APPEND = 2


class FPS:
    def __init__(self):
        # store the start time, end time, and total number of frames
        # that were examined between the start and end intervals
        self._start = None
        self._end = None
        self._numFrames = 0

    def start(self):
        # start the timer
        self._start = datetime.datetime.now()
        return self

    def stop(self):
        # stop the timer
        self._end = datetime.datetime.now()

    def update(self):
        # increment the total number of frames examined during the
        # start and end intervals
        self._numFrames += 1

    def elapsed(self):
        # return the total number of seconds between the start and
        # end interval
        return (self._end - self._start).total_seconds()

    def fps(self):
        # compute the (approximate) frames per second
        return self._numFrames / self.elapsed()


def width(roi):
    return roi[2] - roi[0]


def height(roi):
    return roi[3] - roi[1]


def area(roi):
    return width(roi) * height(roi)


def show_image(img):
    title = 'temp'
    cv2.imshow(title, img)
    cv2.waitKey(Config.WAITKEY_MS)
    cv2.destroyWindow(title)
    cv2.waitKey(Config.WAITKEY_MS)


def get_roi_frame(frame):
    x = Config.ROI[0]
    y = Config.ROI[1]
    w = Config.ROI[2] - x
    h = Config.ROI[3] - y

    return frame[y:y + h, x:x + w]


def show_frame(org=None, detect=None, model=None, threshold=None, difference=None, waitKey=True):
    if org is not None:
        cv2.imshow(Config.video_src, org)
    if detect is not None:
        cv2.imshow('Detector', detect)
    if model is not None:
        cv2.imshow('Model', model)
    if threshold is not None:
        cv2.imshow('Threshold', threshold)
    if difference is not None:
        cv2.imshow('Difference', difference)

    if waitKey and cv2.waitKey(Config.WAITKEY_MS) == ord('q'):
        return False

    return True


def create_image_window():
    scale = 1.0
    wm, hm = (5, 35)
    nw, nh = (int(Config.VIDEO_WIDTH * scale), int(Config.VIDEO_HEIGHT * scale))
    rnw, rnh = (int(width(Config.ROI) * scale), int(height(Config.ROI) * scale))
    wn = [Config.video_src, 'Detector', 'Model', 'Difference', 'Threshold']

    flags = cv2.WINDOW_NORMAL | cv2.WINDOW_GUI_EXPANDED | cv2.WINDOW_KEEPRATIO

    cv2.namedWindow(wn[0], flags)
    cv2.moveWindow(wn[0], 0, 0)
    cv2.resizeWindow(wn[0], nw, nh)

    cv2.namedWindow(wn[1], flags)
    cv2.moveWindow(wn[1], nw + wm, 0)
    cv2.resizeWindow(wn[1], nw, nh)

    cv2.namedWindow(wn[2], flags)
    cv2.moveWindow(wn[2], 0, nh + hm)
    cv2.resizeWindow(wn[2], rnw, rnh)

    cv2.namedWindow(wn[3], flags)
    cv2.moveWindow(wn[3], rnw + wm, nh + hm)
    cv2.resizeWindow(wn[3], rnw, rnh)

    cv2.namedWindow(wn[4], flags)
    cv2.moveWindow(wn[4], 2 * (rnw + wm), nh + hm)
    cv2.resizeWindow(wn[4], rnw, rnh)

    # now supported
    # cv2.displayStatusBar(wn[0], datetime.datetime.now().isoformat())


class VideoFileMerger():
    def __init__(self, q):
        self.q = q

    def merge(self, q):
        merged_files = list()
        write_files = list()
        while True:
            cf = q.get()

            if len(merged_files) < 1:
                merged_files.append(cf)

            if len(q) > 0:
                nf = q.get()
            else:
                write_files.append(merged_files)
                break

            if nf.end_msec - cf.end_msec < 1000:
                merged_files.append(nf)
            else:
                write_files.append(merged_files)
                merged_files.clear()

        return write_files


class VideoFile():
    def __init__(self, filename, start_msec, end_msec):
        self.filename = filename
        self.start_msec = start_msec
        self.end_msec = end_msec
        self.duration = self.end_msec - self.start_msec


class VideoFrame():
    def __init__(self, frame, objstate=State.ABSENT):
        self.frame = frame
        self.objstate = objstate
        self.createdtime = datetime.datetime.now().timestamp()
        self.iframes = 0
        self.is_ir = None

    # capture thread 가 아닌 다른 쓰레드에서 호출해야함(성능)
    def check_ir(self, frame=None):
        fr = frame if frame is not None else self.frame
        if self.is_ir is None:
            self.is_ir = is_color(fr, Config.MAX_COLOR_SAMPLES)
        return self.is_ir


class ImageModel():
    def __init__(self, pixel=None, feature=None):
        # average value of each pixel (ROI size)
        self.pixel = pixel
        # self.histogram=None
        # histogram extreme pionts
        if pixel is not None and feature is None:
            self.feature = ImageModel.extract_feature(self.pixel)
        else:
            self.feature = feature
        self.alpha = 1

    # frame: gray scale
    @staticmethod
    def extract_feature(frame):
        # hist1, bins=np.histogram(frame.flatten(), 64, [0, 256])
        # wlen=3
        # T=1./wlen
        # w=np.ones(wlen)
        # pseq=np.convolve(w/w.sum(), hist1, mode='same')
        # return pseq
        return None


class StateManager():
    def __init__(self, statemodel, nhistory):
        self.max_nhistory = Config.MAX_NHISTORY
        self.nhistory = nhistory
        self.statemodel = statemodel
        self.cur_state = State.UNKNOWN
        self.prev_state = State.UNKNOWN
        # self.framebuf = []
        self.objstatus=deque()
        self.threadlist = []
        self.nhistorystack = []
        self.pool_sema = BoundedSemaphore(value=1)
        self.last_writing_time = datetime.datetime.today().replace(hour=0, minute=0, second=0)
        self.write_frame_q = deque()
        self.timer = None

    def stop_writing_thread(self):
        exitthreadcount = 0
        for i, t in enumerate(self.threadlist):
            t.join(Config.FPS)
            if not t.is_alive():
                exitthreadcount += 1
                logging.info(f'thread exited : {t.name} {i}/{len(self.threadlist)}')

        if exitthreadcount != len(self.threadlist):
            for i, t in enumerate(self.threadlist):
                t.join(Config.FPS)
                if t.is_alive():
                    logging.info(f'thread : {t.name} is alive')

    def __estimate_seq(self, frame):
        # seq = np.array([float(frame.objstate) for frame in self.framebuf])
        seq = self.objstatus
        wlen = min(13, len(seq))
        T = 1. / wlen
        w = np.ones(wlen)
        pseq = np.convolve(w / w.sum(), seq, mode='same')
        return pseq, wlen, T

    def update_state(self, frame, roi_frame, imagemodel, new_state, color_changed):
        # if len(self.framebuf) < self.nhistory:
        #     self.framebuf.append(VideoFrame(frame, new_state))
        #     # imagemodel=LearnModel.update_model(imagemodel, roi_frame, alpha)
        #     return imagemodel

        output_disp_size = 40

        # if color is changed, write previous frame
        if color_changed:
            # pseq, T=self.__estimate_seq(frame)
            # writing_mode = self.__determine_writing(pseq, T)
            # self.__write_sequence(FrameWriteMode.SINGLE, frame)
            # self.nhistory = self.max_nhistory
            self.objstatus = []

        # while len(self.framebuf) > self.nhistory:
        #     self.framebuf.pop(0)

        # self.framebuf.append(VideoFrame(frame, new_state))
        self.objstatus.append(new_state)
        # nframebuf = len(self.framebuf)
        self.cur_state = new_state

        # seq=np.array([ str(frame.objstate) for frame in self.framebuf ])
        # pseq=np.array(self.statemodel.predict(seq))

        pseq, wlen, T = self.__estimate_seq(frame)

        z = output_disp_size / len(pseq)
        pseqs = interpolation.zoom(pseq, z)

        if wlen > 1:
            self.cur_state = pseq[-1]
            self.prev_state = pseq[-wlen]
        else:
            self.prev_state = self.cur_state

        if self.prev_state >= T or self.cur_state >= T:
            logging.info(f"pseq(scale: 1:{1. / z:.1f})={''.join(map(lambda x: 'O' if x >= T else '-', pseqs))}")

        # np.set_printoptions(precision=2)
        # logging.info(''.join(list(map(lambda x : 'T' if x > T else '.', pseq.tolist()))))

        # ABSENT -> ABSENT
        if self.prev_state < T and self.cur_state < T:
            imageinstance = ImageModel(roi_frame, None)
            imagemodel = LearnModel.update_model(imagemodel, imageinstance)
        # ABSENT -> PRESENT
        elif self.prev_state < T <= self.cur_state:
            self.write_frame_q.append(VideoFrame(frame, new_state))
            self.nhistory += 1
        # PRESENT -> PRESENT
        elif self.prev_state >= T and self.cur_state >= T:
            self.write_frame_q.append(VideoFrame(frame, new_state))
            self.nhistory += 1
        # PRESENT -> ABSENT
        elif self.prev_state >= T > self.cur_state:
            writing_mode = self.__determine_writing(pseq, T)
            self.__write_sequence(writing_mode, frame)
            self.nhistory = self.max_nhistory
            self.objstatus.clear()

        return imagemodel

    def __write_sequence(self, writing_mode, frame):
        if writing_mode == FrameWriteMode.SINGLE:
            logging.info(f'FrameWriteMode.SINGLE')
            if len(self.write_frame_q) < 1:
                logging.info(f'Writing queue is empty')
            else:
                self.timer = threading.Timer(3.0, self.__write_frames,
                                         args=(frame.shape[1], frame.shape[2], self.write_frame_q))
                self.timer.start()
        elif writing_mode == FrameWriteMode.APPEND:
            logging.info(f'FrameWriteMode.APPEND')
            if self.timer.is_alive():
                self.timer.cancel()
            self.timer = threading.Timer(3.0, self.__write_frames,
                                         args=(frame.shape[1], frame.shape[2], self.write_frame_q))
            self.timer.start()
        elif writing_mode == FrameWriteMode.IGNORE:
            logging.info(f'FrameWriteMode.IGNORE')
            self.timer = threading.Timer(3.0, self.write_frame_q.clear)
            self.timer.start()
        else:
            logging.info(f'Unknown FrameWriteMode : {writing_mode}')

    def __write_frames(self, w, h, q):
        th = WriteThread(name=f'WriteThread{len(self.threadlist)}',
                         # args=(q, frame.shape[1], frame.shape[2], 'Object detected', self.pool_sema))
                         args=(q, w, h, 'Object detected', 'data/detected/', self.pool_sema))

        self.threadlist.append(th)
        th.start()

    def __determine_writing(self, pseq, T0):

        classification = {}

        # 2 sec
        T1 = Config.FPS
        T2 = T1 / 8
        # 30 seconds
        T3 = 3

        # framebuf size < T
        nobjstatus = len(self.objstatus)
        if nobjstatus < T1:
            logging.info(f'nobjstatus={nobjstatus}/{T1}')
            return FrameWriteMode.IGNORE
        classification['nobjstatus'] = nobjstatus

        # # object in framebuf lasts < T/2 
        # states=[ f.objstate > 0 for f in self.framebuf ]
        # from back to start, longest segment is larger than T/2
        validobj = 0
        if pseq is not None:
            # print(pseq)
            for i in range(len(pseq) - 2, 0, -1):
                if pseq[i] >= T0:
                    validobj += 1
                else:
                    break

        if validobj < T2:
            logging.info(f'validobj={validobj}/{T2}')
            return FrameWriteMode.IGNORE

        # classification['validobj'] = validobj

        # Last written time < T 
        now = datetime.datetime.now()
        d = now - self.last_writing_time
        self.last_writing_time = now
        if d.seconds < T3:
            logging.info(f'last written time={d.seconds}/{T3}')
            return FrameWriteMode.APPEND
        classification['last written time'] = d.seconds

        logging.info(f'Writing file : {classification}')
        return FrameWriteMode.SINGLE


class MainController():

    def __init__(self, statemodel=None, imagemodel=None):
        self.q = collections.deque([], maxlen=Config.MAX_CAPTURE_BUFFER)
        self.max_timeline = Config.MAX_TIMELINE
        self.max_learn_frames = Config.MAX_LEARN_FRAMES
        self.stop_event = threading.Event()
        self.stop_event.clear()
        self.capture_started_event = threading.Event()
        self.capture_started_event.clear()
        self.capture_thread = None
        self.imagemodel = imagemodel
        self.alpha = 1
        self.statemanager = StateManager(statemodel, self.max_timeline)
        self.prev_frame_ir = None
        self.prev_hist = None
        self.scenemodel = SceneModel()

    def __learn(self, q):
        learner = LearnModel(self.max_learn_frames)
        self.imagemodel = learner.learn(q, q_window_name=Config.video_src)

    def __determine_action(self, frame, roi_frame, detect_result, color_changed):
        self.imagemodel = self.statemanager.update_state(frame, roi_frame, self.imagemodel, detect_result,
                                                         color_changed)

    def run(self):

        # create capture thread
        self.capture_thread = CaptureThread(name='CaptureThread',
                                            args=(self.q, self.stop_event, self.capture_started_event, Config.SKIPFRAME,
                                                  Config.RESIZEFRAME))
        self.capture_thread.start()

        logging.info('Waiting for capture starting')
        self.capture_started_event.wait()

        create_image_window()

        # build model if necessary
        if self.imagemodel == None:
            logging.info('Building model ...')
            self.__learn(self.q)

        logging.info('Detecting ...')

        # fetch frame
        prev_frame = None
        frame = None
        timelogcount = 0

        while True:

            # check if capture thread is running
            if not self.capture_thread.is_alive():
                logging.info(f'{self.capture_thread.name} is not running. Nothing we can do.')
                break

            try:
                vframe = self.q.popleft()

                prev_frame = frame
                frame = vframe.frame

                if not show_frame(org=frame, waitKey=False):
                    logging.info('Stopping capture')
                    self.stop_event.set()
                    continue
                # logging.info(f'ir: {cur_frame_ir}')
            except IndexError:
                time.sleep(0.01)
                continue

            # check if CCD switched to IR or vice versa
            # https://docs.opencv.org/2.4/doc/tutorials/imgproc/histograms/histogram_comparison/histogram_comparison.html
            #    CCD histogram
            #    IR histogram
            #    cv2.calcHist, compareHist
            #    if light condition is changed, rebuild model
            cur_frame_ir = vframe.check_ir()
            color_changed = self.is_color_changed(cur_frame_ir, frame)

            # frame 은 CaptureThread 에서 scale 됨
            curr_hist = self.scenemodel.extract_feature(frame, roi=Config.ROI).transpose()
            scene_changed = self.scenemodel.compare_feature(self.prev_hist, curr_hist) > Config.SCENECHANGE_VALUE

            self.prev_hist = curr_hist
            self.prev_frame_ir = cur_frame_ir

            if color_changed or scene_changed:
                # 새로 learning 을 하므로 이전 정보는 무효
                self.prev_hist = None
                self.prev_frame_ir = None
                logging.info(f'Video {"color" if color_changed else "scene"} changed')
                s = WriteThread.get_current_timestring()
                cv2.imwrite(f'{s}-prev.jpg', prev_frame)
                cv2.imwrite(f'{s}-curr.jpg', frame)
                self.__learn(self.q)

            # detect object
            detector = ObjectDetector(self.imagemodel, frame)
            detect_result, roi_frame = detector.detect(draw_frame=vframe)

            if not show_frame(detect=vframe.frame, model=self.imagemodel.pixel.astype('uint8')):
                self.stop_event.set()
                break
            # plt.subplot(1, 3, 3)
            # cv2.imshow("Detector", frame)
            # cv2.imshow('Model', self.imagemodel.pixel.astype('uint8'))
            # # plt.show()
            # if cv2.waitKey(Config.WAITKEY_MS)==ord('q'):
            #     self.stop_event.set()
            #     break
            self.__determine_action(frame, roi_frame, detect_result, color_changed or scene_changed)

            now = datetime.datetime.now()
            c = datetime.datetime.fromtimestamp(vframe.createdtime)
            d0 = now - c
            if timelogcount % 100 == 0:
                logging.info(f'processed frame :{c.isoformat()}, delay(sec)={d0.seconds}')
                timelogcount = 0
            timelogcount += 1

    def is_color_changed(self, cur_frame_ir, frame):
        if self.prev_frame_ir is None:
            return False

        return self.prev_frame_ir != cur_frame_ir


class SegmentObject():
    def __init__(self, imagemodel, frame):
        self.imagemodel = imagemodel
        self.frame = frame
        self.roi = Config.ROI
        self.convexhulls = None
        self.contours = None

    def segment(self):
        roi_frame, difference, mask = self.__process_image(self.frame)
        self.contours, self.convexhulls = self.__find_contours(mask, lambda x: x + (self.roi[0], self.roi[1]))
        return roi_frame, difference, mask

    @staticmethod
    def get_gray_image(frame):
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # frame_gray=cv2.equalizeHist(frame_gray, 256)
        frame_gray = cv2.equalizeHist(frame_gray)
        frame_gray = cv2.GaussianBlur(frame_gray, (5, 5), 0)

        return frame_gray

    def __process_image(self, frame):
        # cv2.putText(frame, "sequence: ", (0, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, 255)
        roi_frame0 = get_roi_frame(frame)
        roi_frame = SegmentObject.get_gray_image(roi_frame0)
        difference = cv2.absdiff(self.imagemodel.pixel.astype('uint8'), roi_frame)
        # logging.info(f'sum(difference): {np.sum(difference)}')
        # cv2.imshow('difference:absdiff', difference)

        # otsu
        # _, difference_t=cv2.threshold(difference, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        # global threshold
        _, difference_t = cv2.threshold(difference, 40, 255, cv2.THRESH_BINARY)
        # adaptive
        # difference_t=cv2.adaptiveThreshold(difference, 255, cv2.ADAPTIVE_THRESqH_MEAN_C, cv2.THRESH_BINARY,11,2)
        # cv2.imshow('difference:threshold', difference)

        # difference_t=cv2.adaptiveThreshold(difference, 245, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 0)

        kernel = np.ones((7, 7), np.uint8)
        # Morphological Closing -> Morphological Opening
        difference_t = cv2.morphologyEx(difference_t, cv2.MORPH_CLOSE, kernel, iterations=2)
        # difference_t = cv2.morphologyEx(difference_t, cv2.MORPH_OPEN, kernel, iterations=1)

        show_frame(threshold=difference_t, difference=difference)

        lbound = 200
        ubound = 255
        mask = cv2.inRange(difference_t, lbound, ubound)

        return roi_frame, difference_t, mask

    # constraint_function: lambda x : x + (roix, roiy)
    def __find_contours(self, mask, constraint_function):
        contours, _ = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

        # side effect occurs
        # --------------------
        # for i, con in enumerate(contours):
        #     wlen=9
        #     T=1./wlen
        #     w=np.ones(wlen)
        #     con2=con
        #     if len(con) > 9:
        #         x0=con2[:, 0, 0]
        #         con2[:, 0, 0]=np.convolve(w/w.sum(), x0, mode='same')
        #         y0=con2[:, 0, 1]
        #         con2[:, 0, 1]=np.convolve(w/w.sum(), y0, mode='same')
        #         contours[i] = con2

        # cv2.blur(contours, )
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]

        contours = list(map(constraint_function, contours))

        convexhull = []
        # for ct in contours:
        #     ch=cv2.convexHull(ct, False)
        #     convexhull.append(ch)

        return contours, convexhull


class ObjectDetector(SegmentObject):
    def __init__(self, imagemodel, frame):
        super(ObjectDetector, self).__init__(imagemodel, frame)
        self.palette = [(0x64, 0xb5, 0xf6), (0x82, 0xe9, 0xde), (0xff, 0xb7, 0x4d), (0xba, 0x68, 0xc8),
                        (0xde, 0xe7, 0x75)]

    # draw_frame : VideoFrame
    def detect(self, draw_frame=None):
        detected = 0
        classifier = {}

        # blob
        roi_frame, difference, mask = self.segment()
        ious, sumiou, maxiou = self.__find_iou(self.contours)

        # local_feature=np.array(ious[:3])
        # instance_feature=self.__extract_feature(roi_frame)
        # feature_score=self.__match_sequence(self.imagemodel.feature, instance_feature)

        classifier['maxiou'] = maxiou > Config.MAX_IOU
        classifier['sumiou'] = sumiou < 0.3
        # classifier['feature_score']=feature_score < 100
        classifier['feature_score'] = True

        # logging.info(f'maxiou: {maxiou:.2f}:{classifier["maxiou"]}  '+
        #     f'sumiou: {sumiou:.2f}:{classifier["sumiou"]}  '+
        #     f'feature_score: {feature_score:.2f}:{classifier["feature_score"]}')

        if len(list(filter(lambda x: x == True, classifier.values()))) == len(classifier):
            detected = 1
            if draw_frame:
                self.__draw(draw_frame, ious)

        return (detected, roi_frame)

    def __match_sequence(self, x, y):
        sse = ((x - y) ** 2).mean(axis=None)
        return sse

    def __extract_feature(self, frame):
        feature = ImageModel.extract_feature(frame)
        return feature

    def __find_iou(self, contours):
        iou = []
        maxiou = 0
        sumiou = 0
        for i, con in enumerate(contours):
            # if not cv2.isContourConvex(con):
            #     continue
            # logging.info(f'Area{i}: {cv2.contourArea(con)}')

            # METHOD1 bounding box
            # x,y,w,h = cv2.boundingRect(con)
            # v=self.__bb_intersection_over_union([x, y, x+w, y+h], self.roi)+0.001

            # METHOD2 contour area
            #  1. smoothing
            #  2. find area
            # wlen = 9
            # T = 1. / wlen
            # w = np.ones(wlen)
            # x

            con2 = con
            # if len(con) > 9:
            #     x0=con2[:, 0, 0]
            #     con2[:, 0, 0]=np.convolve(w/w.sum(), x0, mode='same')
            #     y0=con2[:, 0, 1]
            #     con2[:, 0, 1]=np.convolve(w/w.sum(), y0, mode='same')

            v = float(cv2.contourArea(con2)) / area(Config.ROI)

            sumiou += v
            maxiou = max(v, maxiou)
            iou.append(v)

        return (iou, sumiou, maxiou)

    def __draw(self, vframe, ious):
        frame = vframe.frame
        maxiou = 0
        padding = 25
        margin = 25
        maxheight = Config.VIDEO_ORG_HEIGHT * 0.5
        for i, (con, iou) in enumerate(zip(self.contours, ious)):
            x0 = Config.ROI[0] + (padding + margin) * i
            y0 = Config.ROI[3] - int(maxheight * iou * 3)
            x1 = x0 + margin
            y1 = Config.ROI[3]
            clr = (255 - i * (255 / len(self.contours)), 0, 0)
            cv2.rectangle(frame, (x0, y0), (x1, y1), clr, -1)
            cv2.putText(frame, f"{iou:.2f}", (x0, y0 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, clr)
            cv2.drawContours(frame, self.contours, i, self.palette[i % len(self.palette)], thickness=cv2.FILLED)
        cv2.rectangle(frame, (Config.ROI[0], Config.ROI[1]), (Config.ROI[2], Config.ROI[3]), (220, 220, 220), 2)
        # cv2.putText(frame, f'msec: {vframe.createdtime / 1000.:.0f}', (20, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
        #             (255, 255, 0))
        # cv2.putText(frame, f'iframes: {vframe.iframes}', (20, 250), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0))

    def __bb_intersection_over_union(self, boxA, boxB):
        # determine the (x, y)-coordinates of the intersection rectangle
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        # compute the area of intersection rectangle
        interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
        # compute the area of both the prediction and ground-truth
        # rectangles
        boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
        boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
        # compute the intersection over union by taking the intersection
        # area and dividing it by the sum of prediction + ground-truth
        # areas - the interesection area
        iou = interArea / float(boxAArea + boxBArea - interArea)
        # return the intersection over union value
        return iou


class CaptureThread(threading.Thread):
    def __init__(self, group=None, target=None, name=None,
                 args=(), kwargs=None, verbose=None):
        super(CaptureThread, self).__init__()
        self.target = target
        self.name = name
        self.q = args[0]
        self.stop_event = args[1]
        self.capture_started_event = args[2]
        self.nskipframe = args[3]
        self.resize = args[4]

    def __read_video_params(self, vcap):

        # Config.FPS=int(vcap.get(cv2.CAP_PROP_FPS))
        Config.VIDEO_ORG_WIDTH = int(vcap.get(cv2.CAP_PROP_FRAME_WIDTH))
        Config.VIDEO_ORG_HEIGHT = int(vcap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        Config.VIDEO_WIDTH = int(Config.VIDEO_ORG_WIDTH * Config.SET_FRAME_SCALE + 0.5)
        Config.VIDEO_HEIGHT = int(Config.VIDEO_ORG_HEIGHT * Config.SET_FRAME_SCALE + 0.5)
        Config.ROI = (Config.SET_FRAME_SCALE * np.array(Config.ROI)).astype(np.int).tolist()

    def run(self):
        logging.info(f'Connecting to device {Config.video_src} ...')
        vcap = cv2.VideoCapture(Config.video_src)
        logging.info('Connected.')

        fps = FPS()

        self.__read_video_params(vcap)

        # not working
        # vcap.set(cv2.CAP_PROP_FPS, 5)
        # fps=vcap.get(cv2.CAP_PROP_FPS)

        self.capture_started_event.set()

        start_msec = datetime.datetime.now().timestamp()
        fps.start()

        frame_skip = -1
        while not self.stop_event.wait(0.001):
            ret, frame = vcap.read()
            if not ret:
                logging.info('Video decoding error occurred. Restarting capturing.')
                vcap.release()
                break
                # vcap = cv2.VideoCapture(Config.video_src)
                # continue

            if frame is None or len(frame) < 1:
                self.capture_started_event.clear()
                break

            frame_skip += 1
            if self.nskipframe > 0 and frame_skip % self.nskipframe != 0:
                continue

            if self.resize:
                frame = imutils.resize(frame, Config.VIDEO_WIDTH)
            vframe = VideoFrame(frame)
            fps.update()

            self.q.append(vframe)
            fps.stop()
            # logging.info(f'input fps : {fps.fps()}')

        vcap.release()

        logging.info("Capturing stopped")


class WriteThread(threading.Thread):
    def __init__(self, group=None, target=None, name=None,
                 args=(), kwargs=None, verbose=None):
        super(WriteThread, self).__init__()
        self.target = target
        self.name = name
        self.q = args[0]
        self.info = args[3]
        self.writedir = args[4]
        self.sema = args[5]

    @staticmethod
    def get_current_timestring():
        s = datetime.datetime.now().replace(microsecond=0).isoformat()
        s = re.sub('[-:]', '', s)
        return s

    def __estimate_fps(self):
        q = self.q
        fps = 1000. / np.mean(np.diff(np.array([v.createdtime * 1000 for v in q])))

        return fps

    def run(self):
        n = len(self.q)

        # logging.info(f'{threading.get_ident()} write_framebuf started : {self.q.qsize()}')
        s = self.get_current_timestring()
        fn = self.writedir + s + ".mp4"

        logging.info(f'Writing {fn}')
        with self.sema:
            fps = self.__estimate_fps()
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            vcap_out = cv2.VideoWriter(fn, fourcc, fps, (Config.VIDEO_WIDTH, Config.VIDEO_HEIGHT))

            end_msec = 0
            start_msec = -1
            while len(self.q) > 0:
                videoframe = self.q.popleft()
                # time.sleep(10)
                vcap_out.write(videoframe.frame)
                # logging.info(f'{threading.get_ident()} : {self.q.qsize()}')

                if start_msec < 0:
                    start_msec = videoframe.createdtime
                if len(self.q) < 1:
                    end_msec = videoframe.createdtime

            vcap_out.release()
            # write_event.clear()
            # logging.info(f'{threading.get_ident()} write_framebuf finished')

            # vf=VideoFile(fn, end_msec)            
            # Config.tg_video_q.put_nowait(fn)
            if Config.send_video:
                send_video(VideoFile(fn, start_msec, end_msec))


class LearnModel():
    def __init__(self, max_learn_frames):
        self.n = max_learn_frames

    def learn_from_file(self, filepath):
        # filepath: 
        Config.video_src = filepath
        if not os.path.exists(filepath):
            return None

        event = threading.Event()
        event.clear()
        q = Queue()
        th = CaptureThread(name="CaptureThread", args=(q, event))
        th.start()

        ret = self.learn(q, q_window_name=filepath)
        event.set()
        th.join(5000)
        logging.info(f'Waiting for {th.name} terminate...')
        if th.is_alive():
            logging.info(f"Cannot stop thread : {th.name}")
        logging.info(f'{th.name} terminated.')

        return ret

    def __get_sample_index(self, width=0, height=0):

        w = Config.VIDEO_WIDTH if width == 0 else width
        h = Config.VIDEO_HEIGHT if height == 0 else height

        nsamples = Config.MAX_COLOR_SAMPLES
        widx = np.random.randint(w, size=nsamples)
        hidx = np.random.randint(h, size=nsamples)

        return widx, hidx

    def __mse(self, cur_frame, prev_frame):
        mse = ((cur_frame - prev_frame) ** 2).mean(axis=None)
        return mse

    def wait_for_stable(self, q, q_window_name=None, maxwait=0):
        prev_frame = None
        is_first = True
        stability = 0
        maxlen = int(Config.FPS * 3) if maxwait == 0 else maxwait
        dq = deque([], maxlen)

        widx, hidx=self.__get_sample_index()

        # nskip=1
        while True:
            try:
                vframe = q.popleft()
                frame = vframe.frame
            except IndexError:
                time.sleep(0.01)
                continue

            if is_first:
                prev_frame = frame
                is_first = False
                continue
            # if nskip % 5 == 0:
            # mse=((frame[hidx, widx[:nsamples], :] - prev_frame[hidx[:nsamples], widx[:nsamples], :]) ** 2).mean(axis=None)
            mse=self.__mse(frame[hidx, widx, :], prev_frame[hidx, widx, :])
            dq.append(mse)
            prev_frame = frame
            # logging.info('fetch frame')
            # logging.info(f'dqsize: {len(dq)}/{maxlen}')
            if len(dq) == maxlen:
                v = np.var(dq)
                logging.info(f'var={v:.2f} stability={stability}')
                if v < 40.0:
                    stability += 1
                else:
                    stability = 0

                if stability > 5:
                    return stability
            # nskip+=1

            if q_window_name:
                show_frame(org=frame, detect=frame)
                # logging.info(f'orgframe={vframe.createdtime}')

    def learn(self, q, q_window_name=None, wait_until_stable=True):

        imagemodel = ImageModel()

        if wait_until_stable:
            logging.info('Waiting for scene stable')
            self.wait_for_stable(q, q_window_name)

        logging.info('learning ... ')
        ref_count = 0
        alpha = 0
        widx, hidx=self.__get_sample_index(width(Config.ROI), height(Config.ROI))

        while ref_count < self.n:
            try:
                vframe = q.popleft()
                frame = vframe.frame
            except IndexError:
                time.sleep(0.1)
                continue
            roi_frame = get_roi_frame(frame)
            frame_gray = SegmentObject.get_gray_image(roi_frame)

            ref_count += 1
            # logging.info(f'ref_count={ref_count}/{self.n}')
            if ref_count == 1:
                imagemodel.pixel = frame_gray
                imagemodel.feature = ImageModel.extract_feature(frame_gray)
                continue

            imageinstance = ImageModel(frame_gray, None)

            if self.__mse(imageinstance.pixel[hidx, widx], imagemodel.pixel[hidx, widx]) > 40:
                logging.info(f'Frame is not stable.')
                ref_count=1
                continue

            imagemodel.alpha = 1. / ref_count
            imagemodel = LearnModel.update_model(imagemodel, imageinstance)
            # imagemodel.pixel=LearnModel.update_mean(imagemodel.pixel, imageinstance.pixel, alpha)
            # imagemodel.hist=LearnModel.update_mean(imagemodel.hist, imageinstance.hist, alpha)
            if q_window_name:
                if not show_frame(frame, frame, imagemodel.pixel.astype('uint8'), None):
                    break
                # # plt.subplot(1, 3, 1)
                # cv2.imshow(q_window_name, frame)
                # # plt.subplot(1, 3, 2)
                # cv2.imshow('Model', imagemodel.pixel.astype('uint8'))
                # # plt.show()
                # cv2.imshow('Detector', frame)
                # if cv2.waitKey(Config.WAITKEY_MS)==ord('q'):
                #     break

        logging.info('done')

        return imagemodel

    @staticmethod
    def update_model(imagemodel, instanceimage):
        imagemodel.pixel = LearnModel.update_mean(imagemodel.pixel, instanceimage.pixel, imagemodel.alpha)
        # imagemodel.hist=LearnModel.update_mean(imagemodel.feature, instanceimage.feature, imagemodel.alpha)
        return imagemodel

    @staticmethod
    def update_mean(imagemodel, new_frame, alpha):
        return alpha * new_frame + (1 - alpha) * imagemodel


# def monitor(context, update):
#     fn=''
#     context.bot.send_video(chat_id=update.effective_chat.id, video=open(fn, 'rb'))


# def monitor(update, context):
#     bio = BytesIO()
#     bio.name='20200522T002004.mp4'

#     fn='detected/20200522T002004.mp4'
#     context.bot.send_video(chat_id=update.effective_chat.id,
#         video=open(fn, 'rb'))


def send_video(v):
    # q=Config.tg_video_q
    # while not q.empty():
    #     v=q.get()
    fn = os.path.basename(v.filename)
    # d=datetime.datetime.fromtimestamp(v.createdtime, datetime.timezone.utc)
    msg = '동작 감지'
    Config.tg_bot.send_message(chat_id=Config.TG_CHAT_ID, text=msg)
    Config.tg_bot.send_video(chat_id=Config.TG_CHAT_ID,
                             caption=fn,
                             video=open(v.filename, 'rb'),
                             timeout=120)


def read_video_source(filename):
    pass


def main(argv):
    logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                        level=logging.INFO)

    if len(argv) > 1:
        addr = argv[1].strip()
        if not os.path.exists(addr):
            logging.error(f'Invalid file: {addr}')
            return
    else:
        # addr='act4.avi'
        with open('address.txt') as fd:
            addr = fd.readline().strip()

    Config.video_src = addr

    # model from file
    # ref='ref1.mp4'
    # learner=LearnReference(Const.MAX_LEARN_FRAMES, Const.ROI)
    # model=learner.learn_from_file(ref)

    # model from live stream
    Config.imagemodel = None

    if os.path.exists('config.json'):
        with open('config.json') as fd:
            cf = json.load(fd)
            Config.TG_CHAT_ID = cf['bot_chatid']
            Config.TG_TOKEN = cf['bot_token']
            Config.tg_bot = Bot(Config.TG_TOKEN)

    if os.path.exists('hmm.json'):
        with open('hmm.json', 'rt') as fd:
            # js=json.load(fd)
            js = fd.read()
            # Config.statemodel=HiddenMarkovModel.from_json(js)
            Config.statemodel = None

    controller = MainController(Config.statemodel, Config.imagemodel)

    controller.run()


if __name__ == '__main__':

    logging.basicConfig(level=logging.INFO,
                        #    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
                        format='%(asctime)s : %(funcName)s : %(message)s')

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    if 'OS' in os.environ and os.environ['OS'] != 'Windows_NT':
        os.environ['DISPLAY'] = ':1'

    # cProfile.run('main(sys.argv)')

    main(sys.argv)

    # updater = Updater(token=config.TG_TOKEN, use_context=True)
    # dispatcher = updater.dispatcher

    # start_handler = CommandHandler('start', start)
    # dispatcher.add_handler(start_handler)

    # monitor_handler = CommandHandler('monitor', monitor)
    # dispatcher.add_handler(monitor_handler)

    # j = updater.job_queue
    # j.run_once(callback_job, context=config)

    # updater.start_polling()
    # updater.idle()
