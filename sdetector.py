import cv2
import numpy as np
import threading
import queue
import datetime
import re
import os
from enum import Enum
from queue import Queue
from threading import BoundedSemaphore
import time
import copy
from color_model import get_pixel_statistics
from collections import deque
from matplotlib import pyplot as plt
import collections
from telegram import Bot
from telegram.ext import Updater
import logging
from telegram.ext import CommandHandler
import json
from pomegranate import *
import sys
import random


class Config():

    FPS=10
    WRITE_FPS=10
    # 20 frames/sec * 30sec
    MAX_CAPTURE_BUFFER=FPS*60
    MAX_TIMELINE=5*FPS
    MAX_LEARN_FRAMES=FPS*3
    MAX_NHISTORY=FPS*5
     # 1 minute
    ROI=[444, 0, 1055, 720]
    VIDEO_WIDTH=-1
    VIDEO_HEIGHT=-1
    MAX_COLOR_SAMPLES=100
    WAITKEY_MS=50
    TG_CHAT_ID=None
    TG_TOKEN=None
    MAX_IOU=0.02
    send_video=False
    statemodel=None
    imagemodel=None
    video_src=None


    tg_video_q=Queue()
    tg_bot=None


class State():
    UNKNOWN=0
    ABSENT=1
    ENTERED=2
    PRESENT=3
    LEFT=4


def width(roi):
    return roi[2]-roi[0]

def height(roi):
    return roi[3]-roi[1]

def area(roi):
    return width(roi)*height(roi)

def show_image(img):
    title='temp'
    cv2.imshow(title, img)
    cv2.waitKey(Config.WAITKEY_MS)
    cv2.destroyWindow(title)
    cv2.waitKey(Config.WAITKEY_MS)


def get_roi_frame(frame):

    x=Config.ROI[0]
    y=Config.ROI[1]
    w=Config.ROI[2]-x
    h=Config.ROI[3]-y

    return frame[y:y+h, x:x+w]


def show_frame(org=None, detect=None, model=None, difference=None, waitKey=True):

    if org is not None:
        cv2.imshow(Config.video_src, org)
    if detect is not None:
        cv2.imshow('Detector', detect)
    if model is not None:
        cv2.imshow('Model', model)
    if difference is not None:
        cv2.imshow('Difference', difference)

    if waitKey and cv2.waitKey(Config.WAITKEY_MS)==ord('q'):
        return False

    return True


def create_image_window():
    scale=0.4
    margin=50
    nw, nh=(int(Config.VIDEO_WIDTH*scale), int(Config.VIDEO_HEIGHT*scale))
    rnw, rnh=(int(width(Config.ROI)*scale), int(height(Config.ROI)*scale))
    wn=[Config.video_src, 'Detector', 'Model', 'Difference']
    cv2.namedWindow(wn[0], cv2.WINDOW_NORMAL)
    cv2.moveWindow(wn[0], 0, 0)
    cv2.resizeWindow(wn[0], nw, nh)

    cv2.namedWindow(wn[1], cv2.WINDOW_NORMAL)
    cv2.moveWindow(wn[1], nw+margin, 0)
    cv2.resizeWindow(wn[1], nw, nh)

    cv2.namedWindow(wn[2], cv2.WINDOW_NORMAL)
    cv2.moveWindow(wn[2], 0, nh+margin)
    cv2.resizeWindow(wn[2], rnw, rnh)

    cv2.namedWindow(wn[3], cv2.WINDOW_NORMAL)
    cv2.moveWindow(wn[3], rnw+margin, rnh+margin)
    cv2.resizeWindow(wn[3], rnw, rnh)

class VideoFileMerger():
    def __init__(self, q):
        self.q=q

    def merge(self, q):
        merged_files=list()
        write_files=list()
        while True:
            cf=q.get()

            if len(merged_files) < 1:
                merged_files.append(cf)

            if len(q) > 0:
                nf=q.get()
            else:
                write_files.append(merged_files)
                break
            
            if nf.end_msec-cf.end_msec < 1000:
                merged_files.append(nf)
            else:
                write_files.append(merged_files)
                merged_files.clear()

        return write_files


class VideoFile():
    def __init__(self, filename, start_msec, end_msec):

        self.filename=filename
        self.start_msec=start_msec
        self.end_msec=end_msec
        self.duration=self.end_msec-self.start_msec


class VideoFrame():
    def __init__(self, frame, objstate=State.ABSENT):
        self.frame=frame
        self.objstate=objstate
        self.msec=0
        self.iframes=0
        self.is_ir=None

    # capture thread 가 아닌 다른 쓰레드에서 호출해야함(성능)
    def check_ir(self, frame=None):
        fr=frame if frame is not None else self.frame
        if self.is_ir is None:
            stat=get_pixel_statistics(fr, Config.MAX_COLOR_SAMPLES)
            r=float(len(list(filter(lambda x : x == 0, stat))))/Config.MAX_COLOR_SAMPLES
            self.is_ir = r > 0.95
        return self.is_ir


class ImageModel():
    def __init__(self, pixel=None, feature=None):
        # average value of each pixel (ROI size)
        self.pixel=pixel
        # self.histogram=None
        # histogram extreme pionts
        if pixel is not None and feature is None:
            self.feature=ImageModel.extract_feature(self.pixel)
        else:
            self.feature=feature
        self.alpha=1

    # frame: gray scale
    @staticmethod
    def extract_feature(frame):
        hist1, bins=np.histogram(frame.flatten(), 64, [0, 256])
        wlen=3
        T=1./wlen
        w=np.ones(wlen)
        pseq=np.convolve(w/w.sum(), hist1, mode='same')
        return pseq


class StateManager():
    def __init__(self, statemodel, nhistory):
        self.max_nhistory=Config.MAX_NHISTORY
        self.nhistory=nhistory
        self.clear()
        self.statemodel=statemodel

    def clear(self):
        self.cur_state=State.UNKNOWN
        self.prev_state=State.UNKNOWN
        self.framebuf=[]
        self.threadlist=[]
        self.nhistorystack=[]
        self.pool_sema = BoundedSemaphore(value=1)

    def stop_writing_thread(self):
        exitthreadcount=0
        for i, t in enumerate(self.threadlist):
            t.join(Config.FPS)
            if not t.is_alive():
                exitthreadcount+=1
                print(f'thread exited : {t.name} {i}/{len(self.threadlist)}')

        if exitthreadcount != len(self.threadlist):
            for i, t in enumerate(self.threadlist):
                t.join(Config.FPS)
                if t.is_alive():
                    print(f'thread : {t.name} is alive')


    def update_state(self, frame, roi_frame, imagemodel, new_state):
        # if len(self.framebuf) < self.nhistory:
        #     self.framebuf.append(VideoFrame(frame, new_state))
        #     # imagemodel=LearnModel.update_model(imagemodel, roi_frame, alpha)
        #     return imagemodel

        while len(self.framebuf) > self.nhistory:
            self.framebuf.pop(0)

        self.framebuf.append(VideoFrame(frame, new_state))

        # seq=np.array([ str(frame.objstate) for frame in self.framebuf ])
        # pseq=np.array(self.statemodel.predict(seq))

        seq=np.array([ float(frame.objstate) for frame in self.framebuf ])
        wlen=3
        T=1./wlen
        w=np.ones(wlen)
        pseq=np.convolve(w/w.sum(), seq, mode='same')

        self.cur_state=pseq[-1]
        self.prev_state=pseq[-wlen]

        # np.set_printoptions(precision=2)
        # print(''.join(list(map(lambda x : 'T' if x > T else '.', pseq.tolist()))))

        # ABSENT -> ABSENT
        if self.prev_state < T and self.cur_state < T:
            imageinstance=ImageModel(roi_frame, None)
            imagemodel=LearnModel.update_model(imagemodel, imageinstance)
        # ABSENT -> PRESENT
        elif self.prev_state < T and self.cur_state >= T:
            self.nhistory+=1
        # PRESENT -> PRESENT
        elif self.prev_state >= T and self.cur_state >= T:
            self.nhistory+=1
        # PRESENT -> ABSENT
        elif self.prev_state >= T and self.cur_state < T:
            # if len(self.framebuf) > 50:

            q=Queue(len(self.framebuf))
            for item in self.framebuf:
                q.put(copy.copy(item))
            th=WriteThread(name=f'WriteThread{len(self.threadlist)}', 
                args=(q, frame.shape[1], frame.shape[2], 'Object detected', self.pool_sema))
            self.threadlist.append(th)
            th.start()
            
            self.nhistory=self.max_nhistory
            self.framebuf=[]

        return imagemodel

class MainController():

    def __init__(self, statemodel=None, imagemodel=None):
        self.q=collections.deque([], maxlen=Config.MAX_CAPTURE_BUFFER)
        self.max_timeline=Config.MAX_TIMELINE
        self.max_learn_frames=Config.MAX_LEARN_FRAMES
        self.stop_event=threading.Event()
        self.stop_event.clear()
        self.capture_started_event=threading.Event()
        self.capture_started_event.clear()
        self.capture_thread=None
        self.imagemodel=imagemodel
        self.alpha=1
        self.statemanager=StateManager(statemodel, self.max_timeline)
        self.prev_frame_ir=None


        # dim=[(nw, nh), 
        #     (nw, nh), 
        #     (int(width(Config.ROI)*scale), int(height(Config.ROI)*scale)),
        #     (int(width(Config.ROI)*scale), int(height(Config.ROI)*scale))
        #     ]

        # off=[(0, 0)]
        # for i in range(len(wn)-1):
        #     v=(dim[0][0]+margin, 0), (dim[0][0]+dim[1][0]+2*margin, 0)]
        #     off.append(v)
            
        # for i, w in enumerate(wn):
        #     cv2.namedWindow(w, cv2.WINDOW_NORMAL)
        #     cv2.resizeWindow(w, dim[i][0], dim[i][1])

        # cv2.moveWindow(w, off[i][0], off[i][1])


    def __learn(self, q):
        learner=LearnModel(self.max_learn_frames)
        self.imagemodel=learner.learn(q, q_window_name=Config.video_src)


    def __determine_action(self, frame, roi_frame, detect_result):
        self.imagemodel=self.statemanager.update_state(frame, roi_frame, self.imagemodel, detect_result)


    def run(self):

        # create capture thread
        self.capture_thread=CaptureThread(name='CaptureThread', 
            args=(self.q, self.stop_event, self.capture_started_event))
        self.capture_thread.start()

        print('Waiting for capture starting')
        self.capture_started_event.wait()

        create_image_window()

        # build model if necessary
        if self.imagemodel == None:
            print('Building model ...')
            self.__learn(self.q)

        # fetch frame
        prev_frame=None
        frame=None

        while True:

            # check if capture thread is running
            if not self.capture_thread.is_alive():
                print(f'{self.capture_thread.name} is not running')
                break

            try:
                vframe=self.q.popleft()

                prev_frame=frame
                frame=vframe.frame

                cur_frame_ir=vframe.check_ir()

                # plt.subplot(1, 3, 1)
                # cv2.imshow(Config.video_src, frame)
                # plt.show()

                # if cv2.waitKey(Config.WAITKEY_MS)==ord('q'):
                if not show_frame(frame):
                    print('Stopping capture')
                    self.stop_event.set()
                    continue
                # print(f'ir: {cur_frame_ir}')
            except IndexError:
                time.sleep(0.01)
                continue

            # check if CCD switched to IR or vice versa
            # https://docs.opencv.org/2.4/doc/tutorials/imgproc/histograms/histogram_comparison/histogram_comparison.html
            #    CCD histogram
            #    IR histogram
            #    cv2.calcHist, compareHist
            #    if light condition is changed, rebuild model
            if self.prev_frame_ir != None and self.prev_frame_ir != cur_frame_ir:
                print('Video color changed')
                # cv2.imwrite('p.jpg', prev_frame)
                # cv2.imwrite('c.jpg', frame)
                self.__learn(self.q)

            self.prev_frame_ir=cur_frame_ir

            # detect object
            detector=ObjectDetector(self.imagemodel, frame)
            detect_result, roi_frame=detector.detect(draw_frame=vframe)

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
            self.__determine_action(frame, roi_frame, detect_result)


    def is_color_changed(self, frame):
        return self.prev_frame_ir != frame.is_ir


class SegmentObject():
    def __init__(self, imagemodel, frame):
        self.imagemodel=imagemodel
        self.frame=frame
        self.roi=Config.ROI
        self.convexhulls=None
        self.contours=None

    def segment(self):
        roi_frame, difference, mask=self.__process_image(self.frame)
        self.contours, self.convexhulls=self.__find_contours(mask, lambda x : x + (self.roi[0], self.roi[1]))
        return roi_frame, difference, mask

    @staticmethod
    def get_gray_image(frame):
        frame_gray=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # frame_gray=cv2.equalizeHist(frame_gray, 256)
        frame_gray=cv2.GaussianBlur(frame_gray, (5, 5), 0)

        return frame_gray


    def __process_image(self, frame):

        # cv2.putText(frame, "sequence: ", (0, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, 255)
        roi_frame0=get_roi_frame(frame)
        roi_frame=SegmentObject.get_gray_image(roi_frame0)
        difference=cv2.absdiff(self.imagemodel.pixel.astype('uint8'), roi_frame)
        # print(f'sum(difference): {np.sum(difference)}')
        # cv2.imshow('difference:absdiff', difference)

        # _, difference=cv2.threshold(difference, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        _, difference=cv2.threshold(difference, 35, 255, cv2.THRESH_BINARY)
        # cv2.imshow('difference:threshold', difference)

        # difference=cv2.adaptiveThreshold(difference, 245, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 0)

        kernel = np.ones((7,7),np.uint8)
        difference=cv2.dilate(difference, kernel)
        kernel = np.ones((5,5),np.uint8)
        difference=cv2.erode(difference, kernel)
        
        show_frame(difference=cv2.imshow('Difference', difference))

        lbound=200
        ubound=255
        mask=cv2.inRange(difference, lbound, ubound)

        return roi_frame, difference, mask


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
        contours=sorted(contours, key=cv2.contourArea, reverse=True)[:5]

        contours=list(map(constraint_function, contours))

        convexhull=[]
        # for ct in contours:
        #     ch=cv2.convexHull(ct, False)
        #     convexhull.append(ch)

        return contours, convexhull


class ObjectDetector(SegmentObject):
    def __init__(self, imagemodel, frame):
        super(ObjectDetector, self).__init__(imagemodel, frame)


    # draw_frame : VideoFrame
    def detect(self, draw_frame=None):
        detected=0
        classifier={}

        # blob
        roi_frame, difference, mask=self.segment()
        ious, sumiou, maxiou=self.__find_iou(self.contours)

        local_feature=np.array(ious[:3])
        instance_feature=self.__extract_feature(roi_frame)
        feature_score=self.__match_sequence(self.imagemodel.feature, instance_feature)

        classifier['maxiou']=maxiou > Config.MAX_IOU
        classifier['sumiou']=sumiou < 0.3
        # classifier['feature_score']=feature_score < 100
        classifier['feature_score']=True

        print(f'maxiou: {maxiou:.2f}:{classifier["maxiou"]}  '+
            f'sumiou: {sumiou:.2f}:{classifier["sumiou"]}  '+
            f'feature_score: {feature_score:.2f}:{classifier["feature_score"]}')

        if len(list(filter(lambda x : x == True, classifier.values()))) == len(classifier):
            detected=1
            if draw_frame:
                self.__draw(draw_frame, ious)

        return (detected, roi_frame)


    def __match_sequence(self, x, y):
        sse=((x - y) ** 2).mean(axis=None)
        return sse

    def __extract_feature(self, frame):
        feature=ImageModel.extract_feature(frame)
        return feature

    def __find_iou(self, contours):
        iou=[]
        maxiou=-1
        sumiou=0
        for i, con in enumerate(contours):
            # if not cv2.isContourConvex(con):
            #     continue
            # print(f'Area{i}: {cv2.contourArea(con)}')

            # METHOD1 bounding box
            # x,y,w,h = cv2.boundingRect(con)
            # v=self.__bb_intersection_over_union([x, y, x+w, y+h], self.roi)+0.001

            # METHOD2 contour area
            #  1. smoothing
            #  2. find area
            wlen=9
            T=1./wlen
            w=np.ones(wlen)
            # x

            con2=con
            # if len(con) > 9:
            #     x0=con2[:, 0, 0]
            #     con2[:, 0, 0]=np.convolve(w/w.sum(), x0, mode='same')
            #     y0=con2[:, 0, 1]
            #     con2[:, 0, 1]=np.convolve(w/w.sum(), y0, mode='same')

            v=float(cv2.contourArea(con2))/area(Config.ROI)

            sumiou+=v
            maxiou=max(v, maxiou)
            iou.append(v)

        return (iou, sumiou, maxiou)


    def __draw(self, vframe, ious):
        frame=vframe.frame
        maxiou=0
        padding=25
        maxheight=1000
        for i, (con, iou) in enumerate(zip(self.contours, ious)):
            x0=Config.ROI[0]+(padding+50)*i
            y0=Config.ROI[3]-int(maxheight*iou)
            x1=x0+50
            y1=Config.ROI[3]
            clr=(255-i*(255/len(self.contours)), 0, 0)
            cv2.rectangle(frame, (x0, y0), (x1, y1), clr, -1)
            cv2.putText(frame, f"{iou:.2f}", (x0, y0-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, clr)
        cv2.drawContours(frame, self.contours, -1, (0, 255, 0), thickness=cv2.FILLED)
        cv2.rectangle(frame, (Config.ROI[0], Config.ROI[1]), (Config.ROI[2], Config.ROI[3]), (220, 220, 220), 2)
        cv2.putText(frame, f'msec: {vframe.msec/1000.:.0f}', (20, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0))
        cv2.putText(frame, f'iframes: {vframe.iframes}', (20, 250), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0))

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
        self.target=target
        self.name=name
        self.q=args[0]
        self.stop_event=args[1]
        self.capture_started_event=args[2]


    def run(self):
        print('connecting to device ...')
        vcap_act = cv2.VideoCapture(Config.video_src)
        print('connected ...')

        Config.VIDEO_WIDTH=int(vcap_act.get(3))
        Config.VIDEO_HEIGHT=int(vcap_act.get(4))

        self.capture_started_event.set()

        start_msec=datetime.datetime.now().timestamp()

        # frame_skipped=False
        while not self.stop_event.wait(0.001):
            ret, frame = vcap_act.read()
            if not(ret):
                vcap_act.release()
                vcap_act = cv2.VideoCapture(Config.video_src)
                continue

            if frame is None:
                self.capture_started_event.clear()
                break

            # if not frame_skipped:
            #     frame_skipped=True
            #     continue

            # if len(self.q)==self.q.maxlen:
            #     print('Capture queue is full. dropping previous frame')
                # self.q.get()
                # time.sleep(0.1)

            # s=datetime.datetime.now().isoformat()
            # cv2.putText(frame, f"{s}", (0, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0))
            vframe=VideoFrame(frame)

            vframe.iframes=vcap_act.get(cv2.CAP_PROP_POS_FRAMES)
            vframe.msec=start_msec+vcap_act.get(cv2.CAP_PROP_POS_MSEC)

            if vframe.msec > 0:
                fps = vframe.iframes / (vframe.msec / 1000.)
                # print(f'fps: {fps}')
            
            self.q.append(vframe)

        vcap_act.release()

        print("capture stopped")


class WriteThread(threading.Thread):
    def __init__(self, group=None, target=None, name=None,
                args=(), kwargs=None, verbose=None):
        super(WriteThread, self).__init__()
        self.target=target
        self.name=name
        self.q=args[0]
        self.info=args[3]
        self.sema=args[4]

    def run(self):
        # print(f'{threading.get_ident()} write_framebuf started : {self.q.qsize()}')
        s=datetime.datetime.now().replace(microsecond=0).isoformat()
        s=re.sub('[-:]', '', s)
        fn='detected/'+s+".mp4"    

        with self.sema:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')    
            vcap_out = cv2.VideoWriter(fn, fourcc, Config.WRITE_FPS, (Config.VIDEO_WIDTH, Config.VIDEO_HEIGHT))

            end_msec=0
            start_msec=-1
            while not self.q.empty():
                videoframe=self.q.get()
                # time.sleep(10)
                vcap_out.write(videoframe.frame)
                # print(f'{threading.get_ident()} : {self.q.qsize()}')

                if start_msec < 0:
                    start_msec=videoframe.msec
                if self.q.empty():
                    end_msec=videoframe.msec

            vcap_out.release()
            # write_event.clear()
            # print(f'{threading.get_ident()} write_framebuf finished')

            # vf=VideoFile(fn, end_msec)            
            # Config.tg_video_q.put_nowait(fn)
            if Config.send_video:
                send_video(VideoFile(fn, start_msec, end_msec) )


class LearnModel():
    def __init__(self, max_learn_frames):
        self.n=max_learn_frames


    def learn_from_file(self, filepath):
        # filepath: 
        Config.video_src=filepath
        if not os.path.exists(filepath):
            return None

        event=threading.Event()
        event.clear()
        q=Queue()
        th=CaptureThread(name="CaptureThread", args=(q, event))
        th.start()

        ret=self.learn(q, q_window_name=filepath)
        event.set()
        th.join(5000)
        print(f'Waiting for {th.name} terminate...')
        if th.is_alive():
            print(f"Cannot stop thread : {th.name}")
        print(f'{th.name} terminated.')

        return ret


    def wait_for_stable(self, q, q_window_name=None):
        prev_frame=None
        is_first=True
        maxlen=50
        dq=deque([], maxlen)
        
        w=Config.VIDEO_WIDTH
        h=Config.VIDEO_HEIGHT

        nsamples=100
        widx=np.random.randint(w, size=nsamples)
        hidx=np.random.randint(h, size=nsamples)

        # nskip=1
        while True:
            try:
                vframe=q.popleft()
                frame=vframe.frame
                if q_window_name:
                    show_frame(detect=frame)
            except IndexError:
                # time.sleep(0.01)
                continue

            if is_first:
                prev_frame=frame
                is_first=False
                continue
            # if nskip % 5 == 0:
            dq.append(((frame[hidx[:nsamples], widx[:nsamples], :] - prev_frame[hidx[:nsamples], widx[:nsamples], :]) ** 2).mean(axis=None))
            prev_frame=frame
            # print('fetch frame')
            if len(dq) == maxlen:
                v=np.var(dq)
                print(f'wait_for_stable : {v}')
                if v < 40.0:
                    return
            # nskip+=1


    def learn(self, q, q_window_name=None, wait_until_stable=True):

        imagemodel=ImageModel()

        if wait_until_stable:
            print('Waiting for scene stable')
            self.wait_for_stable(q, q_window_name)

        print('learning ... ', end='')
        ref_count=0
        alpha=0
        while ref_count < self.n:
            try:
                vframe=q.popleft()
                frame=vframe.frame
            except IndexError:
                time.sleep(0.1)
                continue
            roi_frame=get_roi_frame(frame)
            frame_gray=SegmentObject.get_gray_image(roi_frame)

            ref_count+=1
            # print(f'ref_count={ref_count}/{self.n}')
            if ref_count == 1:
                imagemodel.pixel=frame_gray
                imagemodel.feature=ImageModel.extract_feature(frame_gray)
                continue

            imageinstance=ImageModel(frame_gray, None)

            imagemodel.alpha=1./ref_count
            imagemodel=LearnModel.update_model(imagemodel, imageinstance)
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

        print('done')

        return imagemodel


    @staticmethod
    def update_model(imagemodel, instanceimage):
        imagemodel.pixel=LearnModel.update_mean(imagemodel.pixel, instanceimage.pixel, imagemodel.alpha)
        imagemodel.hist=LearnModel.update_mean(imagemodel.feature, instanceimage.feature, imagemodel.alpha)
        return imagemodel

    @staticmethod
    def update_mean(imagemodel, new_frame, alpha):
        return alpha*new_frame+(1-alpha)*imagemodel

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
    Config.tg_bot.send_video(chat_id=Config.TG_CHAT_ID, 
        caption=os.path.basename(v.filename),
        video=open(v.filename, 'rb'),
        timeout=120)


def main(argv):

    logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                     level=logging.INFO)

    # addr='act4.avi'
    with open('address.txt') as fd:
        addr=fd.readline().strip()
    Config.video_src=addr

    # model from file
    # ref='ref1.mp4'
    # learner=LearnReference(Const.MAX_LEARN_FRAMES, Const.ROI)
    # model=learner.learn_from_file(ref)

    # model from live stream
    Config.imagemodel=None

    if os.path.exists('config.json'):
        with open('config.json') as fd:
            cf=json.load(fd)
            Config.TG_CHAT_ID=cf['bot_chatid']
            Config.TG_TOKEN=cf['bot_token']
            Config.tg_bot=Bot(Config.TG_TOKEN)

    if os.path.exists('hmm.json'):
        with open('hmm.json', 'rt') as fd:
            # js=json.load(fd)
            js=fd.read()
            Config.statemodel=HiddenMarkovModel.from_json(js)

    controller=MainController(Config.statemodel, Config.imagemodel)

    controller.run()


if __name__ == '__main__':


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
