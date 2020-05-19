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


class Const():

    MAX_CAPTURE_BUFFER=20*5
    MAX_TIMELINE=2*20
    MAX_LEARN_FRAMES=100
    ROI=[444, 0, 1055, 690]
    roix=0
    roiy=0
    roiw=0
    roih=0
    MAX_NHISTORY=20*60 # 1 minute
    VIDEO_WIDTH=-1
    VIDEO_HEIGHT=-1

    def __init__(self):
        roix=Const.ROI[0]
        roiy=Const.ROI[1]
        roiw=Const.ROI[2]-roix
        roih=Const.ROI[3]-roiy


class State(Enum):
    UNKNOWN=0
    ABSENT=1
    ENTERED=2
    PRESENT=3
    LEFT=4


def show_image(img):
    title='temp'
    cv2.imshow(title, img)
    cv2.waitKey(0)
    cv2.destroyWindow(title)
    cv2.waitKey(1)


class Roi():
    def __init__(self, roi=[0, 0, 0, 0]):
        self.x0=roi[0]
        self.y0=roi[1]
        self.x1=roi[2]
        self.y1=roi[3]
        self.width=roi[2]-roi[0]
        self.height=roi[3]-roi[1]
    

# class Action(Enum):
#     NOOP=0
#     WRITE_VIDEO=1


class VideoFrame():
    def __init__(self, frame, state=State.UNKNOWN):
        self.frame=frame
        self.state=state


class StateManager():
    def __init__(self, nhistory):
        self.max_nhistory=Const.MAX_NHISTORY
        self.nhistory=nhistory
        self.clear()

    def clear(self):
        self.framebuf=[]
        self.threadlist=[]
        self.nhistorystack=[]
        self.pool_sema = BoundedSemaphore(value=3)

    def stop_writing_thread(self):
        exitthreadcount=0
        for i, t in enumerate(self.threadlist):
            t.join(20)
            if not t.is_alive():
                exitthreadcount+=1
                print(f'thread exited : {t.name} {i}/{len(self.threadlist)}')

        if exitthreadcount != len(self.threadlist):
            for i, t in enumerate(self.threadlist):
                t.join(20)
                if t.is_alive():
                    print(f'thread : {t.name} is alive')


    def update_state(self, frame, detect_result):
        if len(self.framebuf) == 0:
            self.framebuf.append(VideoFrame(frame, State.UNKNOWN))

        while len(self.framebuf) > self.nhistory:
            self.framebuf.pop(0)

        cur_state=self.framebuf[-1].state
        next_state=State.UNKNOWN

        if cur_state==State.UNKNOWN or cur_state==State.ABSENT:
            next_state = State.ENTERED if detect_result else State.ABSENT
        elif cur_state==State.ENTERED:
            next_state = State.PRESENT if detect_result else State.ABSENT
            self.nhistorystack.insert(0, self.nhistory)
        elif cur_state==State.PRESENT:
            next_state = State.PRESENT if detect_result else State.LEFT
            self.nhistory=max(self.nhistory+1, self.max_nhistory)
        elif cur_state==State.LEFT:
            next_state = State.ENTERED if detect_result else State.ABSENT
            self.nhistory=self.nhistorystack.pop(0)
            q=Queue(len(self.framebuf))
            for item in self.framebuf:
                q.put(copy.copy(item))
            th=WriteThread(name=f'WriteThread{len(self.threadlist)}', 
                args=(q, frame.shape[1], frame.shape[2], 'Object detected', self.pool_sema))
            self.threadlist.append(th)
            th.start()

        self.framebuf.append(VideoFrame(frame, next_state))


class MainController():

    def __init__(self, vid_src, model=None):
        self.q=Queue(Const.MAX_CAPTURE_BUFFER)
        self.max_timeline=Const.MAX_TIMELINE
        self.max_learn_frames=Const.MAX_LEARN_FRAMES
        self.roi=Const.ROI
        self.vid_src=vid_src
        self.stop_event=threading.Event()
        self.stop_event.clear()
        self.capture_thread=None
        # ref_frame
        self.model=model
        self.statemanager=StateManager(self.max_timeline)


    # def __add_frame_buf(self, frame):
    #     if self.framebuf.full():
    #         self.framebuf.get()
    #     self.framebuf.put(frame)


    def __learn(self):
        learner=LearnReference(self.max_learn_frames, self.roi)
        self.model=learner.learn(self.q)


    def __determine_action(self, frame, detect_result):
        self.statemanager.update_state(frame, detect_result)


    def run(self):

        # create capture thread
        self.capture_thread=CaptureThread(name='CaptureThread', args=(self.q, self.vid_src, self.stop_event))
        self.capture_thread.start()

        # build model if necessary
        if self.model == None:
            self.__learn()

        # fetch frame
        while True:

            # check if capture thread is running
            if not self.capture_thread.is_alive():
                print(f'{self.capture_thread.name} is not running')
                break

            try:
                frame=self.q.get(block=False)
            except queue.Empty:
                time.sleep(10)
                continue

            # check if CCD switched to IR or vice versa
            # https://docs.opencv.org/2.4/doc/tutorials/imgproc/histograms/histogram_comparison/histogram_comparison.html
            #    CCD histogram
            #    IR histogram
            #    cv2.calcHist, compareHist
            #    if light condition is changed, rebuild model
            if self.is_light_changed(frame):
                self.learn(self.q)

            # self.__add_frame_buf(frame)

            # detect object
            detector=ObjectDetector(self.model, frame, self.roi)
            detect_result=detector.detect(draw_frame=frame)
            cv2.imshow("detector", frame)
            cv2.waitKey(30)
            self.__determine_action(frame, detect_result)


    def is_light_changed(self, frame):
        return False


class SegmentObject():
    def __init__(self, ref_frame, frame):
        self.ref_frame=ref_frame
        self.frame=frame
        self.roi=Const.ROI

    def segment(self):
        frame_gray, difference, mask=self.__process_image(self.frame)
        self.contours=self.__find_contours(mask, lambda x : x + (self.roi[0], self.roi[1]))

    @staticmethod
    def get_gray_image(frame):
        frame_gray=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_gray=cv2.GaussianBlur(frame_gray, (5, 5), 0)

        return frame_gray


    def __process_image(self, frame):

        # cv2.putText(frame, "sequence: ", (0, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, 255)
        x=self.roi[0]
        y=self.roi[1]
        w=self.roi[2]-x
        h=self.roi[3]-y
        frame_gray=SegmentObject.get_gray_image(frame[y:y+h, x:x+w])
        difference=cv2.absdiff(self.ref_frame.astype('uint8'), frame_gray)
        # print(f'sum(difference): {np.sum(difference)}')
        # cv2.imshow('difference:absdiff', difference)

        # _, difference=cv2.threshold(difference, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        _, difference=cv2.threshold(difference, 35, 245, cv2.THRESH_BINARY)
        # cv2.imshow('difference:threshold', difference)

        # difference=cv2.adaptiveThreshold(difference, 245, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 0)

        kernel = np.ones((9,9),np.uint8)
        difference=cv2.dilate(difference, kernel)
        # cv2.imshow('difference:dialate', difference)

        lbound=200
        ubound=255
        mask=cv2.inRange(difference, lbound, ubound)

        return frame_gray, difference, mask


    # constraint_function: lambda x : x + (roix, roiy)
    def __find_contours(self, mask, constraint_function):

        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours=sorted(contours, key=cv2.contourArea, reverse=True)[:5]
        contours=list(map(constraint_function, contours))

        return contours


class ObjectDetector(SegmentObject):
    def __init__(self, ref_frame, frame, roi):
        super(ObjectDetector, self).__init__(ref_frame, frame)
        self.contours=None

    def detect(self, draw_frame=None):
        self.segment()
        ious, sumiou, maxiou=self.__find_iou(self.contours)

        if draw_frame.any():
            self.__draw(draw_frame, ious)

        if maxiou > 0.08:
            return True

        return False

    def __find_iou(self, contours):
        iou=[]
        maxiou=-1
        sumiou=0
        for i, con in enumerate(contours):
            # print(f'Area{i}: {cv2.contourArea(con)}')
            x,y,w,h = cv2.boundingRect(con)
            v=self.__bb_intersection_over_union([x, y, x+w, y+h], self.roi)+0.001
            sumiou+=v
            maxiou=max(v, maxiou)
            iou.append(v)

        return (iou, sumiou, maxiou)


    def __draw(self, frame, ious):
        maxiou=0
        padding=25
        maxheight=1000
        for i, (con, iou) in enumerate(zip(self.contours, ious)):
            x0=self.roi[0]+(padding+50)*i
            y0=self.roi[3]-int(maxheight*iou)
            x1=x0+50
            y1=self.roi[3]
            clr=(255-i*(255/len(self.contours)), 0, 0)
            cv2.rectangle(frame, (x0, y0), (x1, y1), clr, -1)
            cv2.putText(frame, f"{iou:.2f}", (x0, y0-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, clr)
        cv2.drawContours(frame, self.contours, -1, (0, 255, 0), 3)
        cv2.rectangle(frame, (self.roi[0], self.roi[1]), (self.roi[2], self.roi[3]), (220, 220, 220), 2)


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
        self.vid_src=args[1]
        self.stop_event=args[2]

    def run(self):
        print('connecting to device ...')
        vcap_act = cv2.VideoCapture(self.vid_src)
        print('connected ...')

        Const.VIDEO_WIDTH=int(vcap_act.get(3))
        Const.VIDEO_HEIGHT=int(vcap_act.get(4))

        while not self.stop_event.wait(0.001):
            ret, frame = vcap_act.read()
            if not(ret):
                vcap_act.release()
                vcap_act = cv2.VideoCapture(self.vid_src)
                continue

            if frame is None:
                break

            if not self.q.full():
                self.q.put(frame)

            cv2.imshow(self.vid_src, frame)
            if cv2.waitKey(30)==ord('q'):
                break

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
        print(f'{threading.get_ident()} write_framebuf started : {self.q.qsize()}')
        s=datetime.datetime.now().replace(microsecond=0).isoformat()
        s=re.sub('[-:]', '', s)
        fn='detected/'+s+".mp4"    

        with self.sema:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')    
            vcap_out = cv2.VideoWriter(fn, fourcc, 20.0, (Const.VIDEO_WIDTH, Const.VIDEO_HEIGHT))

            while not self.q.empty():
                videoframe=self.q.get()
                # time.sleep(10)
                vcap_out.write(videoframe.frame)
                print(f'{threading.get_ident()} : {self.q.qsize()}')
            vcap_out.release()
            # write_event.clear()
            print(f'{threading.get_ident()} write_framebuf finished')


class LearnReference():
    def __init__(self, max_learn_frames, roi):
        self.n=max_learn_frames
        self.roi=roi
        self.roix=self.roi[0]
        self.roiy=self.roi[1]
        self.roiw=self.roi[2]-self.roix
        self.roih=self.roi[3]-self.roiy


    def learn_from_file(self, filepath):
        # filepath: 
        vid_src=filepath
        if not os.path.exists(filepath):
            return None

        # print('connecting to device ...')
        # vcap_ref = cv2.VideoCapture(vid_src)
        # width=int(vcap_ref.get(3))
        # height=int(vcap_ref.get(4))
        # print('connected ...')

        event=threading.Event()
        event.clear()
        q=Queue()
        th=CaptureThread(name="CaptureThread", args=(q, vid_src, event))
        th.start()

        ret=self.learn(q)
        event.set()
        th.join(5000)
        print(f'Waiting for {th.name} terminate...')
        if th.is_alive():
            print(f"Cannot stop thread : {th.name}")
        print(f'{th.name} terminated.')

        return ret


    def learn(self, q):
        ref_count=0
        while ref_count < self.n:
            frame=q.get()
            
            x=self.roi[0]
            y=self.roi[1]
            w=self.roi[2]-x
            h=self.roi[3]-y
            frame_gray=SegmentObject.get_gray_image(frame[y:y+h, x:x+w])

            ref_count+=1
            print(f'ref_count={ref_count}/{self.n}')
            if ref_count == 1:
                ref_frame=frame_gray
                continue
            
            ref_frame=LearnReference.update_ref_frame(ref_frame, frame_gray, 1./ref_count)

        return ref_frame


    @staticmethod
    def update_ref_frame(ref_frame, new_frame, alpha):
        return alpha*new_frame+(1-alpha)*ref_frame


if __name__ == '__main__':

    # addr='act1.avi'
    with open('address.txt') as fd:
        addr=fd.readline().strip()

    # model from file
    # ref='ref1.mp4'
    # learner=LearnReference(Const.MAX_LEARN_FRAMES, Const.ROI)
    # model=learner.learn_from_file(ref)

    # model from live stream
    model=None
    controller=MainController(addr, model)

    controller.run()

    