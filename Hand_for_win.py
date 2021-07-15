import cv2
import mediapipe as m
import time
import math
from datetime import datetime



class Handdetection():
    def __init__(self, mode=False, maxHands=2, detectioncon=0.5, trackcon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectioncon = detectioncon
        self.trackcon = trackcon
        self.mHands = m.solutions.hands
        self.hands = self.mHands.Hands(self.mode, self.maxHands, self.detectioncon, self.trackcon)
        self.Draw = m.solutions.drawing_utils
        self.tipid = [4, 8, 16, 20]
        self.a = 1
        self.b = 5

    def Find(self, video, draw=True):
        imgRGB = cv2.cvtColor(video, cv2.COLOR_BGR2RGB)
        self.result = self.hands.process(imgRGB)
        self.result.multi_hand_landmarks
        if self.result.multi_hand_landmarks:
            for marks in self.result.multi_hand_landmarks:
                if draw:
                    self.Draw.draw_landmarks(video, marks, self.mHands.HAND_CONNECTIONS)
        return video

    def FindP(self, video, handNo=0, draw=True):
        xlist = []
        ylist = []
        bbox = []
        self.lmlist = []
        if self.result.multi_hand_landmarks:
            myhand = self.result.multi_hand_landmarks[handNo]
            for i, l in enumerate(myhand.landmark):
                h, w, c = video.shape
                cx, cy = int(l.x * w), int(l.y * h)
                xlist.append(cx)
                ylist.append(cy)
                self.lmlist.append([i, cx, cy])
                if draw:
                    cv2.circle(video, (cx, cy), 5, (255, 0, 255), cv2.FILLED)
            xmin, xmax = min(xlist), max(xlist)
            ymin, ymax = min(ylist), max(ylist)
            bbox = xmin, ymin, xmax, ymax
            if draw:
                cv2.rectangle(video, (bbox[0] - 20, bbox[1] - 20), (bbox[2] + 20, bbox[3] + 20), (0, 255, 0), 2)
        return self.lmlist, bbox

    def fing(self):
        fingers = []
        if self.result.multi_hand_landmarks:
            if self.lmlist[self.tipid[0]][1] > self.lmlist[self.tipid[0] - 1][1]:
                fingers.append(1)
            else:
                fingers.append(0)
            for id in range(0, 4):
                if self.lmlist[self.tipid[id]][2] < self.lmlist[self.tipid[id] - 2][2]:
                    fingers.append(1)
                else:
                    fingers.append(0)
        return fingers

    def FindD(self, p1, p2, video, draw=True):
        x1, y1 = self.lmlist[p1][1], self.lmlist[p1][2]
        x2, y2 = self.lmlist[p2][1], self.lmlist[p2][2]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        if draw:
            cv2.circle(video, (x1, y1), 15, (255, 0, 255), cv2.FILLED)
            cv2.circle(video, (x2, y2), 15, (255, 0, 255), cv2.FILLED)
            cv2.line(video, (x1, y1), (x2, y2), (255, 0, 255), 3)
            cv2.circle(video, (cx, cy), 15, (255, 0, 255), cv2.FILLED)
        lenght = math.hypot(x2 - x1, y2 - y1)
        return lenght, video, [x1, y1, x2, y2, cx, cy]


    def open():
      ptime = 0
      detect = Handdetection()
      img = cv2.VideoCapture(0)
      name = str(input("Enter your name to display: "))
      date = datetime.now()
      while True:
        t = time.strftime("%H : %M : %S")
        p, video = img.read()
        video = detect.Find(video)
        lmlist, bbox = detect.FindP(video)
        ctime = time.time()
        fps = 1 / (ctime - ptime)
        ptime = ctime
        cv2.putText(video, str(int(fps)), (18, 70), cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 255), 3)
        cv2.putText(video, name, (18, 425), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 255), 2)
        cv2.putText(video, t, (18, 450), cv2.FONT_HERSHEY_SCRIPT_COMPLEX, 0.7, (255, 255, 255), 1)
        cv2.putText(video, ("%s/%s/%s" % (date.day, date.month, date.year)), (18, 475), cv2.FONT_HERSHEY_SCRIPT_COMPLEX, 0.7, (255, 255, 255), 1)

        cv2.imshow("Hand Detect", video)
        if cv2.waitKey(10) == ord("e"):
            break



