import cv2
import mediapipe as mp
import time
import math
import numpy as np

#Hand Detector is a CUstomized class which utilizes the mediapipe library and performs detection of hands, 
#the landmarks on the hands. it has methods to find the hands, fingers, distance and postion of those fingers.
class handDetector():
    def __init__(self, mode=False, maxHands=2, modelComplexity=1, detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.modelComplex = modelComplexity
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands,self.modelComplex,self.detectionCon, self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils
        self.tipIds = [4, 8, 12, 16, 20, 24, 28, 32, 36, 40]

    #findHands has 2 input values, it takes in the image frame from the videocapture and returns the image with 
    #or without drawing the landmarks ont he detected hands
    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        # print(results.multi_hand_landmarks)   <--- prints the landmark values (XYZ coordinates)
        self.handc = 0
        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:    #<--- draws the landmarks on the hand.
                    self.mpDraw.draw_landmarks(img, handLms,self.mpHands.HAND_CONNECTIONS)
                self.handc += 1 #<--- detects the number of hands in the frame.
        return img
    #findPosition has 2 input values, it takes in the image and then with the previously identified 
    #hand-landmark values and creates a list of landmarks with its ID and X,Y values for both the hand.
    def findPosition(self, img, draw=True):
        self.lmList = []
        self.val = 0
        if self.handc == 1:
            if self.results.multi_hand_landmarks:
                myHand = self.results.multi_hand_landmarks[self.handc-1]
                for id, lm in enumerate(myHand.landmark):
                    #print(id, lm)   <-- prints id and landmark position
                    h, w, c = img.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    #print(id, cx, cy) <-- prints id and landmark position in X,Y
                    self.lmList.append([id, cx, cy])
                    if draw:
                        cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)
                    self.val += 1
                self.val -=1
        elif self.handc == 2: 
            if self.results.multi_hand_landmarks:
                myHand = self.results.multi_hand_landmarks[self.handc-2]
                for id, lm in enumerate(myHand.landmark):
                    #print(id, lm)   <-- prints id and landmark position
                    h, w, c = img.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    #print(id, cx, cy)  <-- prints id and landmark position in X,Y
                    self.lmList.append([id, cx, cy])
                    if draw:
                        cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)
                    self.val += 1
                      
                myHand2 = self.results.multi_hand_landmarks[self.handc-1]
                for id, lm in enumerate(myHand2.landmark):
                    #print(id, lm)  <-- prints id and landmark position
                    h, w, c = img.shape
                    cx2, cy2 = int(lm.x * w), int(lm.y * h)
                    self.lmList.append([id, cx2, cy2])
                    if draw:
                        cv2.circle(img, (cx2, cy2), 5, (255, 0, 255), cv2.FILLED)
                    self.val += 1  #  <-- counter for number of landmark
                self.val -=2 
        return self.lmList

    #counts the number of fingers that are up and returns the a list number fingers based on count of hand 
    #and adds a '1' or '0' based if its up or not. 
    def fingersUp(self):
        fingers = []
        # Thumb
        if self.lmList[self.tipIds[0]][1] > self.lmList[self.tipIds[0] - 1][1]:
            fingers.append(1)
        else:
            fingers.append(0)

        # Counters number of fingers and appends 1,0 if the finger is closed or open.
        for id in range(1, int(self.val/4)):
            if self.lmList[self.tipIds[id]][2] < self.lmList[self.tipIds[id] - 2][2]:
                fingers.append(1)
            else:
                fingers.append(0)

            # totalFingers = fingers.count(1)

        return fingers

    #finds distance between two fingers, takes in the two finger number 
    def findDistance(self, p1, p2, img, draw=True,r=15, t=3):
        x1, y1 = self.lmList[p1][1:]
        x2, y2 = self.lmList[p2][1:]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        if draw:
            cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), t)
            cv2.circle(img, (x1, y1), r, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (x2, y2), r, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (cx, cy), r, (0, 0, 255), cv2.FILLED)
            length = math.hypot(x2 - x1, y2 - y1)

        return length, img, [x1, y1, x2, y2, cx, cy]

def main():
    pass

if __name__ == "__main__":
    main()