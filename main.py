import cv2 as cv
import numpy as np
from cvzone.HandTrackingModule import HandDetector

cap = cv.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)
detector = HandDetector(detectionCon=0.8, maxHands=2)
colorR = 255, 0, 255

# instead of defining initial and final points for the rectangles we will define the centre points
cx, cy, w, h = 100, 100, 200, 200 #instead we are giving the cntrepoints and a fixed width and height

while True:
    success, img = cap.read()
    img = cv.flip(img, 1)
    hands, img = detector.findHands(img)
    
    if hands: #all the information from the variable is taken from a pre-built dictionary in cvzone, lmList, bbox etc. are all from the package and pre-defined
        hand = hands[0]
        lmList = hand["lmList"]
        bbox = hand["bbox"]
        centerPoint = hand["center"] #gives us the centrepoint for the hand

        if lmList:
            cursor = lmList[8]
            if cx-w//2 < cursor[0] < cx+w//2 and \
                cy-h//2 < cursor[1] < cy+h//2:
                colorR = 0, 255, 0
                cx, cy = cursor[0], cursor[1]
            else:
                colorR = 255, 0, 255

    

    cv.rectangle(img, (cx-w//2, cy-h//2), (cx+w//2, cy+h//2), (colorR), cv.FILLED) # img, (x1, y1), (x2, y2) initial and final positions

    cv.imshow("Image", img)
    cv.waitKey(1)

