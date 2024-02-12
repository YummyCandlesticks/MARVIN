import cv2 as cv
import numpy as np
from cvzone.HandTrackingModule import HandDetector

cap = cv.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)
detector = HandDetector(detectionCon=0.8, maxHands=2)
colorR = 255, 0, 255


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
            if 100 < cursor[0] < 300 and 100 < cursor[1] < 300:
                colorR = 0, 255, 0
            else:
                colorR = 255, 0, 255

    cv.rectangle(img, (100, 100), (300, 300), (colorR), cv.FILLED)

    cv.imshow("Image", img)
    cv.waitKey(1)

