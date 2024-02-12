import cv2 as cv
import numpy as np
from cvzone.HandTrackingModule import HandDetector

cap = cv.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)
detector = HandDetector(detectionCon=0.8, maxHands=2)


while True:
    success, img = cap.read()
    img = cv.flip(img, 1)
    hands, img = detector.findHands(img)
    
    if hands: #all the information from the variable is taken from a pre-built dictionary in cvzone, lmList, bbox etc. are all from the package and pre-defined
        hand1 = hands[0]
        lmList1 = hand1["lmList"] 
        bbox1 = hand1["bbox"]
        centerPoint1 = hand1["center"] #gives us the centrepoint for the hand

    if len(hands)==2: #if there are 2 hands (this is just in case although the previous function can detect both hands we still need the data) allows us to get data for both hands basically
        hand2 = hands[1]
        lmList2 = hand2["lmList"] 
        bbox2 = hand2["bbox"]
        centerPoint2 = hand2["center"] 

    cv.rectangle(img, (100, 100), (300, 300), (255, 0, 255), cv.FILLED)

    cv.imshow("Image", img)
    cv.waitKey(1)

