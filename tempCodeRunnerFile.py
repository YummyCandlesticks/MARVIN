import cv2 as cv
import numpy as np
import mediapipe as mp
import cvzone
from cvzone.HandTrackingModule import HandDetector

cap = cv.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)
detector = HandDetector(detectionCon=0.8)


while True:
    success, img = cap.read()
    img = detector.findHands(img)
    imList, _ = findPosition(img)
    cv.imshow("Image", img)
    cv.waitKey(1)