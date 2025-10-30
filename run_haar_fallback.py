#!/usr/bin/env python3
"""
Quick Haar Cascade fallback to test webcam and demonstrate face detection
Usage:
  .\.venv\Scripts\python.exe run_haar_fallback.py --input 0
"""
import argparse
import cv2

parser = argparse.ArgumentParser()
parser.add_argument('--input', '-i', default='0', help='webcam index or video URL')
args = parser.parse_args()

src = args.input
if str(src).isdigit():
    cap = cv2.VideoCapture(int(src), cv2.CAP_DSHOW)
else:
    cap = cv2.VideoCapture(src)

if not cap.isOpened():
    print('Unable to open capture', src)
    exit(1)

cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
if not cv2.os.path.exists(cascade_path):
    print('Haar cascade not found at', cascade_path)
    exit(1)

detector = cv2.CascadeClassifier(cascade_path)
print('Starting Haar fallback. Press ESC or q to quit.')
while True:
    ok, frame = cap.read()
    if not ok:
        print('Frame read failed')
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30,30))
    for (x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
    cv2.imshow('Haar fallback', frame)
    k = cv2.waitKey(1) & 0xFF
    if k == 27 or k == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
print('Exited')
