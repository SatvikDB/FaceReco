#!/usr/bin/env python3
"""
Quick Haar Cascade fallback to test webcam and demonstrate face detection
Usage:
  .\.venv\Scripts\python.exe run_haar_fallback.py --input 0
"""
import argparse
import cv2

parser = argparse.ArgumentParser()
parser.add_argument('--input', '-i', default='img/oscars_360x640.gif', help='webcam index, video URL, or image file')
args = parser.parse_args()

src = args.input
# Check if it's an image file first
if src.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
    # Handle image file
    frame = cv2.imread(src)
    if frame is None:
        print('Unable to load image', src)
        exit(1)
    
    cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    if not cv2.os.path.exists(cascade_path):
        print('Haar cascade not found at', cascade_path)
        exit(1)
    
    detector = cv2.CascadeClassifier(cascade_path)
    print('Processing image with Haar cascade...')
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30,30))
    
    print(f'Detected {len(faces)} faces')
    for (x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
    
    # Save result
    output_path = 'haar_image_result.jpg'
    cv2.imwrite(output_path, frame)
    print(f'Result saved as {output_path}')
    
    # Try to display
    try:
        cv2.imshow('Haar fallback - Image', frame)
        print('Press any key to close...')
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    except:
        print('Display not available, result saved to file.')
    exit(0)

# Handle video/webcam
if str(src).isdigit():
    camera_id = int(src)
    print(f'Attempting to open camera {camera_id}...')
    
    # Try AVFoundation backend first (best for macOS)
    try:
        cap = cv2.VideoCapture(camera_id, cv2.CAP_AVFOUNDATION)
        if not cap.isOpened():
            print('AVFoundation failed, trying default backend...')
            cap = cv2.VideoCapture(camera_id)
    except:
        print('AVFoundation not available, using default backend...')
        cap = cv2.VideoCapture(camera_id)
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
